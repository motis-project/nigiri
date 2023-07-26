#include "nigiri/routing/reach.h"

#include <filesystem>

#include "boost/thread/tss.hpp"

#include "geo/box.h"

#include "utl/equal_ranges_linear.h"
#include "utl/erase_if.h"
#include "utl/parallel_for.h"
#include "utl/parser/split.h"
#include "utl/progress_tracker.h"

#include "cista/mmap.h"
#include "cista/serialization.h"

#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

namespace fs = std::filesystem;

namespace nigiri::routing {

constexpr auto const kMode =
    cista::mode::WITH_INTEGRITY | cista::mode::WITH_STATIC_VERSION;

struct state {
  explicit state(timetable const& tt, date::sys_days const base_day)
      : tt_{tt},
        results_(tt.n_locations()),
        base_day_{tt.day_idx(base_day)},
        lb_(tt.n_locations(), 0U),
        is_dest_(tt.n_locations(), false),
        dist_to_dest_(tt.n_locations(), kInvalidDelta<direction::kForward>) {}

  void reset() {
    for (auto& r : results_) {
      r.clear();
    }

    raptor_.reset_arrivals();
    starts_.clear();
  }

  timetable const& tt_;

  std::vector<start> starts_;
  std::vector<pareto_set<journey>> results_{tt_.n_locations()};

  day_idx_t base_day_;
  std::vector<bool> route_filtered_;
  std::vector<std::uint16_t> lb_;
  std::vector<bool> is_dest_;
  std::vector<std::uint16_t> dist_to_dest_;
  raptor_state raptor_state_;
  journey reconstruct_journey_;
  raptor<direction::kForward, /* Rt= */ false, /* OneToAll= */ true> raptor_{
      tt_,      nullptr,       raptor_state_, route_filtered_,
      is_dest_, dist_to_dest_, lb_,           base_day_};
};

static boost::thread_specific_ptr<state> search_state;

reach_info::reach_info() = default;

void reach_info::update(float const new_reach,
                        routing::journey const&,
                        location_idx_t const start_end,
                        location_idx_t const stop_in_route) {
  auto const lck = std::scoped_lock{mutex_};
  if (new_reach > reach_) {
    //    j_ = j;
    reach_ = new_reach;
    stop_in_route_ = stop_in_route;
    start_end_ = start_end;
  }
}

void update_route_reachs(timetable const& tt,
                         journey const& j,
                         std::vector<reach_info>& route_reachs) {
  for (auto const& l : j.legs_) {
    if (!std::holds_alternative<journey::run_enter_exit>(l.uses_)) {
      continue;
    }

    auto const ree = std::get<journey::run_enter_exit>(l.uses_);
    auto const r = tt.transport_route_[ree.r_.t_.t_idx_];

    auto const start = tt.locations_.coordinates_[j.legs_.front().from_];
    auto const dest = tt.locations_.coordinates_[j.legs_.back().to_];

    auto const fr = rt::frun{tt, nullptr, ree.r_};

    auto& reach = route_reachs[to_idx(r)];
    for (auto i = ree.stop_range_.from_; i != ree.stop_range_.to_; ++i) {
      auto const stp = tt.locations_.coordinates_[fr[i].get_location_idx()];
      auto const new_reach = static_cast<float>(
          std::min(geo::distance(start, stp), geo::distance(stp, dest)));
      auto const stop_in_route = fr[i].get_location_idx();
      auto const start_end =
          geo::distance(start, stp) < geo::distance(stp, dest)
              ? j.legs_.front().from_
              : j.legs_.back().to_;
      reach.update(new_reach, j, start_end, stop_in_route);
    }
  }
}

void reach_values_for_source(timetable const& tt,
                             date::sys_days const base_day,
                             interval<date::sys_days> search_interval,
                             location_idx_t const l,
                             std::vector<reach_info>& route_reachs) {
  if (search_state.get() == nullptr) {
    search_state.reset(new state{tt, base_day});
  }

  auto& state = *search_state;

  auto q = query{};
  q.start_match_mode_ = location_match_mode::kEquivalent;
  q.dest_match_mode_ = location_match_mode::kEquivalent;
  q.start_ = {offset{location_idx_t{l}, 0_minutes, 0U}};

  get_starts(direction::kForward, tt, nullptr, search_interval, q.start_,
             location_match_mode::kEquivalent, true, state.starts_, false);

  utl::equal_ranges_linear(
      state.starts_,
      [](start const& a, start const& b) {
        return a.time_at_start_ == b.time_at_start_;
      },
      [&](auto&& from_it, auto&& to_it) {
        state.raptor_.next_start_time();
        auto const start_time = from_it->time_at_start_;

        q.start_time_ = start_time;

        for (auto const& st : it_range{from_it, to_it}) {
          state.raptor_.add_start(st.stop_, st.time_at_stop_);
        }

        auto const worst_time_at_dest = start_time + kMaxTravelTime;
        state.raptor_.execute(start_time, kMaxTransfers, worst_time_at_dest,
                              state.results_[to_idx(l)]);

        // Reconstruct for each target.
        for (auto t = location_idx_t{0U}; t != tt.n_locations(); ++t) {
          if (t == l) {
            continue;
          }

          // Collect journeys for each number of transfers.
          for (auto k = 1U; k != kMaxTransfers + 1U; ++k) {
            auto const dest_time =
                state.raptor_state_.round_times_[k][to_idx(t)];
            if (dest_time == kInvalidDelta<direction::kForward>) {
              continue;
            }
            state.results_[to_idx(t)].add(journey{
                .legs_ = {},
                .start_time_ = start_time,
                .dest_time_ = delta_to_unix(state.raptor_.base(), dest_time),
                .dest_ = t,
                .transfers_ = static_cast<std::uint8_t>(k - 1)});
          }

          // Reconstruct journeys and update reach values.
          for (auto& j : state.results_[to_idx(t)]) {
            if (j.reconstructed_) {
              continue;
            }
            q.destination_ = {{t, 0_minutes, 0U}};
            state.reconstruct_journey_.copy_from(j);
            state.raptor_.reconstruct(q, state.reconstruct_journey_);
            update_route_reachs(tt, state.reconstruct_journey_, route_reachs);
            j.reconstructed_ = true;
          }
        }
      });
}

std::vector<reach_info> compute_reach_values(
    timetable const& tt,
    std::vector<location_idx_t> const& source_locations,
    interval<date::sys_days> const search_interval) {
  auto route_reachs = std::vector<reach_info>(tt.n_routes());

  auto progress_tracker = utl::activate_progress_tracker("reach");
  progress_tracker->out_bounds(0, 100);
  progress_tracker->in_high(source_locations.size());
  utl::parallel_for(
      source_locations,
      [&](location_idx_t const l) {
        reach_values_for_source(tt, search_interval.from_, search_interval, l,
                                route_reachs);
      },
      progress_tracker->increment_fn());
  return route_reachs;
}

std::pair<std::vector<unsigned>, float> get_separation_fn(
    timetable const& tt,
    std::vector<reach_info> const& route_reachs,
    double const reach_factor,
    double const outlier_percent) {
  auto perm = std::vector<unsigned>{};
  perm.resize(route_reachs.size());
  std::generate(begin(perm), end(perm), [i = 0U]() mutable { return i++; });

  utl::erase_if(perm, [&](unsigned i) { return !route_reachs[i].valid(); });

  utl::sort(perm, [&](unsigned const a, unsigned const b) {
    return tt.route_bbox_diagonal_[route_idx_t{a}] -
               reach_factor * route_reachs[a].reach_ <
           tt.route_bbox_diagonal_[route_idx_t{b}] -
               reach_factor * route_reachs[b].reach_;
  });

  for (auto const idx : perm) {
    auto const t = tt.route_transport_ranges_[route_idx_t{idx}][0];
    auto const [type, name] =
        utl::split<' ', utl::cstr, utl::cstr>(tt.transport_name(t));
    std::cout << "  reach=" << route_reachs[idx].reach_
              << ", bbox_diagonal=" << tt.route_bbox_diagonal_[route_idx_t{idx}]
              << ", sum="
              << (tt.route_bbox_diagonal_[route_idx_t{idx}] -
                  reach_factor * route_reachs[idx].reach_)
              << ", name=" << name.view() << "\n";
  }

  auto const last_idx = static_cast<std::size_t>(
      std::round(static_cast<double>(route_reachs.size()) * outlier_percent));
  auto const fn = tt.route_bbox_diagonal_[route_idx_t{perm[last_idx]}] -
                  (route_reachs[perm[last_idx]].reach_ * reach_factor);

  perm.resize(last_idx);

  return {perm, fn};
}

using reach_values_vec_t = vector_map<route_idx_t, unsigned>;

void write_reach_values(timetable const& tt,
                        float const y,
                        float const x_slope,
                        std::vector<reach_info> const& route_reachs,
                        fs::path const& path) {
  auto reach_values = vector_map<route_idx_t, unsigned>{};
  reach_values.resize(tt.n_routes());
  for (auto r = 0U; r != tt.n_routes(); ++r) {
    reach_values[route_idx_t{r}] = std::max(
        route_reachs[r].reach_,
        std::ceil(y + x_slope * tt.route_bbox_diagonal_[route_idx_t{r}]));
  }

  auto mmap =
      cista::mmap{path.string().c_str(), cista::mmap::protection::WRITE};
  auto writer = cista::buf<cista::mmap>(std::move(mmap));
  cista::serialize<kMode>(writer, reach_values);
}

cista::wrapped<reach_values_vec_t> read_reach_values(
    cista::memory_holder&& mem) {
  return std::visit(
      utl::overloaded{[&](cista::buf<cista::mmap>& b) {
                        auto const ptr = reinterpret_cast<reach_values_vec_t*>(
                            &b[cista::data_start(kMode)]);
                        return cista::wrapped{std::move(mem), ptr};
                      },
                      [&](cista::buffer& b) {
                        auto const ptr =
                            cista::deserialize<reach_values_vec_t, kMode>(b);
                        return cista::wrapped{std::move(mem), ptr};
                      },
                      [&](cista::byte_buf& b) {
                        auto const ptr =
                            cista::deserialize<reach_values_vec_t, kMode>(b);
                        return cista::wrapped{std::move(mem), ptr};
                      }},
      mem);
}

}  // namespace nigiri::routing