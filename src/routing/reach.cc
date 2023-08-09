#include "nigiri/routing/reach.h"

#include <filesystem>
#include <random>

#include "boost/iterator/function_output_iterator.hpp"
#include "boost/thread/tss.hpp"

#include "geo/box.h"
#include "geo/point_rtree.h"

#include "utl/equal_ranges_linear.h"
#include "utl/erase_if.h"
#include "utl/parallel_for.h"
#include "utl/parser/split.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "cista/mmap.h"
#include "cista/serialization.h"

#include "nigiri/routing/no_route_filter.h"
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
  std::vector<std::uint16_t> lb_;
  std::vector<bool> is_dest_;
  std::vector<std::uint16_t> dist_to_dest_;
  raptor_state raptor_state_;
  journey reconstruct_journey_;
  no_route_filter no_route_filter_;
  raptor<no_route_filter,
         direction::kForward,
         /* Rt= */ false,
         /* OneToAll= */ true>
      raptor_{no_route_filter_, tt_,           nullptr, raptor_state_,
              is_dest_,         dist_to_dest_, lb_,     base_day_};
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
  state.reset();

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
            try {
              state.raptor_.reconstruct(q, state.reconstruct_journey_);
            } catch (std::exception const& e) {
              log(log_lvl::info, "routing.reach",
                  "reconsturct {}@{} to {}@{}: {}", location{tt, l},
                  j.start_time_, location{tt, t}, j.dest_time_, e.what());
              continue;
            }
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

using reach_values_vec_t = vector_map<route_idx_t, std::uint32_t>;

void compute_reach_values(timetable& tt, unsigned n_reach_queries) {
  auto source_locations = std::vector<location_idx_t>{};
  source_locations.resize(tt.n_locations());
  std::generate(begin(source_locations), end(source_locations),
                [i = location_idx_t{0U}]() mutable { return i++; });

  auto rd = std::random_device{};
  auto g = std::mt19937{rd()};
  std::shuffle(begin(source_locations), end(source_locations), g);
  source_locations.resize(n_reach_queries);

  auto const x_slope = .8F;
  auto const route_reachs = routing::compute_reach_values(
      tt, source_locations, tt.internal_interval_days());
  auto const [perm, y] =
      routing::get_separation_fn(tt, route_reachs, x_slope, 0.08);

  tt.route_reachs_.resize(tt.n_routes());
  for (auto r = 0U; r != tt.n_routes(); ++r) {
    if (route_reachs[r].valid()) {
      tt.route_reachs_[route_idx_t{r}] = route_reachs[r].reach_ * 1.1;
    } else {
      tt.route_reachs_[route_idx_t{r}] =
          std::ceil(y + x_slope * tt.route_bbox_diagonal_[route_idx_t{r}]) *
          1.1;
    }
    //    reach_values[route_idx_t{r}] = std::max(
    //        route_reachs[r].reach_,
    //        std::ceil(y + x_slope * tt.route_bbox_diagonal_[route_idx_t{r}]));
  }

  auto const connecting_routes = routing::get_big_station_connection_routes(tt);
  for (auto const r : connecting_routes) {
    tt.route_reachs_[r] = 1'000'000;
  }
}

std::vector<route_idx_t> get_big_station_connection_routes(
    timetable const& tt) {
  auto perm = std::vector<unsigned>{};
  perm.resize(tt.n_routes());

  std::generate(begin(perm), end(perm), [i = 0U]() mutable { return i++; });

  utl::sort(perm, [&](unsigned const a, unsigned const b) {
    return tt.route_bbox_diagonal_[route_idx_t{a}] <
           tt.route_bbox_diagonal_[route_idx_t{b}];
  });

  perm.resize(0.05 * tt.n_routes());

  auto locations = hash_set<location_idx_t>{};
  for (auto const r : perm) {
    for (auto const stp : tt.route_location_seq_[route_idx_t{r}]) {
      locations.emplace(stop{stp}.location_idx());
    }
  }

  auto const& locations_ordered = locations.values();
  auto const rtree = geo::make_point_rtree(utl::to_vec(
      locations_ordered,
      [&](location_idx_t const l) { return tt.locations_.coordinates_[l]; }));

  auto connecting_routes = hash_set<route_idx_t>{};
  for (auto const l : locations_ordered) {
    for (auto const close_l_idx :
         rtree.in_radius(tt.locations_.coordinates_[l], 10'000)) {
      auto const close_l = locations_ordered[close_l_idx];
      if (close_l == l) {
        continue;
      }

      auto const routes_a = tt.location_routes_[l];
      auto const routes_b = tt.location_routes_[close_l];

      std::set_intersection(
          begin(routes_a), end(routes_a), begin(routes_b), end(routes_b),
          boost::make_function_output_iterator(
              [&](route_idx_t const cr) { connecting_routes.emplace(cr); }));
    }
  }

  return utl::to_vec(connecting_routes.values(),
                     [](route_idx_t const r) { return r; });
}

}  // namespace nigiri::routing