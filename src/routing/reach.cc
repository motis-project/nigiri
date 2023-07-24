#include "nigiri/routing/reach.h"

#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"
#include "utl/equal_ranges_linear.h"

namespace nigiri::routing {

vector_map<route_idx_t, reach_info> get_reach_values(
    timetable const& tt,
    std::vector<location_idx_t> const& source_locations,
    interval<unixtime_t> const interval) {
  auto route_reachs = vector_map<route_idx_t, reach_info>{};
  route_reachs.resize(tt.n_routes());

  auto const update_route_reachs = [&](journey const& j) {
    for (auto const& l : j.legs_) {
      if (!std::holds_alternative<journey::run_enter_exit>(l.uses_)) {
        continue;
      }

      auto const ree = std::get<journey::run_enter_exit>(l.uses_);
      auto const r = tt.transport_route_[ree.r_.t_.t_idx_];

      auto const start = tt.locations_.coordinates_[j.legs_.front().from_];
      auto const dest = tt.locations_.coordinates_[j.legs_.back().to_];

      auto const fr = rt::frun{tt, nullptr, ree.r_};

      auto& reach = route_reachs[r];
      for (auto i = ree.stop_range_.from_; i != ree.stop_range_.to_; ++i) {
        auto const stp = tt.locations_.coordinates_[fr[i].get_location_idx()];
        auto const new_reach =
            std::min(geo::distance(start, stp), geo::distance(stp, dest));

        if (new_reach > reach.reach_) {
          reach.j_ = j;
          reach.reach_ = new_reach;
          reach.stop_in_route_ = fr[i].get_location_idx();
          if (geo::distance(start, stp) < geo::distance(stp, dest)) {
            reach.start_end_ = j.legs_.front().from_;
          } else {
            reach.start_end_ = j.legs_.back().to_;
          }
        }
      }
    }
  };

  auto s = raptor_state{};
  auto dist_to_dest = std::vector<std::uint16_t>(
      tt.n_locations(), kInvalidDelta<direction::kForward>);
  auto lb = std::vector<std::uint16_t>(tt.n_locations(), 0U);
  auto is_dest = std::vector<bool>(tt.n_locations());
  // TODO(felix) iterate timetable and change base day
  auto const base_day = tt.day_idx(tt.internal_interval_days().from_);
  auto r = raptor<direction::kForward, false, true>{
      tt, nullptr, s, is_dest, dist_to_dest, lb, base_day};
  auto starts = std::vector<start>{};
  auto results = std::vector<pareto_set<journey>>{};
  results.resize(tt.n_locations());

  for (auto const l : source_locations) {
    results.clear();
    results.resize(tt.n_locations());

    r.reset_arrivals();
    starts.clear();

    auto q = query{};
    q.start_match_mode_ = location_match_mode::kEquivalent;
    q.dest_match_mode_ = location_match_mode::kEquivalent;
    q.start_ = {offset{location_idx_t{l}, 0_minutes, 0U}};

    get_starts(direction::kForward, tt, nullptr, interval, q.start_,
               location_match_mode::kEquivalent, true, starts, false);

    utl::equal_ranges_linear(
        starts,
        [](start const& a, start const& b) {
          return a.time_at_start_ == b.time_at_start_;
        },
        [&](auto&& from_it, auto&& to_it) {
          r.next_start_time();
          auto const start_time = from_it->time_at_start_;

          q.start_time_ = start_time;

          for (auto const& st : it_range{from_it, to_it}) {
            r.add_start(st.stop_, st.time_at_stop_);
          }

          auto const worst_time_at_dest = start_time + kMaxTravelTime;
          r.execute(start_time, kMaxTransfers, worst_time_at_dest,
                    results[to_idx(l)]);

          // Reconstruct for each target.
          for (auto t = location_idx_t{0U}; t != tt.n_locations(); ++t) {
            if (t == l) {
              continue;
            }

            // Collect journeys for each number of transfers.
            for (auto k = 1U; k != kMaxTransfers + 1U; ++k) {
              auto const dest_time = s.round_times_[k][to_idx(t)];
              if (dest_time == kInvalidDelta<direction::kForward>) {
                continue;
              }
              results[to_idx(t)].add(
                  journey{.legs_ = {},
                          .start_time_ = start_time,
                          .dest_time_ = delta_to_unix(r.base(), dest_time),
                          .dest_ = t,
                          .transfers_ = static_cast<std::uint8_t>(k - 1)});
            }

            // Reconstruct journeys and update reach values.
            for (auto& j : results[to_idx(t)]) {
              if (!j.legs_.empty()) {
                continue;
              }
              q.destination_ = {{t, 0_minutes, 0U}};
              r.reconstruct(q, j);
              update_route_reachs(j);
            }
          }
        });
  }

  return route_reachs;
}

}  // namespace nigiri::routing