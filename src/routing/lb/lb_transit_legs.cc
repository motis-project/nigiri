#include "nigiri/routing/lb/lb_transit_legs.h"

namespace nigiri::routing {

template <direction SearchDir>
void lb_transit_legs_round(timetable const& tt,
                           query const& q,
                           lb_transit_legs_state& state,
                           std::uint8_t const k) {
  constexpr auto kFwd = SearchDir == direction::kForward;
 auto const start_time = std::chrono::steady_clock::now();

  utl::fill(state.route_mark_.blocks_, 0U);

  state.any_marked_ = false;
  state.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
    for (auto const r : tt.location_routes_[location_idx_t{i}]) {
      state.any_marked_ = true;
      state.route_mark_.set(to_idx(r), true);
    }
  });
  if (!state.any_marked_) {
    return;
  }

  std::swap(state.prev_station_mark_, state.station_mark_);
  utl::fill(state.station_mark_.blocks_, 0U);

  state.any_marked_ = false;
  state.route_mark_.for_each_set_bit([&](std::uint64_t const i) {
    auto const r = route_idx_t{i};
    auto const& seq = tt.route_location_seq_[r];
    for (auto x = 0U; x != seq.size(); ++x) {
      auto const in = kFwd ? seq.size() - x - 1U : x;
      auto const l_in = stop{seq[in]}.location_idx();

      if (!state.prev_station_mark_.test(to_idx(l_in))) {
        continue;
      }

      auto const step = [&](auto const y) { return kFwd ? y - 1U : y + 1U; };
      for (auto out = step(in); out < seq.size(); out = step(out)) {
        auto const l_out = stop{seq[out]}.location_idx();
        if (k < state.lb_[l_out]) {
          state.lb_[l_out] = k;
          state.station_mark_.set(to_idx(l_out), true);
          state.any_marked_ = true;
          for (auto const fp :
               kFwd ? tt.locations_.footpaths_in_[q.prf_idx_][l_out]
                    : tt.locations_.footpaths_out_[q.prf_idx_][l_out]) {
            if (k < state.lb_[fp.target()]) {
              state.lb_[fp.target()] = k;
              state.station_mark_.set(to_idx(fp.target()), true);
            }
          }
        } else if (k > state.lb_[l_out]) {
          break;
        }
      }
    }
  });

  fmt::println("[lb_transit_legs_round] ran for k = {} which took {}", k,
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start_time));
}

template void lb_transit_legs_round<direction::kForward>(timetable const&,
                                                         query const&,
                                                         lb_transit_legs_state&,
                                                         std::uint8_t);
template void lb_transit_legs_round<direction::kBackward>(
    timetable const&, query const&, lb_transit_legs_state&, std::uint8_t);

}  // namespace nigiri::routing