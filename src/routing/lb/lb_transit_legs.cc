#include "nigiri/routing/lb/lb_transit_legs.h"

#include "nigiri/for_each_meta.h"

namespace nigiri::routing {

constexpr auto kUnreachable = std::numeric_limits<std::uint8_t>::max();
constexpr auto kUnknown = kUnreachable - 1U;

lb_transit_legs::lb_transit_legs(timetable const& tt,
                                 query const& q,
                                 lb_transit_legs_state& state)
    : tt_{tt}, q_{q}, state_{state}, k_{0U} {
  state_.station_mark_.resize(tt_.n_locations());
  utl::fill(state_.station_mark_.blocks_, 0U);
  state_.prev_station_mark_.resize(tt_.n_locations());
  state_.route_mark_.resize(tt_.n_routes());
  state_.lb_.resize(tt_.n_locations());
  utl::fill(state_.lb_, kUnknown);

  auto const set_terminal = [&](auto const i) {
    state_.station_mark_.set(to_idx(i), true);
    state_.lb_[i] = 0U;
  };

  for (auto const& o : q_.destination_) {
    for_each_meta(tt_, q_.dest_match_mode_, o.target(),
                  [&](location_idx_t const meta) { set_terminal(meta); });
  }

  for (auto const& [l, tds] : q_.td_dest_) {
    for (auto const& td : tds) {
      if (td.duration() != footpath::kMaxDuration &&
          td.duration() < q_.max_travel_time_) {
        for_each_meta(tt, q_.dest_match_mode_, l,
                      [&](location_idx_t const meta) { set_terminal(meta); });
      }
    }
  }

  k_ = 1U;
}

template <direction SearchDir>
void lb_transit_legs(timetable const& tt,
                     query const& q,
                     lb_transit_legs_state& state) {
  constexpr auto kFwd = SearchDir == direction::kForward;

  auto const start_time = std::chrono::steady_clock::now();

  // run
  auto k = std::uint8_t{1U};
  for (; k != std::min(q.max_transfers_, kMaxTransfers) + 2U; ++k) {
    utl::fill(state.route_mark_.blocks_, 0U);

    auto any_marked = false;
    state.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      for (auto const r : tt.location_routes_[location_idx_t{i}]) {
        any_marked = true;
        state.route_mark_.set(to_idx(r), true);
      }
    });
    if (!any_marked) {
      break;
    }

    std::swap(state.prev_station_mark_, state.station_mark_);
    utl::fill(state.station_mark_.blocks_, 0U);

    any_marked = false;
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
            any_marked = true;
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
    if (!any_marked) {
      break;
    }
  }

  fmt::println("all stations reached after {} rounds, {}", k,
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start_time));
}

template void lb_transit_legs<direction::kForward>(timetable const&,
                                                   query const&,
                                                   lb_transit_legs_state&);
template void lb_transit_legs<direction::kBackward>(timetable const&,
                                                    query const&,
                                                    lb_transit_legs_state&);

}  // namespace nigiri::routing