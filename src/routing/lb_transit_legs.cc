#include "nigiri/routing/lb_transit_legs.h"

#include "nigiri/for_each_meta.h"

namespace nigiri::routing {

template <direction SearchDir>
void lb_transit_legs(timetable const& tt,
                     query const& q,
                     raptor_state& state,
                     std::vector<std::uint8_t>& lb) {
  constexpr auto kFwd = SearchDir == direction::kForward;

  auto const start_time = std::chrono::steady_clock::now();

  // init
  state.station_mark_.resize(tt.n_locations());
  utl::fill(state.station_mark_.blocks_, 0U);
  state.prev_station_mark_.resize(tt.n_locations());
  state.route_mark_.resize(tt.n_routes());
  lb.resize(tt.n_locations());
  utl::fill(lb, std::numeric_limits<std::uint8_t>::max());

  // k = 0
  auto const set_terminal = [&](auto const i) {
    state.station_mark_.set(i, true);
    lb[i] = 0U;
  };

  for (auto const& o : q.destination_) {
    for_each_meta(
        tt, q.dest_match_mode_, o.target(),
        [&](location_idx_t const meta) { set_terminal(to_idx(meta)); });
  }

  for (auto const& [l, tds] : q.td_dest_) {
    for (auto const& td : tds) {
      if (td.duration() != footpath::kMaxDuration &&
          td.duration() < q.max_travel_time_) {
        for_each_meta(
            tt, q.dest_match_mode_, l,
            [&](location_idx_t const meta) { set_terminal(to_idx(meta)); });
      }
    }
  }

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
          if (k < lb[to_idx(l_out)]) {
            lb[to_idx(l_out)] = k;
            state.station_mark_.set(to_idx(l_out), true);
            any_marked = true;
            for (auto const fp :
                 kFwd ? tt.locations_.footpaths_in_[q.prf_idx_][l_out]
                      : tt.locations_.footpaths_out_[q.prf_idx_][l_out]) {
              if (k < lb[to_idx(fp.target())]) {
                lb[to_idx(fp.target())] = k;
                state.station_mark_.set(to_idx(fp.target()), true);
              }
            }

          } else if (k > lb[to_idx(l_out)]) {
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
                                                   raptor_state&,
                                                   std::vector<std::uint8_t>&);
template void lb_transit_legs<direction::kBackward>(timetable const&,
                                                    query const&,
                                                    raptor_state&,
                                                    std::vector<std::uint8_t>&);

}  // namespace nigiri::routing