#include "nigiri/routing/lb_raptor.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/timing.h"
#include "utl/zip.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <direction SearchDir>
void lb_raptor(
    timetable const& tt,
    query const& q,
    raptor_state& state,
    vector_map<location_idx_t, std::array<std::uint16_t, kMaxTransfers + 2U>>&
        location_round_lb) {
  std::cout << std::endl << std::endl;
  UTL_START_TIMING(lb_raptor);
  auto const& adjacency =
      (SearchDir == direction::kForward ? tt.fwd_lb_adjacency_
                                        : tt.bwd_lb_adjacency_)[q.prf_idx_];
  auto const& footpaths = (SearchDir == direction::kForward
                               ? tt.locations_.footpaths_in_
                               : tt.locations_.footpaths_out_)[q.prf_idx_];

  // resize & clear
  state.prev_station_mark_.resize(tt.n_locations());
  state.station_mark_.resize(tt.n_locations());
  utl::fill(state.station_mark_.blocks_, 0U);
  location_round_lb.resize(tt.n_locations());
  static constexpr auto kRoundLbInit = []() {
    auto ret = std::array<std::uint16_t, kMaxTransfers + 2>{};
    ret.fill(std::numeric_limits<std::uint16_t>::max());
    return ret;
  }();
  utl::fill(location_round_lb, kRoundLbInit);

  // init (k = 0)
  std::map<location_idx_t, std::uint16_t> min;
  auto const update_min = [&](location_idx_t const l, std::uint16_t const t) {
    auto const r = tt.locations_.get_root_idx(l);
    auto& m =
        utl::get_or_create(min, r, [&]() { return location_round_lb[r][0]; });
    m = std::min(t, m);
  };
  for (auto const& o : q.destination_) {
    for_each_meta(
        tt, q.dest_match_mode_, o.target(), [&](location_idx_t const l) {
          update_min(l, static_cast<std::uint16_t>(o.duration().count()));
        });
  }
  for (auto const& [l, tds] : q.td_dest_) {
    for (auto const& td : tds) {
      if (td.duration() != footpath::kMaxDuration &&
          td.duration() < q.max_travel_time_) {
        update_min(l, static_cast<std::uint16_t>(td.duration().count()));
      }
    }
  }
  for (auto const& [l, t] : min) {
    for_each_meta(tt, q.dest_match_mode_, l, [&](location_idx_t const meta) {
      location_round_lb[meta].fill(
          std::min(t, location_round_lb[meta][0]));  // necessary to min again?
      state.station_mark_.set(to_idx(meta), true);
    });
  }

  // run
  auto k = 1U;
  for (; k != kMaxTransfers + 2U; ++k) {
    UTL_START_TIMING(lb_raptor_round);
    std::swap(state.prev_station_mark_, state.station_mark_);
    utl::fill(state.station_mark_.blocks_, 0U);

    auto any_marked = false;
    state.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      auto const l = location_idx_t{i};

      auto const visit = [&](lb_neighbor const n) {
        auto const pt_lb =
            static_cast<std::uint16_t>(location_round_lb[l][k - 1U] + n.dist_);
        if (pt_lb < location_round_lb[n.l_][k]) {
          any_marked = true;
          std::fill(begin(location_round_lb[n.l_]) + k,
                    end(location_round_lb[n.l_]), pt_lb);
          state.station_mark_.set(to_idx(n.l_), true);
          for (auto const fp : footpaths[n.l_]) {
            auto const walk_lb =
                static_cast<std::uint16_t>(pt_lb + fp.duration().count());
            if (walk_lb < location_round_lb[fp.target()][k]) {
              std::fill(begin(location_round_lb[fp.target()]) + k,
                        end(location_round_lb[fp.target()]), walk_lb);
              state.station_mark_.set(to_idx(fp.target()), true);
            }
          }
        }
      };

      for (auto const n : adjacency[l]) {
        visit(n);
      }
    });
    UTL_STOP_TIMING(lb_raptor_round);
    fmt::println("lb_raptor_round, k: {}, #marked: {}, time: {} ms", k,
                 state.prev_station_mark_.count(),
                 UTL_TIMING_MS(lb_raptor_round));

    if (!any_marked) {
      break;
    }
  }

  // propagate lb to children
  // for (auto const l :
  //      interval{location_idx_t{0}, location_idx_t{tt.n_locations()}}) {
  //   for (auto const c : tt.locations_.children_[l]) {
  //     for (auto [plb, clb] :
  //          utl::zip(location_round_lb[l], location_round_lb[c])) {
  //       clb = std::min(plb, clb);
  //     }
  //   }
  // }

  UTL_STOP_TIMING(lb_raptor);
  fmt::println("lb_raptor k: {}, time: {} ms", k, UTL_TIMING_MS(lb_raptor));
  std::cout << std::endl << std::endl;
}

template void lb_raptor<direction::kForward>(
    timetable const&,
    query const&,
    raptor_state&,
    vector_map<location_idx_t, std::array<std::uint16_t, kMaxTransfers + 2U>>&);

template void lb_raptor<direction::kBackward>(
    timetable const&,
    query const&,
    raptor_state&,
    vector_map<location_idx_t, std::array<std::uint16_t, kMaxTransfers + 2U>>&);

}  // namespace nigiri::routing
