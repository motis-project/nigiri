#include "nigiri/routing/lb_raptor.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/timing.h"
#include "utl/zip.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <direction SearchDir>
void lb_raptor(timetable const& tt, query const& q, lb_raptor_state& state) {
  auto const& adjacency =
      (SearchDir == direction::kForward ? tt.fwd_lb_adjacency_
                                        : tt.bwd_lb_adjacency_)[q.prf_idx_];
  auto const& footpaths = (SearchDir == direction::kForward
                               ? tt.locations_.footpaths_in_
                               : tt.locations_.footpaths_out_)[q.prf_idx_];

  state.resize(tt.n_locations());
  state.clear();

  // init (k = 0)
  std::map<location_idx_t, std::uint16_t> min;
  auto const update_min = [&](location_idx_t const l, std::uint16_t const t) {
    auto const r = tt.locations_.get_root_idx(l);
    auto& m = utl::get_or_create(
        min, r, [&] { return state.location_round_lb_[r][0]; });
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
      state.location_round_lb_[meta].fill(std::min(
          t, state.location_round_lb_[meta][0]));  // necessary to min again?
      state.station_mark_.set(to_idx(meta), true);
    });
  }
  for (auto const& s : q.start_) {
    state.is_start_.set(to_idx(s.target()), true);
  }
  for (auto const& [l, _] : q.td_start_) {
    state.is_start_.set(to_idx(l), true);
  }

  // run
  for (auto k = 1U; k != std::min(q.max_transfers_, kMaxTransfers) + 2U; ++k) {
    std::swap(state.prev_station_mark_, state.station_mark_);
    utl::fill(state.station_mark_.blocks_, 0U);

    auto any_marked = false;
    state.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      auto const l = location_idx_t{i};

      auto const visit = [&](lb_neighbor const n) {
        auto const lb = static_cast<std::uint16_t>(
            state.location_round_lb_[l][k - 1U] + n.pt_duration_);

        if (state.is_start_.test(to_idx(n.l_)) &&
            lb < state.location_round_lb_[n.l_][k]) [[unlikely]] {
          std::fill(begin(state.location_round_lb_[n.l_]) + k,
                    end(state.location_round_lb_[n.l_]), lb);
          return;
        }

        auto const lb_transfer = static_cast<std::uint16_t>(
            lb +
            adjusted_transfer_time(q.transfer_time_settings_,
                                   tt.locations_.transfer_time_[n.l_].count()));
        if (lb_transfer < state.location_round_lb_[n.l_][k]) {
          std::fill(begin(state.location_round_lb_[n.l_]) + k,
                    end(state.location_round_lb_[n.l_]), lb_transfer);
          state.station_mark_.set(to_idx(n.l_), true);
          any_marked = true;

          for (auto const fp : footpaths[n.l_]) {
            if (state.is_start_.test(to_idx(fp.target()))) [[unlikely]] {
              continue;
            }

            auto const lb_fp = static_cast<std::uint16_t>(
                lb + adjusted_transfer_time(q.transfer_time_settings_,
                                            fp.duration().count()));
            if (lb_fp < state.location_round_lb_[fp.target()][k]) {
              std::fill(begin(state.location_round_lb_[fp.target()]) + k,
                        end(state.location_round_lb_[fp.target()]), lb_fp);
              state.station_mark_.set(to_idx(fp.target()), true);
            }
          }
        }
      };

      for (auto const n : adjacency[l]) {
        visit(n);
      }
    });

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
}

template void lb_raptor<direction::kForward>(timetable const&,
                                             query const&,
                                             lb_raptor_state&);

template void lb_raptor<direction::kBackward>(timetable const&,
                                              query const&,
                                              lb_raptor_state&);

}  // namespace nigiri::routing
