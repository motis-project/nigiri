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
void lb_raptor(
    timetable const& tt,
    query const& q,
    bitvec& station_mark,
    bitvec& prev_station_mark,
    bitvec& is_start,
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
  prev_station_mark.resize(tt.n_locations());
  station_mark.resize(tt.n_locations());
  utl::fill(station_mark.blocks_, 0U);
  location_round_lb.resize(tt.n_locations());
  static constexpr auto kRoundLbInit = []() {
    auto ret = std::array<std::uint16_t, kMaxTransfers + 2>{};
    ret.fill(std::numeric_limits<std::uint16_t>::max());
    return ret;
  }();
  utl::fill(location_round_lb, kRoundLbInit);
  is_start.resize(tt.n_locations());
  utl::fill(is_start.blocks_, 0U);

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
      station_mark.set(to_idx(meta), true);
    });
  }
  for (auto const& s : q.start_) {
    is_start.set(to_idx(s.target()), true);
  }
  for (auto const& [l, _] : q.td_start_) {
    is_start.set(to_idx(l), true);
  }

  // run
  auto k = 1U;
  for (; k != kMaxTransfers + 2U; ++k) {
    UTL_START_TIMING(lb_raptor_round);
    std::swap(prev_station_mark, station_mark);
    utl::fill(station_mark.blocks_, 0U);

    auto any_marked = false;
    prev_station_mark.for_each_set_bit([&](std::uint64_t const i) {
      auto const l = location_idx_t{i};

      auto const visit = [&](lb_neighbor const n) {
        auto const lb =
            static_cast<std::uint16_t>(location_round_lb[l][k - 1U] + n.dist_);

        if (is_start.test(to_idx(n.l_)) && lb < location_round_lb[n.l_][k])
            [[unlikely]] {
          std::fill(begin(location_round_lb[n.l_]) + k,
                    end(location_round_lb[n.l_]), lb);
          return;
        }

        auto const lb_transfer = static_cast<std::uint16_t>(
            lb +
            adjusted_transfer_time(q.transfer_time_settings_,
                                   tt.locations_.transfer_time_[n.l_].count()));
        if (lb_transfer < location_round_lb[n.l_][k]) {
          std::fill(begin(location_round_lb[n.l_]) + k,
                    end(location_round_lb[n.l_]), lb_transfer);
          station_mark.set(to_idx(n.l_), true);
          any_marked = true;

          for (auto const fp : footpaths[n.l_]) {
            if (is_start.test(to_idx(fp.target()))) [[unlikely]] {
              continue;
            }

            auto const lb_fp = static_cast<std::uint16_t>(
                lb + adjusted_transfer_time(q.transfer_time_settings_,
                                            fp.duration().count()));
            if (lb_fp < location_round_lb[fp.target()][k]) {
              std::fill(begin(location_round_lb[fp.target()]) + k,
                        end(location_round_lb[fp.target()]), lb_fp);
              station_mark.set(to_idx(fp.target()), true);
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
                 prev_station_mark.count(), UTL_TIMING_MS(lb_raptor_round));

    for (auto const& s : q.start_) {
      fmt::println("k: {}, s: {}", k, location_round_lb[s.target()]);
    }

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
    bitvec&,
    bitvec&,
    bitvec&,
    vector_map<location_idx_t, std::array<std::uint16_t, kMaxTransfers + 2U>>&);

template void lb_raptor<direction::kBackward>(
    timetable const&,
    query const&,
    bitvec&,
    bitvec&,
    bitvec&,
    vector_map<location_idx_t, std::array<std::uint16_t, kMaxTransfers + 2U>>&);

}  // namespace nigiri::routing
