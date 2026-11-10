#include "nigiri/routing/raptor/reconstruct.h"

#include <optional>

#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/special_stations.h"
#include "nigiri/td_footpath.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <direction SearchDir>
void add_offset_legs(timetable const& tt,
                     rt_timetable const*,
                     query const& q,
                     journey& j) {
  utl::verify(!j.legs_.empty(), "reconstruct: journey without transit legs");

  // The legs are chronological.
  constexpr auto const is_fwd = SearchDir == direction::kForward;

  // Front-side mumo leg: special_station -> first transit stop.
  auto const from = j.legs_.front().from_;
  auto const dep_time = j.legs_.front().dep_time_;
  auto const front_match_mode =
      is_fwd ? q.start_match_mode_ : q.dest_match_mode_;
  if (front_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.start_ : q.destination_;
    auto const& td_offsets = is_fwd ? q.td_start_ : q.td_dest_;
    auto const special = get_special_station(is_fwd ? special_station::kStart
                                                    : special_station::kEnd);
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt, front_match_mode, x.target(), from) &&
             (is_fwd
                  // fwd: query start, check feasibility (allows ontrip start)
                  ? dep_time - x.duration() >= j.start_time_
                  // bwd: destination, anchored exactly at j.dest_time_
                  : dep_time - x.duration() == j.dest_time_);
    });
    if (o != end(offsets)) {
      auto const dep = dep_time - o->duration();
      j.legs_.insert(begin(j.legs_), journey::leg{direction::kForward, special,
                                                  from, dep, dep_time, *o});
    } else {
      // td offset (e.g. flex), mirrors the CPU reconstruct:
      // - fwd = query start side (find_start_footpath): evaluated backward
      //   from the first transit departure, feasible if not before the
      //   journey start
      // - bwd = td egress side (get_legs intermodal dest): evaluated forward
      //   from the final label j.dest_time_
      // specify_td_offsets() refines both to the raw entry duration below.
      auto inserted = false;
      for (auto const& [target, tds] : td_offsets) {
        if (!matches(tt, front_match_mode, target, from)) {
          continue;
        }
        if (is_fwd) {
          auto const fp = get_td_duration<direction::kBackward>(tds, dep_time);
          if (!fp.has_value() || dep_time - fp->first < j.start_time_) {
            continue;
          }
          j.legs_.insert(
              begin(j.legs_),
              journey::leg{direction::kForward, special, from,
                           dep_time - fp->first, dep_time,
                           offset{target, fp->first, fp->second.mode()}});
        } else {
          auto const t = j.dest_time_;
          auto const fp = get_td_duration<direction::kForward>(tds, t);
          if (!fp.has_value() || t + fp->first > dep_time) {
            continue;  // must reach the stop before the transit departure
          }
          j.legs_.insert(
              begin(j.legs_),
              journey::leg{direction::kForward, special, from, t, t + fp->first,
                           offset{target, fp->first, fp->second.mode()}});
        }
        inserted = true;
        break;
      }
      utl::verify(inserted, "reconstruct: no front offset");
    }
  }

  // offset: last transit stop -> special_station.
  auto const to = j.legs_.back().to_;
  auto const arr_time = j.legs_.back().arr_time_;
  auto const back_match_mode =
      is_fwd ? q.dest_match_mode_ : q.start_match_mode_;
  if (back_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.destination_ : q.start_;
    auto const& td_offsets = is_fwd ? q.td_dest_ : q.td_start_;
    auto const special = get_special_station(is_fwd ? special_station::kEnd
                                                    : special_station::kStart);
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt, back_match_mode, x.target(), to) &&
             (is_fwd
                  // fwd: destination, anchored exactly at j.dest_time_
                  ? arr_time + x.duration() == j.dest_time_
                  // bwd: query start, anchored by feasibility
                  : arr_time + x.duration() <= j.start_time_);
    });
    if (o != end(offsets)) {
      auto const arr = arr_time + o->duration();
      j.legs_.push_back(
          journey::leg{direction::kForward, to, special, arr_time, arr, *o});
      j.dest_ = special;
    } else {
      // dest td offset
      auto inserted = false;
      for (auto const& [target, tds] : td_offsets) {
        if (!matches(tt, back_match_mode, target, to)) {
          continue;
        }

        if (is_fwd) {
          auto const fp =
              get_td_duration<direction::kBackward>(tds, j.dest_time_);
          if (!fp.has_value() ||
              j.dest_time_ - fp->first /* duration*/ < arr_time) {
            continue;
          }

          j.legs_.push_back(journey::leg{
              direction::kForward, to, special, j.dest_time_ - fp->first,
              j.dest_time_, offset{target, fp->first, fp->second.mode()}});
        } else {
          auto const fp = get_td_duration<direction::kForward>(tds, arr_time);
          if (!fp.has_value() || arr_time + fp->first > j.start_time_) {
            continue;
          }

          j.legs_.push_back(journey::leg{
              direction::kForward, to, special, arr_time, arr_time + fp->first,
              offset{target, fp->first, fp->second.mode()}});
        }

        j.dest_ = special;
        inserted = true;

        break;
      }
      utl::verify(inserted, "reconstruct: no back offset");
    }
  }

  // use_start_footpaths_ == true:
  // reconstruct the start footpath that seeded round k=0 at the first stop.
  if (q.start_match_mode_ != location_match_mode::kIntermodal) {
    auto const is_journey_start = [&](location_idx_t const l) {
      for (auto const& o : q.start_) {
        if (matches(tt, q.start_match_mode_, o.target(), l)) {
          return true;
        }
      }
      return false;
    };
    auto const start_l = is_fwd ? j.legs_.front().from_ : j.legs_.back().to_;
    auto const start_t =
        is_fwd ? j.legs_.front().dep_time_ : j.legs_.back().arr_time_;
    auto const direct_start_ok =
        is_fwd ? j.start_time_ <= start_t : j.start_time_ >= start_t;
    if (!is_journey_start(start_l) || !direct_start_ok) {
      auto const fps = is_fwd
                           ? tt.locations_.footpaths_in_[q.prf_idx_][start_l]
                           : tt.locations_.footpaths_out_[q.prf_idx_][start_l];
      auto best = std::optional<footpath>{};
      for (auto const fp : fps) {
        if ((!best.has_value() || fp.duration() < best->duration()) &&
            is_journey_start(fp.target())) {
          best = fp;
        }
      }
      if (best.has_value()) {
        auto const dur = duration_t{adjusted_transfer_time(
            q.transfer_time_settings_, best->duration().count())};
        auto const fp_arr = j.start_time_ + (is_fwd ? dur : -dur);
        if (is_fwd ? fp_arr <= start_t : fp_arr >= start_t) {
          auto const lg = journey::leg{
              SearchDir,     best->target(), start_l,
              j.start_time_, fp_arr,         footpath{best->target(), dur}};
          if (is_fwd) {
            j.legs_.insert(begin(j.legs_), lg);
          } else {
            j.legs_.push_back(lg);
          }
        }
      }
    }
  }
}

template void add_offset_legs<direction::kForward>(timetable const&,
                                                   rt_timetable const*,
                                                   query const&,
                                                   journey&);
template void add_offset_legs<direction::kBackward>(timetable const&,
                                                    rt_timetable const*,
                                                    query const&,
                                                    journey&);

}  // namespace nigiri::routing
