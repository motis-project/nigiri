#include "nigiri/routing/raptor/reconstruct.h"

#include "utl/verify.h"

#include "nigiri/routing/direct.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <direction SearchDir>
void add_offset_legs(timetable const& tt,
                     rt_timetable const* rtt,
                     query const& q,
                     journey& j) {
  utl::verify(!j.legs_.empty(), "reconstruct: journey without transit legs");

  constexpr auto const is_fwd = SearchDir == direction::kForward;

  auto const add_offset_leg = [&](bool const is_front) {
    auto const is_start = is_front == is_fwd;
    if ((is_start ? q.start_match_mode_ : q.dest_match_mode_) !=
        location_match_mode::kIntermodal) {
      return;
    }

    auto const& transit_leg = is_front ? j.legs_.front() : j.legs_.back();
    auto const stop =
        rt::base(tt, rtt, is_front ? transit_leg.from_ : transit_leg.to_);
    auto leg = get_offset_leg(
        tt, is_front ? event_type::kDep : event_type::kArr,
        is_front ? transit_leg.dep_time_ : transit_leg.arr_time_, stop,
        is_start ? special_station::kStart : special_station::kEnd,
        is_start ? q.start_ : q.destination_,
        is_start ? q.td_start_ : q.td_dest_);
    utl::verify(leg.has_value(), "reconstruct: no mumo offset");

    auto const special_time = is_front ? leg->dep_time_ : leg->arr_time_;
    utl::verify(!is_start ? special_time == j.dest_time_
                : is_fwd  ? special_time >= j.start_time_
                          : special_time <= j.start_time_,
                "reconstruct: mumo offset does not fit");

    (is_front ? leg->to_ : leg->from_) = stop;
    j.legs_.insert(is_front ? begin(j.legs_) : end(j.legs_), *leg);
  };
  add_offset_leg(true);
  add_offset_leg(false);

  // the start footpath that seeded round 0, cf. CPU find_start_footpath
  if (q.start_match_mode_ != location_match_mode::kIntermodal) {
    auto const search_start_l =
        is_fwd ? j.legs_.front().from_ : j.legs_.back().to_;
    auto const start_l = rt::static_location(rtt, search_start_l);
    auto const start_t =
        is_fwd ? j.legs_.front().dep_time_ : j.legs_.back().arr_time_;
    auto const direct_start_ok =
        is_fwd ? j.start_time_ <= start_t : j.start_time_ >= start_t;
    if (!is_journey_start(tt, q, start_l) || !direct_start_ok) {
      auto fp = lookup_footpath(
          search_start_l, start_t, is_fwd ? event_type::kDep : event_type::kArr,
          tt, rtt, q, q.start_, q.start_match_mode_, true);
      if (fp.has_value() && is_fwd) {
        fp->arr_time_ = j.start_time_ + (fp->arr_time_ - fp->dep_time_);
        fp->dep_time_ = j.start_time_;
      }
      if (fp.has_value() &&
          fp->arr_time_ <= (is_fwd ? start_t : j.start_time_)) {
        j.legs_.insert(is_fwd ? begin(j.legs_) : end(j.legs_), *fp);
      }
    }
  }

  j.dest_ = is_fwd ? j.legs_.back().to_ : j.legs_.front().from_;
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
