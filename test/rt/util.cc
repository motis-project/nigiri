#include "./util.h"

#include <sstream>

#include "utl/to_vec.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"

using namespace std::chrono_literals;

namespace nigiri::test {

transit_realtime::FeedMessage to_msg(
    std::vector<trip_update> const& updates,
    date::sys_seconds const msg_time,
    std::optional<std::string> const& start_date) {
  auto msg = transit_realtime::FeedMessage{};
  auto* const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(msg_time));

  auto id = 0U;
  for (auto const& u : updates) {
    auto* const e = msg.add_entity();
    e->set_id(fmt::format("{}", ++id));
    e->set_is_deleted(false);
    auto* const tu = e->mutable_trip_update();
    tu->mutable_trip()->set_trip_id(u.trip_id_);
    if (start_date.has_value()) {
      tu->mutable_trip()->set_start_date(*start_date);
    }
    for (auto const& s : u.stus_) {
      auto* const x = tu->add_stop_time_update();
      if (s.has_seq_) {
        x->set_stop_sequence(s.seq_);
      }
      if (s.is_assigned_as_stop_id_) {
        x->set_stop_id(*s.assigned_);
      } else {
        if (s.has_stop_id_) {
          x->set_stop_id(s.stop_id_);
        }
        if (s.assigned_.has_value()) {
          x->mutable_stop_time_properties()->set_assigned_stop_id(*s.assigned_);
        }
      }
      if (s.arr_delay_.has_value()) {
        x->mutable_arrival()->set_delay(*s.arr_delay_ * 60);
      }
      if (s.dep_delay_.has_value()) {
        x->mutable_departure()->set_delay(*s.dep_delay_ * 60);
      }
      if (s.is_skipped_) {
        x->set_schedule_relationship(
            transit_realtime::
                TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED);
      }
    }
  }
  return msg;
}

transit_realtime::FeedMessage to_feed_msg(std::vector<trip> const& trip_delays,
                                          date::sys_seconds const msg_time) {
  return to_msg(
      utl::to_vec(
          trip_delays,
          [](trip const& t) {
            return trip_update{
                .trip_id_ = t.trip_id_,
                .stus_ = utl::to_vec(t.delays_, [](trip::delay const& d) {
                  auto const is_dep = d.ev_type_ == event_type::kDep;
                  auto const delay = std::optional{d.delay_minutes_};
                  return stu{.seq_ = d.seq_.value_or(0U),
                             .stop_id_ = d.stop_id_.value_or(""),
                             .arr_delay_ = is_dep ? std::nullopt : delay,
                             .dep_delay_ = is_dep ? delay : std::nullopt,
                             .has_seq_ = d.seq_.has_value(),
                             .has_stop_id_ = d.stop_id_.has_value()};
                })};
          }),
      msg_time);
}

void with_rt_trips(
    timetable const& tt,
    date::sys_days const base_day,
    std::vector<std::string> const& trip_ids,
    std::function<void(rt_timetable*, std::string_view)> const& fn) {
  auto const trips = trip_ids.size();
  auto const combinations = 1ULL << trips;  // 2^n combinations

  // without rt timetable
  fn(nullptr, "");

  // with all combinations of trips
  for (auto i = 1ULL; i < combinations; ++i) {
    auto rtt = rt::create_rt_timetable(tt, base_day);
    auto trip_delays = std::vector<trip>{};
    std::stringstream s;
    for (auto j = 0ULL; j < trips; ++j) {
      if ((i & (1 << j)) != 0ULL) {
        if (!trip_delays.empty()) {
          s << ", ";
        }
        s << trip_ids[j];
        trip_delays.emplace_back(trip{.trip_id_ = trip_ids[j], .delays_ = {}});
      }
    }
    rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "",
                          to_feed_msg(trip_delays, base_day + 1h));
    fn(&rtt, s.str());
  }
}

}  // namespace nigiri::test
