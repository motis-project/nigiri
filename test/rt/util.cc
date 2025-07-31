#include "./util.h"

#include <sstream>

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"

using namespace std::chrono_literals;

namespace nigiri::test {

transit_realtime::FeedMessage to_feed_msg(std::vector<trip> const& trip_delays,
                                          date::sys_seconds const msg_time) {
  transit_realtime::FeedMessage msg;

  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(msg_time));

  auto id = 0U;
  for (auto const& trip : trip_delays) {
    auto const e = msg.add_entity();
    e->set_id(fmt::format("{}", ++id));
    e->set_is_deleted(false);

    auto const td = e->mutable_trip_update()->mutable_trip();
    td->set_trip_id(trip.trip_id_);

    for (auto const& stop_delay : trip.delays_) {
      auto* const upd = e->mutable_trip_update()->add_stop_time_update();
      if (stop_delay.stop_id_.has_value()) {
        *upd->mutable_stop_id() = *stop_delay.stop_id_;
      }
      if (stop_delay.seq_.has_value()) {
        upd->set_stop_sequence(*stop_delay.seq_);
      }
      stop_delay.ev_type_ == nigiri::event_type::kDep
          ? upd->mutable_departure()->set_delay(stop_delay.delay_minutes_ * 60)
          : upd->mutable_arrival()->set_delay(stop_delay.delay_minutes_ * 60);
    }
  }

  return msg;
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
                          to_feed_msg(trip_delays, base_day + 1h), false);
    fn(&rtt, s.str());
  }
}

}  // namespace nigiri::test
