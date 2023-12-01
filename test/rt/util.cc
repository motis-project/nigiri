#include "./util.h"

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

}  // namespace nigiri::test