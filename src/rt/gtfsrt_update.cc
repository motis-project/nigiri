#include "nigiri/rt/gtfsrt_update.h"

#include "nigiri/logging.h"
#include "nigiri/rt/gtfsrt_resolve_trip.h"

#include "gtfs-realtime.pb.h"

namespace gtfsrt = transit_realtime;

namespace nigiri::rt {

statistics update_gtfsrt(timetable const& tt,
                         rt_timetable& rtt,
                         source_idx_t const src,
                         std::string_view tag,
                         std::string_view s) {
  gtfsrt::FeedMessage feed_message;

  auto const success = feed_message.ParseFromArray(
      reinterpret_cast<void const*>(s.data()), s.size());
  if (!success) {
    log(log_lvl::error, "rt.gtfs",
        "GTFS-RT error (tag={}): unable to parse protobuf message: {}", tag,
        s.substr(0, std::min(s.size(), size_t{1000U})));
    return {.parser_error_ = true};
  }

  if (!feed_message.has_header()) {
    log(log_lvl::error, "rt.gtfs",
        "GTFS-RT error (tag={}): skipping message without header");
    return {.no_header_ = true};
  }

  auto stats = statistics{.total_entities_ = feed_message.entity_size()};
  auto const message_time = date::sys_seconds{
      std::chrono::seconds{feed_message.header().timestamp()}};
  for (auto const& entity : feed_message.entity()) {
    if (entity.is_deleted()) {
      ++stats.unsupported_deleted_;
      continue;
    } else if (entity.has_alert()) {
      ++stats.unsupported_alert_;
      continue;
    } else if (entity.has_vehicle()) {
      ++stats.unsupported_vehicle_;
      continue;
    } else if (!entity.has_trip_update()) {
      ++stats.no_trip_update_;
      continue;
    } else if (!entity.trip_update().has_trip()) {
      ++stats.trip_update_without_trip_;
      continue;
    } else if (!entity.trip_update().trip().has_trip_id()) {
      ++stats.unsupported_no_trip_id_;
      continue;
    }

    try {
      utl::verify(
          entity.has_trip_update(),
          "exactly one of [trip_update, vehicle, alert] must be present");
      auto const trp =
          gtfsrt_resolve_trip(tt, rtt, src, entity.trip_update().trip());
      (void)trp;
      ++stats.total_entities_success_;
    } catch (const std::exception& e) {
      ++stats.total_entities_fail_;
      log(log_lvl::error, "rt.gtfs",
          "GTFS-RT error (tag={}): time={}, entitiy={}, message={}, error={}",
          tag, date::format("%T", message_time), entity.id(),
          entity.DebugString(), e.what());
    }
  }

  return stats;
}

}  // namespace nigiri::rt