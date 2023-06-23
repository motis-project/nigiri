#include "nigiri/rt/gtfsrt_update.h"

#include "utl/pairwise.h"

#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/logging.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/run.h"

namespace gtfsrt = transit_realtime;
namespace pb = google::protobuf;

namespace nigiri::rt {

struct delay_propagation {
  unixtime_t pred_time_;
  duration_t pred_delay_;
};

delay_propagation update_delay(timetable const& tt,
                               rt_timetable& rtt,
                               run const& r,
                               stop_idx_t const stop_idx,
                               event_type const ev_type,
                               duration_t const delay,
                               unixtime_t const min) {
  auto const static_time = tt.event_time(r.t_, stop_idx, ev_type);
  rtt.update_time(r.rt_, stop_idx, ev_type, std::max(min, static_time + delay));
  return {rtt.unix_event_time(r.rt_, stop_idx, ev_type), delay};
}

delay_propagation update_event(timetable const& tt,
                               rt_timetable& rtt,
                               run const& r,
                               stop_idx_t const stop_idx,
                               event_type const ev_type,
                               gtfsrt::TripUpdate_StopTimeEvent const& ev,
                               unixtime_t const pred_time) {
  if (ev.has_time()) {
    auto const static_time = tt.event_time(r.t_, stop_idx, ev_type);
    auto const new_time =
        unixtime_t{std::chrono::duration_cast<unixtime_t::duration>(
            std::chrono::seconds{ev.time()})};
    rtt.update_time(r.rt_, stop_idx, ev_type, std::max(pred_time, new_time));
    return {new_time, new_time - static_time};
  } else /* if (ev.has_delay()) */ {
    return update_delay(tt, rtt, r, stop_idx, ev_type,
                        std::chrono::duration_cast<unixtime_t::duration>(
                            std::chrono::seconds{ev.delay()}),
                        pred_time);
  }
}

void update_run(
    source_idx_t const src,
    timetable const& tt,
    rt_timetable& rtt,
    run& r,
    pb::RepeatedPtrField<gtfsrt::TripUpdate_StopTimeUpdate> const& stops) {
  using std::begin;
  using std::end;

  if (!r.is_rt()) {
    r.rt_ = rtt.add_rt_transport(src, tt, r.t_);
  }

  auto const location_seq =
      tt.route_location_seq_[tt.transport_route_[r.t_.t_idx_]];
  auto const seq_numbers = ::nigiri::loader::gtfs::stop_seq_number_range{
      std::span{tt.transport_stop_seq_numbers_[r.t_.t_idx_]},
      static_cast<stop_idx_t>(location_seq.size())};

  auto pred = std::optional<delay_propagation>{};
  auto stop_idx = stop_idx_t{0U};
  auto seq_it = begin(seq_numbers);
  auto upd_it = begin(stops);
  for (; seq_it != end(seq_numbers) && stop_idx != location_seq.size();
       ++stop_idx, ++seq_it) {
    auto const matches =
        upd_it != end(stops) &&
        ((upd_it->has_stop_sequence() && upd_it->stop_sequence() == *seq_it) ||
         (upd_it->has_stop_id() &&
          upd_it->stop_id() ==
              tt.locations_.ids_[stop{location_seq[stop_idx]}.location_idx()]
                  .view()));

    if (stop_idx != 0U) {
      if (matches && upd_it->has_arrival() &&
          (upd_it->arrival().has_delay() || upd_it->arrival().has_time())) {
        pred = update_event(
            tt, rtt, r, stop_idx, event_type::kArr, upd_it->arrival(),
            pred.has_value() ? pred->pred_time_ : unixtime_t{0_minutes});
      } else {
        pred = update_delay(
            tt, rtt, r, stop_idx, event_type::kArr, pred->pred_delay_,
            pred.has_value() ? pred->pred_time_ : unixtime_t{0_minutes});
      }
    }

    if (stop_idx == 0U && matches && upd_it->has_arrival() &&
        !upd_it->has_departure() &&
        (upd_it->arrival().has_delay() || upd_it->arrival().has_time())) {
      // First arrival has update, but first departure doesn't. Update departure
      // with arrival info (assuming they have the same static timetable,
      // because we don't store the static first arrival) to enable delay
      // propagation.
      pred = update_event(tt, rtt, r, stop_idx, event_type::kDep,
                          upd_it->arrival(), unixtime_t{0_minutes});
    } else if (stop_idx != location_seq.size() - 1U) {
      if (matches && upd_it->has_departure() &&
          (upd_it->departure().has_time() || upd_it->departure().has_delay())) {
        pred = update_event(
            tt, rtt, r, stop_idx, event_type::kDep, upd_it->departure(),
            pred.has_value() ? pred->pred_time_ : unixtime_t{0_minutes});
      } else {
        pred = update_delay(
            tt, rtt, r, stop_idx, event_type::kDep, pred->pred_delay_,
            pred.has_value() ? pred->pred_time_ : unixtime_t{0_minutes});
      }
    }

    if (matches) {
      ++upd_it;
    }
  }
}

std::string remove_nl(std::string s) {
  s.erase(std::remove(begin(s), end(s), '\n'), end(s));
  return s;
}

statistics gtfsrt_update_msg(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             gtfsrt::FeedMessage const& msg) {
  if (!msg.has_header()) {
    return {.no_header_ = true};
  }

  auto stats = statistics{.total_entities_ = msg.entity_size()};
  auto const message_time =
      date::sys_seconds{std::chrono::seconds{msg.header().timestamp()}};
  auto const today =
      std::chrono::time_point_cast<date::sys_days::duration>(message_time);
  for (auto const& entity : msg.entity()) {
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
    } else if (entity.trip_update().trip().schedule_relationship() !=
                   gtfsrt::TripDescriptor_ScheduleRelationship_SCHEDULED &&
               entity.trip_update().trip().schedule_relationship() !=
                   gtfsrt::TripDescriptor_ScheduleRelationship_CANCELED) {
      ++stats.unsupported_schedule_relationship_;
      continue;
    }

    try {
      auto const td = entity.trip_update().trip();
      auto r = gtfsrt_resolve_run(today, tt, rtt, src, td);

      if (!r.valid()) {
        log(log_lvl::error, "rt.gtfs.resolve", "could not resolve (tag={}) {}",
            tag, remove_nl(entity.trip_update().trip().DebugString()));
        ++stats.trip_resolve_error_;
        continue;
      }

      update_run(src, tt, rtt, r, entity.trip_update().stop_time_update());
      ++stats.total_entities_success_;
    } catch (const std::exception& e) {
      ++stats.total_entities_fail_;
      log(log_lvl::error, "rt.gtfs",
          "GTFS-RT error (tag={}): time={}, entitiy={}, message={}, error={}",
          tag, date::format("%T", message_time), entity.id(),
          remove_nl(entity.DebugString()), e.what());
    }
  }

  return stats;
}

statistics gtfsrt_update_buf(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             std::string_view protobuf) {
  gtfsrt::FeedMessage msg;

  auto const success =
      msg.ParseFromArray(reinterpret_cast<void const*>(protobuf.data()),
                         static_cast<int>(protobuf.size()));
  if (!success) {
    log(log_lvl::error, "rt.gtfs",
        "GTFS-RT error (tag={}): unable to parse protobuf message: {}", tag,
        protobuf.substr(0, std::min(protobuf.size(), size_t{1000U})));
    return {.parser_error_ = true};
  }

  return gtfsrt_update_msg(tt, rtt, src, tag, msg);
}

}  // namespace nigiri::rt