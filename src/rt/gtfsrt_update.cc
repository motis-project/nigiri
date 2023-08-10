#include "nigiri/rt/gtfsrt_update.h"

#include "utl/pairwise.h"

#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/logging.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/run.h"

namespace gtfsrt = transit_realtime;
namespace pb = google::protobuf;

namespace nigiri::rt {

std::ostream& operator<<(std::ostream& out, statistics const& s) {
  auto first = true;
  auto const print_if_no_empty = [&](char const* name, auto const& value,
                                     bool print_percent = false) {
    if (!value) {
      return;
    }
    if (!first) {
      out << ", ";
    }
    first = false;
    out << name << "=" << value;
    if (print_percent && value) {
      out << " ("
          << static_cast<float>(value) / static_cast<float>(s.total_entities_) *
                 100
          << "%)";
    }
  };

  print_if_no_empty("parser_error", s.parser_error_);
  print_if_no_empty("no_header", s.no_header_);
  print_if_no_empty("total_entities", s.total_entities_);
  print_if_no_empty("total_entities_success", s.total_entities_success_, true);
  print_if_no_empty("total_entities_fail", s.total_entities_fail_, true);
  print_if_no_empty("unsupported_deleted", s.unsupported_deleted_, true);
  print_if_no_empty("unsupported_vehicle", s.unsupported_vehicle_, true);
  print_if_no_empty("unsupported_alert", s.unsupported_alert_, true);
  print_if_no_empty("unsupported_no_trip_id", s.unsupported_no_trip_id_, true);
  print_if_no_empty("no_trip_update", s.no_trip_update_, true);
  print_if_no_empty("trip_update_without_trip", s.trip_update_without_trip_,
                    true);
  print_if_no_empty("trip_resolve_error", s.trip_resolve_error_, true);
  print_if_no_empty("unsupported_schedule_relationship",
                    s.unsupported_schedule_relationship_, true);

  return out;
}

struct delay_propagation {
  unixtime_t pred_time_;
  duration_t pred_delay_;
};

std::string remove_nl(std::string s) {
  std::replace(begin(s), end(s), '\n', ' ');
  return s;
}

delay_propagation update_delay(timetable const& tt,
                               rt_timetable& rtt,
                               run const& r,
                               stop_idx_t const stop_idx,
                               event_type const ev_type,
                               duration_t const delay,
                               std::optional<unixtime_t> const min) {
  auto const static_time = tt.event_time(r.t_, stop_idx, ev_type);
  rtt.update_time(r.rt_, stop_idx, ev_type,
                  min.has_value() ? std::max(*min, static_time + delay)
                                  : static_time + delay);
  return {rtt.unix_event_time(r.rt_, stop_idx, ev_type), delay};
}

delay_propagation update_event(timetable const& tt,
                               rt_timetable& rtt,
                               run const& r,
                               stop_idx_t const stop_idx,
                               event_type const ev_type,
                               gtfsrt::TripUpdate_StopTimeEvent const& ev,
                               std::optional<unixtime_t> const pred_time) {
  if (ev.has_delay()) {
    return update_delay(tt, rtt, r, stop_idx, ev_type,
                        std::chrono::duration_cast<unixtime_t::duration>(
                            std::chrono::seconds{ev.delay()}),
                        pred_time);
  } else /* if (ev.has_time()) */ {
    auto const static_time = tt.event_time(r.t_, stop_idx, ev_type);
    auto const new_time =
        unixtime_t{std::chrono::duration_cast<unixtime_t::duration>(
            std::chrono::seconds{ev.time()})};
    rtt.update_time(
        r.rt_, stop_idx, ev_type,
        pred_time.has_value() ? std::max(*pred_time, new_time) : new_time);
    return {new_time, new_time - static_time};
  }
}

void cancel_run(timetable const&, rt_timetable& rtt, run& r) {
  if (r.is_rt()) {
    rtt.rt_transport_is_cancelled_.set(to_idx(r.rt_), true);
  }
  if (r.is_scheduled()) {
    auto const& bf = rtt.bitfields_[rtt.transport_traffic_days_[r.t_.t_idx_]];
    rtt.bitfields_.emplace_back(bf).set(to_idx(r.t_.day_), false);
    rtt.transport_traffic_days_[r.t_.t_idx_] =
        bitfield_idx_t{rtt.bitfields_.size() - 1U};
  }
}

void update_run(
    source_idx_t const src,
    timetable const& tt,
    rt_timetable& rtt,
    trip_idx_t const trip,
    run& r,
    pb::RepeatedPtrField<gtfsrt::TripUpdate_StopTimeUpdate> const& stops) {
  using std::begin;
  using std::end;

  if (!r.is_rt()) {
    r.rt_ = rtt.add_rt_transport(src, tt, r.t_);
  } else {
    rtt.rt_transport_is_cancelled_.set(to_idx(r.rt_), false);
  }

  auto const location_seq =
      tt.route_location_seq_[tt.transport_route_[r.t_.t_idx_]];
  auto const seq_numbers = loader::gtfs::stop_seq_number_range{
      {tt.trip_stop_seq_numbers_[trip]},
      static_cast<stop_idx_t>(r.stop_range_.size())};

  auto pred = r.stop_range_.from_ > 0U
                  ? std::make_optional<delay_propagation>(delay_propagation{
                        .pred_time_ = rtt.unix_event_time(
                            r.rt_, r.stop_range_.from_, event_type::kArr),
                        .pred_delay_ = 0_minutes})
                  : std::nullopt;
  auto stop_idx = r.stop_range_.from_;
  auto seq_it = begin(seq_numbers);
  auto upd_it = begin(stops);
  for (; seq_it != end(seq_numbers); ++stop_idx, ++seq_it) {
    auto const matches =
        upd_it != end(stops) &&
        ((upd_it->has_stop_sequence() && upd_it->stop_sequence() == *seq_it) ||
         (upd_it->has_stop_id() &&
          upd_it->stop_id() ==
              tt.locations_.ids_[stop{location_seq[stop_idx]}.location_idx()]
                  .view()));

    if (matches) {
      auto& stp = rtt.rt_transport_location_seq_[r.rt_][stop_idx];
      if (upd_it->schedule_relationship() ==
          gtfsrt::TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED) {
        // Cancel skipped stops (in_allowed = out_allowed = false).
        stp = stop{stop{stp}.location_idx(), false, false}.value();
      } else if (upd_it->stop_time_properties().has_assigned_stop_id()) {
        // Handle track change.
        auto const& new_id = upd_it->stop_time_properties().assigned_stop_id();
        auto const l_it = tt.locations_.location_id_to_idx_.find(
            {.id_ = new_id, .src_ = src});
        if (l_it != end(tt.locations_.location_id_to_idx_)) {
          stp = stop{l_it->second, stop{stp}.in_allowed(),
                     stop{stp}.out_allowed()}
                    .value();
          auto transports = rtt.location_rt_transports_[l_it->second];
          if (utl::find(transports, r.rt_) == end(transports)) {
            transports.push_back(r.rt_);
          }
        } else {
          log(log_lvl::error, "gtfsrt.stop_assignment",
              "stop assignment: src={}, stop_id=\"{}\" not found", src, new_id);
        }
      } else {
        // Just reset in case a track change / skipped stop got reversed.
        stp = location_seq[stop_idx];
      }
    }

    // Update arrival, propagate delay.
    if (stop_idx != r.stop_range_.from_) {
      if (matches && upd_it->has_arrival() &&
          (upd_it->arrival().has_delay() || upd_it->arrival().has_time())) {
        pred = update_event(
            tt, rtt, r, stop_idx, event_type::kArr, upd_it->arrival(),
            pred.has_value() ? pred->pred_time_ : unixtime_t{0_minutes});
      } else if (pred.has_value()) {
        pred = update_delay(tt, rtt, r, stop_idx, event_type::kArr,
                            pred->pred_delay_, pred->pred_time_);
      }
    }

    // Update departure, propagate delay.
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
      } else if (pred.has_value()) {
        pred = update_delay(tt, rtt, r, stop_idx, event_type::kDep,
                            pred->pred_delay_, pred->pred_time_);
      }
    }

    if (matches) {
      ++upd_it;
    }
  }
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
      auto [r, trip] = gtfsrt_resolve_run(today, tt, rtt, src, td);

      if (!r.valid()) {
        log(log_lvl::error, "rt.gtfs.resolve", "could not resolve (tag={}) {}",
            tag, remove_nl(entity.trip_update().trip().DebugString()));
        ++stats.trip_resolve_error_;
        continue;
      }

      if (entity.trip_update().trip().schedule_relationship() ==
          gtfsrt::TripDescriptor_ScheduleRelationship_CANCELED) {
        cancel_run(tt, rtt, r);
      } else {
        update_run(src, tt, rtt, trip, r,
                   entity.trip_update().stop_time_update());
      }
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