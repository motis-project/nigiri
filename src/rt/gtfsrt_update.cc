#include "nigiri/rt/gtfsrt_update.h"

#include "utl/pairwise.h"

#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/get_otel_tracer.h"
#include "nigiri/logging.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/run.h"

namespace gtfsrt = transit_realtime;
namespace protob = google::protobuf;

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
  auto const lower_bounded_new_time = min.has_value()
                                          ? std::max(*min, static_time + delay)
                                          : static_time + delay;
  rtt.update_time(r.rt_, stop_idx, ev_type, lower_bounded_new_time);
  rtt.dispatch_delay(r, stop_idx, ev_type,
                     lower_bounded_new_time - static_time);
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
    auto const lower_bounded_new_time =
        pred_time.has_value() ? std::max(*pred_time, new_time) : new_time;
    rtt.update_time(r.rt_, stop_idx, ev_type, lower_bounded_new_time);
    rtt.dispatch_delay(r, stop_idx, ev_type,
                       lower_bounded_new_time - static_time);
    return {lower_bounded_new_time, new_time - static_time};
  }
}

void cancel_run(timetable const&, rt_timetable& rtt, run& r) {
  if (r.is_rt()) {
    rtt.rt_transport_is_cancelled_.set(to_idx(r.rt_), true);
  }
  if (r.is_scheduled()) {
    auto const bf = rtt.bitfields_[rtt.transport_traffic_days_[r.t_.t_idx_]];
    rtt.bitfields_.emplace_back(bf).set(to_idx(r.t_.day_), false);
    rtt.transport_traffic_days_[r.t_.t_idx_] =
        bitfield_idx_t{rtt.bitfields_.size() - 1U};

    for (auto i = r.stop_range_.from_; i != r.stop_range_.to_; ++i) {
      rtt.dispatch_stop_change(r, i, event_type::kArr, std::nullopt, false);
      rtt.dispatch_stop_change(r, i, event_type::kDep, std::nullopt, false);
    }
  }
}

unixtime_t fallback_pred(rt_timetable const& rtt,
                         run const& r,
                         std::optional<delay_propagation> const pred,
                         stop_idx_t const stop_idx,
                         event_type const ev_type) {
  if (pred.has_value()) {
    return pred->pred_time_;
  }
  if (stop_idx == 0U) {
    return unixtime_t{0_minutes};
  }
  return rtt.unix_event_time(
      r.rt_, ev_type == event_type::kDep ? stop_idx - 1U : stop_idx, ev_type);
}

void update_run(
    source_idx_t const src,
    timetable const& tt,
    rt_timetable& rtt,
    trip_idx_t const trip,
    run& r,
    protob::RepeatedPtrField<gtfsrt::TripUpdate_StopTimeUpdate> const& stops) {
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
        auto l_idx = stop{stp}.location_idx();
        // Cancel skipped stops (in_allowed = out_allowed = false).
        stp = stop{l_idx, false, false, false, false}.value();
        rtt.dispatch_stop_change(r, stop_idx, event_type::kArr, l_idx, false);
        rtt.dispatch_stop_change(r, stop_idx, event_type::kDep, l_idx, false);
      } else if (upd_it->stop_time_properties().has_assigned_stop_id() ||
                 (upd_it->has_stop_id() &&
                  upd_it->stop_id() !=
                      tt.locations_
                          .ids_[stop{location_seq[stop_idx]}.location_idx()]
                          .view())) {
        // Handle track change.
        auto const& new_id =
            upd_it->stop_time_properties().has_assigned_stop_id()
                ? upd_it->stop_time_properties().assigned_stop_id()
                : upd_it->stop_id();
        auto const l_it = tt.locations_.location_id_to_idx_.find(
            {.id_ = new_id, .src_ = src});
        if (l_it != end(tt.locations_.location_id_to_idx_)) {
          auto const s = stop{stp};
          stp = stop{l_it->second, s.in_allowed(), s.out_allowed(),
                     s.in_allowed_wheelchair(), s.out_allowed_wheelchair()}
                    .value();
          auto transports = rtt.location_rt_transports_[l_it->second];
          if (utl::find(transports, r.rt_) == end(transports)) {
            transports.push_back(r.rt_);
          }
          rtt.dispatch_stop_change(r, stop_idx, event_type::kArr, l_it->second,
                                   s.out_allowed());
          rtt.dispatch_stop_change(r, stop_idx, event_type::kDep, l_it->second,
                                   s.in_allowed());
        } else {
          log(log_lvl::error, "gtfsrt.stop_assignment",
              "stop assignment: src={}, stop_id=\"{}\" not found", src, new_id);
        }
      } else {
        // Just reset in case a track change / skipped stop got reversed.
        if (location_seq[stop_idx] != stp) {
          stp = location_seq[stop_idx];
          auto reset_stop = stop{stp};
          rtt.dispatch_stop_change(r, stop_idx, event_type::kArr,
                                   reset_stop.location_idx(),
                                   reset_stop.out_allowed());
          rtt.dispatch_stop_change(r, stop_idx, event_type::kDep,
                                   reset_stop.location_idx(),
                                   reset_stop.in_allowed());
        }
      }
    }

    // Update arrival, propagate delay.
    if (stop_idx != r.stop_range_.from_) {
      if (matches && upd_it->has_arrival() &&
          (upd_it->arrival().has_delay() || upd_it->arrival().has_time())) {
        pred = update_event(
            tt, rtt, r, stop_idx, event_type::kArr, upd_it->arrival(),
            fallback_pred(rtt, r, pred, stop_idx, event_type::kDep));
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
            fallback_pred(rtt, r, pred, stop_idx, event_type::kArr));
      } else if (pred.has_value()) {
        pred = update_delay(tt, rtt, r, stop_idx, event_type::kDep,
                            pred->pred_delay_, pred->pred_time_);
      }
    }

    if (matches) {
      ++upd_it;
    }
  }

  auto const n_not_cancelled_stops = utl::count_if(
      rtt.rt_transport_location_seq_[r.rt_],
      [](stop::value_type const s) { return !stop{s}.is_cancelled(); });
  if (n_not_cancelled_stops <= 1U) {
    cancel_run(tt, rtt, r);
  }
}

statistics gtfsrt_update_msg(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             gtfsrt::FeedMessage const& msg) {
  auto span = get_otel_tracer()->StartSpan("gtfsrt_update_msg", {{"tag", tag}});
  auto scope = opentelemetry::trace::Scope{span};

  if (!msg.has_header()) {
    span->SetStatus(opentelemetry::trace::StatusCode::kError, "missing header");
    return {.no_header_ = true};
  }

  auto const message_time =
      date::sys_seconds{std::chrono::seconds{msg.header().timestamp()}};
  auto const today =
      std::chrono::time_point_cast<date::sys_days::duration>(message_time);
  auto stats = statistics{.total_entities_ = msg.entity_size(),
                          .feed_timestamp_ = message_time};

  span->SetAttribute("nigiri.gtfsrt.header.timestamp",
                     msg.header().timestamp());
  span->SetAttribute("nigiri.gtfsrt.total_entities", msg.entity_size());

  for (auto const& entity : msg.entity()) {
    auto const unsupported = [&](bool const is_set, char const* field,
                                 int& stat) {
      if (is_set) {
        log(log_lvl::error, "rt.gtfs.unsupported",
            R"(ignoring unsupported "{}" field (tag={}, id={}))", field, tag,
            entity.id());
        ++stat;
      }
    };

    unsupported(entity.has_vehicle(), "vehicle", stats.unsupported_vehicle_);
    unsupported(entity.has_alert(), "alert", stats.unsupported_alert_);
    unsupported(entity.has_is_deleted() && entity.is_deleted(), "deleted",
                stats.unsupported_deleted_);

    if (!entity.has_trip_update()) {
      log(log_lvl::error, "rt.gtfs.unsupported",
          R"(unsupported: no "trip_update" field (tag={}, id={}), skipping message)",
          tag, entity.id());
      ++stats.no_trip_update_;
      continue;
    }

    if (!entity.trip_update().has_trip()) {
      log(log_lvl::error, "rt.gtfs.unsupported",
          R"(unsupported: no "trip" field in "trip_update" field (tag={}, id={}), skipping message)",
          tag, entity.id());
      ++stats.trip_update_without_trip_;
      continue;
    }

    if (!entity.trip_update().trip().has_trip_id()) {
      log(log_lvl::error, "rt.gtfs.unsupported",
          R"(unsupported: no "trip_id" field in "trip_update.trip" (tag={}, id={}), skipping message)",
          tag, entity.id());
      ++stats.unsupported_no_trip_id_;
      continue;
    }

    if (entity.trip_update().trip().schedule_relationship() !=
            gtfsrt::TripDescriptor_ScheduleRelationship_SCHEDULED &&
        entity.trip_update().trip().schedule_relationship() !=
            gtfsrt::TripDescriptor_ScheduleRelationship_CANCELED) {
      log(log_lvl::error, "rt.gtfs.unsupported",
          "unsupported schedule relationship {} (tag={}, id={}), skipping "
          "message",
          TripDescriptor_ScheduleRelationship_Name(
              entity.trip_update().trip().schedule_relationship()),
          tag, entity.id());
      ++stats.unsupported_schedule_relationship_;
      continue;
    }

    try {
      auto const td = entity.trip_update().trip();
      auto [r, trip] = gtfsrt_resolve_run(today, tt, &rtt, src, td);

      if (!r.valid()) {
        log(log_lvl::error, "rt.gtfs.resolve", "could not resolve (tag={}) {}",
            tag, remove_nl(td.DebugString()));
        span->AddEvent(
            "unresolved trip",
            {
                {"entity.id", entity.id()},
                {"trip.trip_id", td.has_trip_id() ? td.trip_id() : ""},
                {"trip.route_id", td.has_route_id() ? td.route_id() : ""},
                {"trip.direction_id", td.direction_id()},
                {"trip.start_time", td.has_start_time() ? td.start_time() : ""},
                {"trip.start_date", td.has_start_date() ? td.start_date() : ""},
                {"trip.schedule_relationship",
                 td.has_schedule_relationship()
                     ? TripDescriptor_ScheduleRelationship_Name(
                           td.schedule_relationship())
                     : ""},
                {"trip.str", remove_nl(td.DebugString())},
            });
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
          "GTFS-RT error (tag={}): time={}, entity={}, message={}, error={}",
          tag, date::format("%T", message_time), entity.id(),
          remove_nl(entity.DebugString()), e.what());
      span->AddEvent("exception",
                     {{"exception.message", e.what()},
                      {"entity.id", entity.id()},
                      {"message", remove_nl(entity.DebugString())}});
    }
  }

  return stats;
}

statistics gtfsrt_update_buf(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             std::string_view protobuf,
                             gtfsrt::FeedMessage& msg) {
  msg.Clear();

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

statistics gtfsrt_update_buf(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             std::string_view protobuf) {
  auto msg = gtfsrt::FeedMessage{};
  return gtfsrt_update_buf(tt, rtt, src, tag, protobuf, msg);
}

}  // namespace nigiri::rt
