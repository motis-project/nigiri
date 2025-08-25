#include "nigiri/rt/gtfsrt_update.h"

#include <string_view>
#include <vector>

#include "utl/helpers/algorithm.h"
#include "utl/pairwise.h"
#include "utl/verify.h"

#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/get_otel_tracer.h"
#include "nigiri/location.h"
#include "nigiri/logging.h"
#include "nigiri/lookup/get_transport.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_alert.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/run.h"

namespace gtfsrt = transit_realtime;
namespace protob = google::protobuf;

namespace nigiri::rt {

std::ostream& operator<<(std::ostream& out, statistics const& s) {
  auto first = true;
  auto const print_if_no_empty = [&](char const* name, auto const& value,
                                     std::variant<bool, int> print_percent =
                                         false) {
    if (!value) {
      return;
    }
    if (!first) {
      out << ", ";
    }
    first = false;
    out << name << "=" << value;
    if (std::holds_alternative<int>(print_percent)) {
      out << " ("
          << static_cast<float>(value) /
                 static_cast<float>(std::get<int>(print_percent)) * 100
          << "%)";
    } else if (std::holds_alternative<bool>(print_percent) &&
               std::get<bool>(print_percent)) {
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
  print_if_no_empty("total_alerts", s.total_alerts_, true);
  print_if_no_empty("alert_total_informed_entities",
                    s.alert_total_informed_entities_, false);
  print_if_no_empty("alert_total_resolve_success",
                    s.alert_total_resolve_success_,
                    s.alert_total_informed_entities_);
  print_if_no_empty("alert_trip_not_found", s.alert_trip_not_found_,
                    s.alert_total_informed_entities_);
  print_if_no_empty("alert_empty_selector", s.alert_empty_selector_,
                    s.alert_total_informed_entities_);
  print_if_no_empty("alert_stop_not_found", s.alert_stop_not_found_,
                    s.alert_total_informed_entities_);
  print_if_no_empty("alert_direction_without_route",
                    s.alert_direction_without_route_,
                    s.alert_total_informed_entities_);
  print_if_no_empty("alert_route_id_not_found", s.alert_route_id_not_found_,
                    s.alert_total_informed_entities_);
  print_if_no_empty("alert_agency_id_not_found", s.alert_agency_id_not_found_,
                    s.alert_total_informed_entities_);
  print_if_no_empty("alert_invalid_route_type", s.alert_invalid_route_type_,
                    s.alert_total_informed_entities_);
  print_if_no_empty("unsupported_no_trip_id", s.unsupported_no_trip_id_, true);
  print_if_no_empty("total_vehicles", s.total_vehicles_, true);
  print_if_no_empty("no_vehicle_position", s.no_vehicle_position_,
                    s.total_vehicles_);
  print_if_no_empty("vehicle_position_without_position",
                    s.vehicle_position_without_position_, s.total_vehicles_);
  print_if_no_empty("vehicle_position_without_trip",
                    s.vehicle_position_without_trip_, s.total_vehicles_);
  print_if_no_empty("vehicle_position_trip_without_trip_id",
                    s.vehicle_position_trip_without_trip_id_,
                    s.total_vehicles_);
  print_if_no_empty("vehicle_position_position_not_at_stop",
                    s.vehicle_position_position_not_at_stop_,
                    s.total_vehicles_);
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
  auto const static_time =
      r.is_scheduled() ? tt.event_time(r.t_, stop_idx, ev_type) : min.value();
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
  if (ev.has_delay() && r.is_scheduled()) {
    return update_delay(tt, rtt, r, stop_idx, ev_type,
                        std::chrono::duration_cast<unixtime_t::duration>(
                            std::chrono::seconds{ev.delay()}),
                        pred_time);
  } else /* if (ev.has_time()) */ {
    auto const new_time =
        unixtime_t{std::chrono::duration_cast<unixtime_t::duration>(
            std::chrono::seconds{ev.time()})};
    auto const static_time =
        r.is_scheduled() ? tt.event_time(r.t_, stop_idx, ev_type) : new_time;
    auto const lower_bounded_new_time =
        pred_time.has_value() ? std::max(*pred_time, new_time) : new_time;
    rtt.update_time(r.rt_, stop_idx, ev_type, lower_bounded_new_time);
    rtt.dispatch_delay(r, stop_idx, ev_type,
                       lower_bounded_new_time - static_time);
    return {lower_bounded_new_time, new_time - static_time};
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

bool is_added(gtfsrt::TripDescriptor_ScheduleRelationship const sr) {
  return sr == gtfsrt::TripDescriptor_ScheduleRelationship_ADDED ||
         sr == gtfsrt::TripDescriptor_ScheduleRelationship_NEW;
}

bool is_added_with_ref(gtfsrt::TripDescriptor_ScheduleRelationship const sr) {
  return sr == gtfsrt::TripDescriptor_ScheduleRelationship_REPLACEMENT ||
         sr == gtfsrt::TripDescriptor_ScheduleRelationship_DUPLICATED;
}

bool add_rt_trip(source_idx_t const src,
                 timetable const& tt,
                 rt_timetable& rtt,
                 run& r,
                 gtfsrt::TripUpdate const& tripUpdate) {
  auto const& stus = tripUpdate.stop_time_update();
  auto const sr = tripUpdate.trip().schedule_relationship();
  auto const added_or_replaced =
      is_added(sr) ||
      sr == transit_realtime::TripDescriptor_ScheduleRelationship_REPLACEMENT;

  auto stops = std::vector<stop::value_type>{};
  if (added_or_replaced) {
    for (auto const& stu : stus) {
      utl::verify((!stu.has_departure() || stu.departure().has_time()) &&
                      (!stu.has_arrival() || stu.arrival().has_time()),
                  "absolute times are required for unscheduled trips");
      utl::verify(stu.has_stop_id(),
                  "stop_id is required for unscheduled trips");
      auto const it =
          tt.locations_.location_id_to_idx_.find({stu.stop_id(), src});
      if (it == end(tt.locations_.location_id_to_idx_)) {
        log(log_lvl::error, "rt.gtfs.unsupported",
            "NEW/ADDED stop_id must be contained in stops.txt "
            "(src={}, trip_id={}, stop_id={}), skipping",
            src, tripUpdate.trip().trip_id(), stu.stop_id());
        return false;
      }
      auto in_allowed = true, out_allowed = true;
      if (stu.has_stop_time_properties()) {
        if (stu.stop_time_properties().has_pickup_type()) {
          in_allowed =
              stu.stop_time_properties().pickup_type() !=
              transit_realtime::
                  TripUpdate_StopTimeUpdate_StopTimeProperties_DropOffPickupType_NONE;
        }
        if (stu.stop_time_properties().has_drop_off_type()) {
          out_allowed =
              stu.stop_time_properties().drop_off_type() !=
              transit_realtime::
                  TripUpdate_StopTimeUpdate_StopTimeProperties_DropOffPickupType_NONE;
        }
      }
      stops.emplace_back(stop{it->second, in_allowed, out_allowed, false, false}
                             .value());  // TODO wheelchair
    }
    utl::verify(stops.size() > 1,
                "added trip must contain more than 1 valid stop");
  }

  auto times = added_or_replaced
                   ? std::vector<delta_t>(stops.size() * 2U - 2U, 0)
                   : std::vector<delta_t>{};

  auto const new_trip_id = [&]() -> std::string_view {
    if (is_added(sr) && tripUpdate.trip().has_trip_id()) {
      return std::string_view{tripUpdate.trip().trip_id()};
    }
    if (sr == gtfsrt::TripDescriptor_ScheduleRelationship_DUPLICATED &&
        tripUpdate.has_trip_properties() &&
        tripUpdate.trip_properties().has_trip_id()) {
      return std::string_view{tripUpdate.trip_properties().trip_id()};
    }
    return {};
  };
  auto const route_id = [&]() -> std::string_view {
    if ((is_added(sr) ||
         sr == gtfsrt::TripDescriptor_ScheduleRelationship_DUPLICATED) &&
        tripUpdate.trip().has_route_id()) {
      return std::string_view{tripUpdate.trip().route_id()};
    }
    return {};
  };
  auto const trip_short_name =
      tripUpdate.has_trip_properties() &&
              tripUpdate.trip_properties().has_trip_short_name()
          ? std::string_view{tripUpdate.trip_properties().trip_short_name()}
          : std::string_view{};
  // ADDED/NEW stops+times+new_trip_id
  // REPLACEMENT stops+times
  // DUPL new_trip_id
  r.rt_ = rtt.add_rt_transport(src, tt, r.t_, stops, times, new_trip_id(),
                               route_id(), trip_short_name);
  if (sr == transit_realtime::TripDescriptor_ScheduleRelationship_REPLACEMENT) {
    r.t_ = transport::invalid();
  }
  return true;
}

bool update_run(source_idx_t const src,
                timetable const& tt,
                rt_timetable& rtt,
                trip_idx_t const trip,
                run& r,
                gtfsrt::TripUpdate const& tripUpdate) {
  using std::begin;
  using std::end;

  if (!r.is_rt()) {
    if (!add_rt_trip(src, tt, rtt, r, tripUpdate)) {
      return false;
    }
  } else {
    rtt.rt_transport_is_cancelled_.set(to_idx(r.rt_), false);
  }

  auto const& rtt_const = rtt;
  auto const location_seq =
      r.is_scheduled()
          ? std::span{tt.route_location_seq_[tt.transport_route_[r.t_.t_idx_]]}
          : std::span{rtt_const.rt_transport_location_seq_[r.rt_]};
  auto const seq_numbers =
      r.is_scheduled()
          ? loader::gtfs::
                stop_seq_number_range{{tt.trip_stop_seq_numbers_[trip]},
                                      static_cast<stop_idx_t>(
                                          r.stop_range_.size())}
          : loader::gtfs::stop_seq_number_range{
                std::span<stop_idx_t>{},
                static_cast<stop_idx_t>(location_seq.size())};

  auto pred = r.is_scheduled() && r.stop_range_.from_ > 0U
                  ? std::make_optional<delay_propagation>(delay_propagation{
                        .pred_time_ = rtt.unix_event_time(
                            r.rt_, r.stop_range_.from_, event_type::kArr),
                        .pred_delay_ = 0_minutes})
                  : std::nullopt;
  auto stop_idx =
      r.is_scheduled() ? r.stop_range_.from_ : static_cast<unsigned short>(0U);
  auto seq_it = begin(seq_numbers);
  auto const& stus = tripUpdate.stop_time_update();
  auto upd_it = begin(stus);
  for (; seq_it != end(seq_numbers); ++stop_idx, ++seq_it) {
    auto const loc_idx = stop{location_seq[stop_idx]}.location_idx();
    auto const matches =
        upd_it != end(stus) &&
        ((r.is_scheduled() && upd_it->has_stop_sequence() &&
          upd_it->stop_sequence() == *seq_it) ||
         (upd_it->has_stop_id() &&
          upd_it->stop_id() == tt.locations_.ids_[loc_idx].view()));

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
                  upd_it->stop_id() != tt.locations_.ids_[loc_idx].view())) {
        // Handle track change.
        auto const& new_id =
            upd_it->stop_time_properties().has_assigned_stop_id()
                ? upd_it->stop_time_properties().assigned_stop_id()
                : upd_it->stop_id();
        auto const l_it = tt.locations_.location_id_to_idx_.find(
            {.id_ = new_id, .src_ = src});
        if (l_it == end(tt.locations_.location_id_to_idx_)) {
          log(log_lvl::error, "gtfsrt.stop_assignment",
              "stop assignment: src={}, old_stop_id=\"{}\", new_stop_id=\"{}\" "
              "not found",
              src, tt.locations_.ids_[loc_idx].view(), new_id);
          continue;
        }
        auto const& equiv_locs = tt.locations_.equivalences_.at(loc_idx);
        if (utl::find(equiv_locs, l_it->second) == end(equiv_locs)) {
          log(log_lvl::error, "gtfsrt.stop_assignment",
              "stop assignment: src={}, old_stop_id=\"{}\", new_stop_id=\"{}\" "
              "is not a mere track change, skipping",
              src, tt.locations_.ids_[loc_idx].view(), new_id);
          continue;
        }
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
    rtt.cancel_run(r);
  }
  return true;
}

void handle_vehicle_position(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             gtfsrt::FeedEntity const& entity,
                             date::sys_days const today,
                             date::sys_seconds const message_time,
                             std::shared_ptr<opentelemetry::trace::Span>& span,
                             statistics& stats) {

  try {
    auto const& vp = entity.vehicle();
    auto const& td = vp.trip();

    auto [r, trip_idx] = gtfsrt_resolve_run(today, tt, &rtt, src, td,
                                            std::string_view{td.trip_id()});

    if (!r.is_scheduled()) {
      log(log_lvl::info, "rt.gtfs.unsupported",
          R"(Run is not scheduled. Skipping Message)", tag, entity.id());
      return;
    }

    auto const& rtt_const = rtt;
    auto const location_seq =
        std::span{tt.route_location_seq_[tt.transport_route_[r.t_.t_idx_]]};

    // add rt_transport if not existent
    if (!r.is_rt()) {
      r.rt_ = rtt.add_rt_transport(src, tt, r.t_);
    }

    // match position to stop
    auto const vp_position =
        geo::latlng{vp.position().latitude(), vp.position().longitude()};
    auto const app_dist_lng_deg_vp =
        geo::approx_distance_lng_degrees(vp_position);
    auto const stop_it = utl::find_if(location_seq, [&](auto const& stp) {
      auto const loc = tt.locations_.get(stop{stp}.location_idx());
      return geo::approx_squared_distance(loc.pos_, vp_position,
                                          app_dist_lng_deg_vp) < 10;
    });
    if (stop_it == end(location_seq)) {
      log(log_lvl::debug, "rt.gtfs.vehicle_update",
          "Position of Vehicle was not near stop. Skipping message.", tag,
          entity.id());
      ++stats.vehicle_position_position_not_at_stop_;
      return;
    }

    // get remaining stops
    auto const stopped_at_idx =
        static_cast<stop_idx_t>(std::distance(begin(location_seq), stop_it));
    auto const fr = frun::from_t(tt, &rtt_const, r.t_);

    // get delay
    auto const vp_ts = vp.has_timestamp()
                           ? unixtime_t{std::chrono::duration_cast<i32_minutes>(
                                 std::chrono::seconds{vp.timestamp()})}
                           : std::chrono::system_clock::now();
    auto const ev_time =
        stopped_at_idx == 0
            ? tt.event_time(r.t_, stopped_at_idx, event_type::kDep)
            : tt.event_time(r.t_, stopped_at_idx, event_type::kArr);
    auto const delay_cast =
        std::chrono::duration_cast<duration_t>(vp_ts - ev_time);

    // update delay for remaining stops
    auto const stops_after = interval{stopped_at_idx, fr.stop_range_.to_};
    if (stopped_at_idx != *fr.stop_range_.begin()) {
      update_delay(tt, rtt, r, stopped_at_idx, event_type::kArr, delay_cast,
                   std::nullopt);
    }
    for (auto const [first, second] : utl::pairwise(stops_after)) {
      update_delay(tt, rtt, r, first, event_type::kDep, delay_cast,
                   std::nullopt);
      update_delay(tt, rtt, r, second, event_type::kArr, delay_cast,
                   std::nullopt);
    }

    // update delay for previous stops if necessary
    auto const stops_before =
        interval{fr.stop_range_.from_, stopped_at_idx + 1};
    for (auto const [curr, prev] :
         utl::pairwise(it_range(stops_before.rbegin(), stops_before.rend()))) {
      auto const curr_stop = static_cast<stop_idx_t>(curr);
      auto const prev_stop = static_cast<stop_idx_t>(prev);

      if (rtt.unix_event_time(r.rt_, curr_stop, event_type::kArr) <
          rtt.unix_event_time(r.rt_, prev_stop, event_type::kDep)) {
        update_delay(tt, rtt, r, prev_stop, event_type::kDep, delay_cast,
                     std::nullopt);

        if (prev != *stops_before.begin() &&
            rtt.unix_event_time(r.rt_, prev_stop, event_type::kDep) <
                rtt.unix_event_time(r.rt_, prev_stop, event_type::kArr)) {
          update_delay(tt, rtt, r, prev_stop, event_type::kArr, delay_cast,
                       std::nullopt);
          continue;
        }
      }
      break;
    }
    ++stats.total_entities_success_;
  } catch (std::exception const& e) {
    ++stats.total_entities_fail_;
    log(log_lvl::error, "rt.gtfs",
        "GTFS-RT error (tag={}): time={}, entity={}, message={}, error={}", tag,
        date::format("%T", message_time), entity.id(),
        remove_nl(entity.DebugString()), e.what());
    span->AddEvent("exception", {{"exception.message", e.what()},
                                 {"entity.id", entity.id()},
                                 {"message", remove_nl(entity.DebugString())}});
  }
}

statistics gtfsrt_update_msg(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             gtfsrt::FeedMessage const& msg,
                             bool const use_vehicle_position) {
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

    unsupported(entity.has_is_deleted() && entity.is_deleted(), "deleted",
                stats.unsupported_deleted_);

    if (entity.has_alert()) {
      handle_alert(today, tt, rtt, src, tag, entity.alert(), stats);
      continue;
    }

    if (use_vehicle_position) {
      ++stats.total_vehicles_;
      if (!entity.has_vehicle()) {
        log(log_lvl::error, "rt.gtfs.unsupported",
            R"(unsupported: no "vehicle_position" field (tag={}, id={}), skipping message)",
            tag, entity.id());
        ++stats.no_vehicle_position_;
      } else if (!entity.vehicle().has_position()) {
        log(log_lvl::error, "rt.gtfs.unsupported",
            R"(unsupported: no "position" field in "vehicle_position" field (tag={}, id={}), skipping message)",
            tag, entity.id());
        ++stats.vehicle_position_without_position_;
      } else if (!entity.vehicle().has_trip()) {
        log(log_lvl::error, "rt.gtfs.unsupported",
            R"(unsupported: no "trip" field in "vehicle_position" field (tag={}, id={}), skipping message)",
            tag, entity.id());
        ++stats.vehicle_position_without_trip_;
      } else if (!entity.vehicle().trip().has_trip_id()) {
        log(log_lvl::error, "rt.gtfs.unsupported",
            R"(unsupported: no "trip_id" field in "trip" field (tag={}, id={}), skipping message)",
            tag, entity.id());
        ++stats.vehicle_position_trip_without_trip_id_;
      } else {
        handle_vehicle_position(tt, rtt, src, tag, entity, today, message_time,
                                span, stats);
      }
      continue;
    }

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

    auto const sr = entity.trip_update().trip().has_schedule_relationship()
                        ? entity.trip_update().trip().schedule_relationship()
                        : gtfsrt::TripDescriptor_ScheduleRelationship_SCHEDULED;

    if (sr == gtfsrt::TripDescriptor_ScheduleRelationship_DUPLICATED &&
        (!entity.trip_update().has_trip_properties() ||
         !entity.trip_update().trip_properties().has_trip_id())) {
      log(log_lvl::error, "rt.gtfs.unsupported",
          R"(unsupported: no "trip_properties.trip_id" field in "trip_update.trip" for DUPLICATED (tag={}, id={}), skipping message)",
          tag, entity.id());
      ++stats.unsupported_no_trip_id_;
      continue;
    }

    auto const added = is_added(sr);
    // auto const added_with_ref = is_added_with_ref(sr);

    if (sr != gtfsrt::TripDescriptor_ScheduleRelationship_SCHEDULED &&
        sr != gtfsrt::TripDescriptor_ScheduleRelationship_CANCELED && !added) {
      log(log_lvl::error, "rt.gtfs.unsupported",
          "unsupported schedule relationship {} (tag={}, id={}), skipping "
          "message",
          TripDescriptor_ScheduleRelationship_Name(sr), tag, entity.id());
      ++stats.unsupported_schedule_relationship_;
      continue;
    }

    try {
      auto const td = entity.trip_update().trip();
      auto const trip_id =
          entity.trip_update().has_trip_properties() &&
                  entity.trip_update().trip_properties().has_trip_id()
              ? std::string_view{entity.trip_update()
                                     .trip_properties()
                                     .trip_id()}
              : std::string_view{};

      auto is_resolved_static = false;
      resolve_static(today, tt, src, td, [&](run r, trip_idx_t const trip) {
        is_resolved_static = true;

        resolve_rt(rtt, r, trip_id);

        if (sr == gtfsrt::TripDescriptor_ScheduleRelationship_CANCELED) {
          rtt.cancel_run(r);
          ++stats.total_entities_success_;
        } else {
          if (update_run(src, tt, rtt, trip, r, entity.trip_update())) {
            ++stats.total_entities_success_;
          }
        }

        return utl::continue_t::kContinue;
      });

      if (added) {
        utl::verify(!is_resolved_static,
                    "NEW/ADDED trip is required to have a new trip_id");
        auto r = rt::run{};
        resolve_rt(rtt, r, trip_id.empty() ? td.trip_id() : trip_id);
        if (update_run(src, tt, rtt, trip_idx_t::invalid(), r,
                       entity.trip_update())) {
          ++stats.total_entities_success_;
        }
        continue;
      }

      if (!is_resolved_static) {
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
                 TripDescriptor_ScheduleRelationship_Name(sr)},
                {"trip.str", remove_nl(td.DebugString())},
            });
        ++stats.trip_resolve_error_;
      }
    } catch (std::exception const& e) {
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

  rtt.alerts_.strings_.cache_.clear();

  return stats;
}

statistics gtfsrt_update_buf(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             std::string_view protobuf,
                             gtfsrt::FeedMessage& msg,
                             bool const use_vehicle_position) {
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

  return gtfsrt_update_msg(tt, rtt, src, tag, msg, use_vehicle_position);
}

statistics gtfsrt_update_buf(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             std::string_view protobuf,
                             bool const use_vehicle_position) {
  auto msg = gtfsrt::FeedMessage{};
  return gtfsrt_update_buf(tt, rtt, src, tag, protobuf, msg,
                           use_vehicle_position);
}

}  // namespace nigiri::rt
