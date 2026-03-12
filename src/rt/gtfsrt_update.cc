#include "nigiri/rt/gtfsrt_update.h"

#include "boost/chrono/duration.hpp"

#include <string_view>
#include <vector>

#include "utl/helpers/algorithm.h"
#include "utl/pairwise.h"
#include "utl/verify.h"

#include "geo/latlng.h"

#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/delay_prediction.h"
#include "nigiri/get_otel_tracer.h"
#include "nigiri/logging.h"
#include "nigiri/lookup/get_transport.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_alert.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/run.h"
#include "nigiri/types.h"

#include "utl/pipes/avg.h"

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
    return {lower_bounded_new_time, lower_bounded_new_time - static_time};
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
    auto last_pos = geo::latlng{};
    auto last_time = unixtime_t{};
    for (auto const& stu : stus) {
      auto new_time = unixtime_t{};
      if (stu.has_departure() && stu.departure().has_time()) {
        new_time = unixtime_t{std::chrono::duration_cast<unixtime_t::duration>(
            std::chrono::seconds{stu.departure().time()})};
      } else if (stu.has_arrival() && stu.arrival().has_time()) {
        new_time = unixtime_t{std::chrono::duration_cast<unixtime_t::duration>(
            std::chrono::seconds{stu.arrival().time()})};
      } else {
        throw utl::fail("absolute times are required for unscheduled trips");
      }
      utl::verify(stu.has_stop_id(),
                  "stop_id is required for unscheduled trips");
      auto const it =
          tt.locations_.location_id_to_idx_.find({stu.stop_id(), src});
      if (it == end(tt.locations_.location_id_to_idx_)) {
        log(log_lvl::debug, "rt.gtfs.unsupported",
            "NEW/ADDED stop_id must be contained in stops.txt "
            "(src={}, trip_id={}, stop_id={}), skipping",
            src, tripUpdate.trip().trip_id(), stu.stop_id());
        return false;
      }
      if (last_time != unixtime_t{}) {
        auto const time_between_stops = (new_time - last_time).count();
        auto const dist_between_stops =
            geo::distance(tt.locations_.coordinates_.at(it->second), last_pos);
        if (dist_between_stops / std::max(time_between_stops, 1) / 60 >
            kMaxTransitSpeed) {
          log(log_lvl::debug, "rt.gtfs.invalid",
              "NEW/ADDED trip is travelling too fast "
              "(src={}, trip_id={}, stop_id={}, dist={}, delta={}), skipping",
              src, tripUpdate.trip().trip_id(), stu.stop_id(),
              dist_between_stops, time_between_stops);
          return false;
        }
      }
      last_pos = tt.locations_.coordinates_.at(it->second);
      last_time = new_time;
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
  auto const direction_id = [&]() -> direction_id_t {
    if ((is_added(sr) ||
         sr == gtfsrt::TripDescriptor_ScheduleRelationship_DUPLICATED) &&
        tripUpdate.trip().has_direction_id()) {
      return direction_id_t{tripUpdate.trip().direction_id() != 0U};
    }
    return direction_id_t{};
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
                               route_id(), direction_id(), trip_short_name);
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

  auto pred_time = std::numeric_limits<delta_t>::min();
  auto i = 0U;
  for (auto& curr : rtt.rt_transport_stop_times_[r.rt_]) {
    if (curr < pred_time) {
      curr = pred_time;

      auto const stop = static_cast<stop_idx_t>((i + 1U) / 2U);
      auto const ev_type = i % 2U == 0U ? event_type::kDep : event_type::kArr;
      auto const curr_unix_time = rtt.base_day_ + duration_t{curr};
      auto const static_time = r.is_scheduled()
                                   ? tt.event_time(r.t_, stop, ev_type)
                                   : curr_unix_time;
      rtt.dispatch_delay(r, stop, ev_type, curr_unix_time - static_time);
    }
    pred_time = curr;
    ++i;
  }

  auto const n_not_cancelled_stops = utl::count_if(
      rtt.rt_transport_location_seq_[r.rt_],
      [](stop::value_type const s) { return !stop{s}.is_cancelled(); });
  if (n_not_cancelled_stops <= 1U) {
    rtt.cancel_run(r);
  }
  return true;
}

void calculate_delay_simple(timetable const& tt,
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

    auto [r, trip_idx] = gtfsrt_resolve_run(today, tt, &rtt, src, td);

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
      return geo::approx_squared_distance(
                 tt.locations_.coordinates_[stop{stp}.location_idx()],
                 vp_position, app_dist_lng_deg_vp) < 10;
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
         utl::pairwise(it_range{stops_before.rbegin(), stops_before.rend()})) {
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
    log(log_lvl::debug, "rt.gtfs",
        "GTFS-RT error (tag={}): time={}, entity={}, message={}, error={}", tag,
        date::format("%T", message_time), entity.id(),
        remove_nl(entity.DebugString()), e.what());
    span->AddEvent("exception", {{"exception.message", e.what()},
                                 {"entity.id", entity.id()},
                                 {"message", remove_nl(entity.DebugString())}});
  }
}

void calculate_delay_intelligent(
    timetable const& tt,
    rt_timetable& rtt,
    source_idx_t const src,
    std::string_view tag,
    gtfsrt::FeedEntity const& entity,
    date::sys_days const today,
    date::sys_seconds const message_time,
    std::shared_ptr<opentelemetry::trace::Span>& span,
    statistics& stats,
    delay_prediction* delay_prediction) {

  try {
    // insertion of new data

    auto const& vp = entity.vehicle();
    auto const vehicle_position =
        geo::latlng{vp.position().latitude(), vp.position().longitude()};
    auto const td = vp.trip();

    auto r =
        vp.has_trip() &&
                (vp.trip().has_trip_id() ||
                 (vp.trip().has_route_id() && vp.trip().has_direction_id() &&
                  vp.trip().has_start_date() && vp.trip().has_start_time()))
            ? gtfsrt_resolve_run(today, tt, &rtt, src, td).first
            : gtfsrt_vp_resolve_run(tt, src, vp,
                                    delay_prediction->vehicle_trip_match);

    if (!r.valid()) {
      log(log_lvl::info, "rt.gtfs.unsupported",
          R"(Run is not valid. Eventual information were saved. Skipping Message)",
          tag, entity.id());
      ++stats.vehicle_position_without_matching_run_;
      return;
    }

    if (!r.is_scheduled()) {
      log(log_lvl::info, "rt.gtfs.unsupported",
          R"(Run is not scheduled. Skipping Message)", tag, entity.id());
      return;
    }

    // add rt_transport if not existent
    if (!r.is_rt()) {
      r.rt_ = rtt.add_rt_transport(src, tt, r.t_);
    }

    key key{r.t_.t_idx_, src};

    auto const location_seq =
        std::span{tt.route_location_seq_[tt.transport_route_[r.t_.t_idx_]]};

    auto const location_idx_seq = to_vec(location_seq, [&](auto const& stp) {
      return stop{stp}.location_idx();
    });

    auto const coord_seq_idx =
        delay_prediction->hist_trip_time_store->match_trip_to_coord_seq(
            tt, key, location_idx_seq);

    auto const [current_segment, current_progress, nearest_stop] =
        delay_prediction->hist_trip_time_store->get_segment_progress(
            tt, vehicle_position, coord_seq_idx);

    auto const vp_ts = vp.has_timestamp()
                           ? unixtime_t{std::chrono::duration_cast<i32_minutes>(
                                 std::chrono::seconds{vp.timestamp()})}
                           : std::chrono::time_point_cast<i32_minutes>(
                                 std::chrono::system_clock::now());

    trip_seg_data const new_tsd{current_segment, current_progress, vp_ts,
                                vehicle_position};

    auto const ut_start_time = tt.event_time(r.t_, 0, event_type::kDep);

    auto const current_ttd_idx_it =
        utl::find_if(delay_prediction->hist_trip_time_store
                         ->coord_seq_idx_ttd_[coord_seq_idx],
                     [&](auto current_ttd_idx) {
                       return delay_prediction->hist_trip_time_store
                                  ->ttd_idx_trip_time_data_[current_ttd_idx]
                                  .start_timestamp == ut_start_time;
                     });

    if (current_ttd_idx_it != end(delay_prediction->hist_trip_time_store
                                      ->coord_seq_idx_ttd_[coord_seq_idx])) {
      auto& current_ttd = delay_prediction->hist_trip_time_store
                              ->ttd_idx_trip_time_data_[*current_ttd_idx_it];
      //just for safety - should never happen
      if (current_ttd.seg_data_.empty()) {
        return;
      }
      // calculate time since last tsd and add to time spent at stop/segment
      auto const last_tsd = current_ttd.seg_data_.end() - 1;

      auto const delta = new_tsd.timestamp - last_tsd->timestamp;
      // If the vehicle is near the stop, add time since last tsd to the stop
      // duration, otherwise to the segment duration
      if (geo::approx_squared_distance(
              last_tsd->position, new_tsd.position,
              geo::approx_distance_lng_degrees(last_tsd->position)) < 20 &&
          geo::approx_squared_distance(
              new_tsd.position, nearest_stop,
              geo::approx_distance_lng_degrees(new_tsd.position)) < 20) {
        auto const stop_to_add_to = current_progress < 0.5 ? current_segment.v_ : current_segment.v_ + 1;
        if (stop_to_add_to < current_ttd.stop_durations_.size()) {
          current_ttd.stop_durations_[stop_to_add_to] += delta;
        }
      } else {
        current_ttd.segment_durations_[current_segment.v_] += delta;
      }
      current_ttd.seg_data_.emplace_back(new_tsd);
    }

    auto const mode_div = delay_prediction->mode == hist_trip_mode::kSameDay ? 10080 : 1440;

    // create kalman here (before creation of new ttd), so it is easier to
    // calculate averages
    auto& kalman =
        delay_prediction->delay_prediction_store->get_or_create_kalman(
            key, ut_start_time, delay_prediction->number_of_predecessors,
            delay_prediction->number_of_hist_trips,
            mode_div,
            delay_prediction->hist_trip_time_store);

    if (current_ttd_idx_it == end(delay_prediction->hist_trip_time_store
                                      ->coord_seq_idx_ttd_[coord_seq_idx])) {
      trip_time_data const new_ttd{ut_start_time,
                                   {new_tsd},
                                   static_cast<uint32_t>(location_seq.size())};
      delay_prediction->hist_trip_time_store->coord_seq_idx_ttd_[coord_seq_idx]
          .push_back(
              trip_time_data_idx_t{delay_prediction->hist_trip_time_store
                                       ->ttd_idx_trip_time_data_.size()});
      delay_prediction->hist_trip_time_store->ttd_idx_trip_time_data_
          .emplace_back(new_ttd);
    }

    // calculation of delay prediction
    if (kalman.predecessors_.empty() || kalman.hist_trips_.empty()) {
      calculate_delay_simple(tt, rtt, src, tag, entity, today, message_time,
                             span, stats);
      return;
    }

    // calculation of remaining time till next stop of predecessor at same place
    vector<duration_t> pred_remaining_times;
    for (auto pred_trip_idx : kalman.predecessors_) {
      auto const& pred_trip = delay_prediction->hist_trip_time_store
                                  ->ttd_idx_trip_time_data_[pred_trip_idx];
      trip_seg_data const* pred_tsd_candidate_hist = nullptr;
      for (auto const& tsd_to_check : pred_trip.seg_data_) {
        if (tsd_to_check.seg_idx == current_segment &&
            tsd_to_check.progress <= current_progress &&
            (pred_tsd_candidate_hist == nullptr ||
             tsd_to_check.progress > pred_tsd_candidate_hist->progress)) {
          pred_tsd_candidate_hist = &tsd_to_check;
        }
      }
      if (pred_tsd_candidate_hist != nullptr) {
        pred_remaining_times.emplace_back(
            hist_trip_times_storage::get_remaining_time_till_next_stop(
                pred_tsd_candidate_hist, &pred_trip));
      }
    }

    auto const avg_pred_trips_remaining_time =
        delay_prediction_storage::get_avg_duration(pred_remaining_times);

    // calculation of average remaining time till next stop of historic trips at
    // same place
    vector<duration_t> hist_remaining_times;
    for (auto hist_trip_idx : kalman.hist_trips_) {
      auto const& hist_trip = delay_prediction->hist_trip_time_store
                                  ->ttd_idx_trip_time_data_[hist_trip_idx];
      trip_seg_data const* pred_tsd_candidate_hist = nullptr;
      for (auto const& tsd_to_check : hist_trip.seg_data_) {
        if (tsd_to_check.seg_idx == current_segment &&
            tsd_to_check.progress <= current_progress &&
            (pred_tsd_candidate_hist == nullptr ||
             tsd_to_check.progress > pred_tsd_candidate_hist->progress)) {
          pred_tsd_candidate_hist = &tsd_to_check;
        }
      }
      if (pred_tsd_candidate_hist != nullptr) {
        hist_remaining_times.emplace_back(
            hist_trip_times_storage::get_remaining_time_till_next_stop(
                pred_tsd_candidate_hist, &hist_trip));
      }
    }

    auto const avg_hist_trips_remaining_time =
        delay_prediction_storage::get_avg_duration(hist_remaining_times);

    // Kalman calculation ...

    // ... next stop
    auto const next_stop = static_cast<stop_idx_t>(current_segment.v_ + 1);

    double hist_variance = 0;
    if (hist_remaining_times.size() > 1) {
      for (auto const hist_del : hist_remaining_times) {
        hist_variance +=
            (hist_del.count() - avg_hist_trips_remaining_time.count()) *
            (hist_del.count() - avg_hist_trips_remaining_time.count());
      }
      hist_variance /= hist_remaining_times.size();
    }

    kalman.filter_gain = kalman.error == 0 && hist_variance == 0 ? 1 :
        (kalman.error + hist_variance) / (kalman.error + 2 * hist_variance);
    kalman.gain_loop = 1 - kalman.filter_gain;
    kalman.error = hist_variance * kalman.filter_gain;

    // delay for arrival at next stop = now + prognosed driving time to next
    // stop - scheduled arrival time at next stop
    auto delay = new_tsd.timestamp +
                 std::chrono::duration_cast<duration_t>(
                     kalman.gain_loop * avg_pred_trips_remaining_time +
                     kalman.filter_gain * avg_hist_trips_remaining_time) -
                 tt.event_time(r.t_, next_stop, event_type::kArr);

    delay = std::chrono::duration_cast<duration_t>(delay);

    auto const& rtt_const = rtt;
    auto const fr = frun::from_t(tt, &rtt_const, r.t_);
    auto const stops_after_next_stop = interval{next_stop, fr.stop_range_.to_};

    // negative delay for an arrival can not lead to a time previous to the
    // departure at the previous stop negative delay for a departure is not
    // possible at all

    // update delay for arrival at next stop
    if (next_stop != *fr.stop_range_.begin()) {
      update_delay(tt, rtt, r, next_stop, event_type::kArr, delay,
                     rtt.unix_event_time(r.rt_, next_stop - 1, event_type::kDep));
    }

    // update delay for stops after next stop
    auto const [avg_pred_stop_durations, avg_pred_segment_durations] =
        delay_prediction_storage::get_avg_stop_segment_durations(
            delay_prediction->hist_trip_time_store, kalman.predecessors_);

    for (auto const [first, second] : utl::pairwise(stops_after_next_stop)) {
      delay +=
          std::chrono::duration_cast<duration_t>(
              kalman.gain_loop * avg_pred_stop_durations[first] +
              kalman.filter_gain * kalman.hist_avg_stop_durations_[first]) -
          (tt.event_time(r.t_, first, event_type::kDep) -
           tt.event_time(r.t_, first, event_type::kArr));

      update_delay(tt, rtt, r, first, event_type::kDep, delay,
          rtt.unix_event_time(r.rt_, first, event_type::kArr));

      delay +=
          std::chrono::duration_cast<duration_t>(
              kalman.gain_loop * avg_pred_segment_durations[first] +
              kalman.filter_gain * kalman.hist_avg_segment_durations_[first]) -
          (tt.event_time(r.t_, second, event_type::kArr) -
           tt.event_time(r.t_, first, event_type::kDep));

      update_delay(tt, rtt, r, second, event_type::kArr, delay,
      rtt.unix_event_time(r.rt_, first, event_type::kDep));
    }
    ++stats.total_entities_success_;
  } catch (std::exception const& e) {
    ++stats.total_entities_fail_;
    log(log_lvl::debug, "rt.gtfs",
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
                             delay_prediction* delay_prediction) {
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
        log(log_lvl::debug, "rt.gtfs.unsupported",
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

    if (delay_prediction != nullptr) {
      ++stats.total_vehicles_;
      if (!entity.has_vehicle()) {
        log(log_lvl::debug, "rt.gtfs.unsupported",
            R"(unsupported: no "vehicle_position" field (tag={}, id={}), skipping message)",
            tag, entity.id());
        ++stats.no_vehicle_position_;
      } else if (!entity.vehicle().has_position()) {
        log(log_lvl::debug, "rt.gtfs.unsupported",
            R"(unsupported: no "position" field in "vehicle_position" field (tag={}, id={}), skipping message)",
            tag, entity.id());
        ++stats.vehicle_position_without_position_;
      } else if (delay_prediction->algo == algorithm::kIntelligent) {
        calculate_delay_intelligent(tt, rtt, src, tag, entity, today,
                                    message_time, span, stats,
                                    delay_prediction);
      } else if (!entity.vehicle().has_trip()) {
        log(log_lvl::debug, "rt.gtfs.unsupported",
            R"(unsupported: no "trip" field in "vehicle_position" field (tag={}, id={}), skipping message)",
            tag, entity.id());
        ++stats.vehicle_position_without_trip_;
      } else if (!entity.vehicle().trip().has_trip_id()) {
        log(log_lvl::debug, "rt.gtfs.unsupported",
            R"(unsupported: no "trip_id" field in "trip" field (tag={}, id={}), skipping message)",
            tag, entity.id());
        ++stats.vehicle_position_trip_without_trip_id_;
      } else {
        calculate_delay_simple(tt, rtt, src, tag, entity, today, message_time,
                               span, stats);
      }
      continue;
    }

    if (!entity.has_trip_update()) {
      log(log_lvl::debug, "rt.gtfs.unsupported",
          R"(unsupported: no "trip_update" field (tag={}, id={}, vehicle={}), skipping message)",
          tag, entity.id(), entity.has_vehicle());
      ++stats.no_trip_update_;
      continue;
    }

    if (!entity.trip_update().has_trip()) {
      log(log_lvl::debug, "rt.gtfs.unsupported",
          R"(unsupported: no "trip" field in "trip_update" field (tag={}, id={}), skipping message)",
          tag, entity.id());
      ++stats.trip_update_without_trip_;
      continue;
    }

    if (!entity.trip_update().trip().has_trip_id() &&
        !(entity.trip_update().trip().has_schedule_relationship() &&
          (entity.trip_update().trip().schedule_relationship() ==
               transit_realtime::
                   TripDescriptor_ScheduleRelationship_SCHEDULED ||
           entity.trip_update().trip().schedule_relationship() ==
               transit_realtime::
                   TripDescriptor_ScheduleRelationship_CANCELED) &&
          entity.trip_update().trip().has_start_date() &&
          entity.trip_update().trip().has_start_time() &&
          entity.trip_update().trip().has_route_id() &&
          entity.trip_update().trip().has_direction_id())) {
      log(log_lvl::debug, "rt.gtfs.unsupported",
          R"(unsupported: no "trip_id" field in "trip_update.trip" (tag={}, td={}), skipping message)",
          tag, entity.trip_update().trip().DebugString());
      ++stats.unsupported_no_trip_id_;
      continue;
    }

    auto const sr = entity.trip_update().trip().has_schedule_relationship()
                        ? entity.trip_update().trip().schedule_relationship()
                        : gtfsrt::TripDescriptor_ScheduleRelationship_SCHEDULED;

    if (sr == gtfsrt::TripDescriptor_ScheduleRelationship_DUPLICATED &&
        (!entity.trip_update().has_trip_properties() ||
         !entity.trip_update().trip_properties().has_trip_id())) {
      log(log_lvl::debug, "rt.gtfs.unsupported",
          R"(unsupported: no "trip_properties.trip_id" field in "trip_update.trip" for DUPLICATED (tag={}, id={}), skipping message)",
          tag, entity.id());
      ++stats.unsupported_no_trip_id_;
      continue;
    }

    auto const added = is_added(sr);
    // auto const added_with_ref = is_added_with_ref(sr);

    if (sr != gtfsrt::TripDescriptor_ScheduleRelationship_SCHEDULED &&
        sr != gtfsrt::TripDescriptor_ScheduleRelationship_CANCELED && !added) {
      log(log_lvl::debug, "rt.gtfs.unsupported",
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

        resolve_rt(rtt, r, trip_id, src);

        if (sr == gtfsrt::TripDescriptor_ScheduleRelationship_CANCELED) {
          rtt.cancel_run(r);
          ++stats.total_entities_success_;
        } else if (!added) {
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
        resolve_rt(rtt, r, trip_id.empty() ? td.trip_id() : trip_id, src);
        if (update_run(src, tt, rtt, trip_idx_t::invalid(), r,
                       entity.trip_update())) {
          ++stats.total_entities_success_;
        }
        continue;
      }

      if (!is_resolved_static) {
        log(log_lvl::debug, "rt.gtfs.resolve", "could not resolve (tag={}) {}",
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
      // add matching-information (vehicle -> run) if needed
      if (delay_prediction != nullptr &&
          delay_prediction->algo == algorithm::kIntelligent &&
          entity.trip_update().has_vehicle() &&
          entity.trip_update().vehicle().has_id()) {
        auto const lb = std::lower_bound(
            begin(delay_prediction->vehicle_trip_match->vehicle_id_to_idx_),
            end(delay_prediction->vehicle_trip_match->vehicle_id_to_idx_),
            entity.trip_update().vehicle().id(),
            [&](pair<vehicle_id_idx_t, vehicle_idx_t> const& a, auto&& b) {
              return std::tuple{delay_prediction->vehicle_trip_match
                                    ->vehicle_id_src_[a.first],
                                delay_prediction->vehicle_trip_match
                                    ->vehicle_id_strings_[a.first]
                                    .view()} <
                     std::tuple{src, static_cast<std::string_view>(b)};
            });

        auto const id_matches = [&](vehicle_id_idx_t const v_id_idx) {
          return delay_prediction->vehicle_trip_match
                         ->vehicle_id_src_[v_id_idx] == src &&
                 delay_prediction->vehicle_trip_match
                         ->vehicle_id_strings_[v_id_idx]
                         .view() == entity.trip_update().vehicle().id();
        };

        std::chrono::sys_seconds now{
            std::chrono::time_point_cast<std::chrono::seconds>(
                std::chrono::system_clock::now())};
        auto vehicle_idx = lb->second;
        if (lb !=
                end(delay_prediction->vehicle_trip_match->vehicle_id_to_idx_) &&
            id_matches(lb->first)) {
          // if we don't already have a run for this vehicle
          if (delay_prediction->vehicle_trip_match->vehicle_idx_run_.count(
                  lb->second) == 0) {
            delay_prediction->vehicle_trip_match->vehicle_idx_run_.emplace(
                vehicle_idx,
                gtfsrt_resolve_run(today, tt, &rtt, src, td).first);
            delay_prediction->vehicle_trip_match->vehicle_idx_last_access_
                .emplace(vehicle_idx, now);
          }
        } else {
          vehicle_idx = vehicle_idx_t{
              delay_prediction->vehicle_trip_match->vehicle_id_to_idx_.size() -
              1};
          delay_prediction->vehicle_trip_match->vehicle_id_strings_
              .emplace_back(entity.trip_update().vehicle().id());
          delay_prediction->vehicle_trip_match->vehicle_id_src_.emplace_back(
              src);
          delay_prediction->vehicle_trip_match->vehicle_id_to_idx_.emplace_back(
              vehicle_id_idx_t{delay_prediction->vehicle_trip_match
                                   ->vehicle_id_strings_.size() -
                               1},
              vehicle_idx);
          delay_prediction->vehicle_trip_match->vehicle_idx_run_.emplace(
              vehicle_idx, gtfsrt_resolve_run(today, tt, &rtt, src, td).first);
          delay_prediction->vehicle_trip_match->vehicle_idx_last_access_
              .emplace(vehicle_idx, now);
        }
      }

    } catch (std::exception const& e) {
      ++stats.total_entities_fail_;
      log(log_lvl::debug, "rt.gtfs",
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
                             gtfsrt::FeedMessage& msg,
                             delay_prediction* delay_prediction) {
  msg.Clear();

  auto const success =
      msg.ParseFromArray(protobuf.data(), static_cast<int>(protobuf.size()));
  if (!success) {
    log(log_lvl::debug, "rt.gtfs",
        "GTFS-RT error (tag={}): unable to parse protobuf message: {}", tag,
        protobuf.substr(0, std::min(protobuf.size(), size_t{1000U})));
    return {.parser_error_ = true};
  }

  return gtfsrt_update_msg(tt, rtt, src, tag, msg, delay_prediction);
}

statistics gtfsrt_update_buf(timetable const& tt,
                             rt_timetable& rtt,
                             source_idx_t const src,
                             std::string_view tag,
                             std::string_view protobuf,
                             delay_prediction* delay_prediction) {
  auto msg = gtfsrt::FeedMessage{};
  return gtfsrt_update_buf(tt, rtt, src, tag, protobuf, msg, delay_prediction);
}

}  // namespace nigiri::rt
