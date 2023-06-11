#include "nigiri/rt/gtfsrt_update.h"

#include "utl/pairwise.h"

#include "nigiri/loader/gtfs/stop_seq_number_encoding.h"
#include "nigiri/logging.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/run.h"

namespace gtfsrt = transit_realtime;
namespace pb = google::protobuf;

namespace nigiri::rt {

rt_transport_idx_t create_rt_run(
    source_idx_t const src,
    timetable const& tt,
    rt_timetable& rtt,
    transport const t,
    std::span<stop::value_type> const& stop_seq = {},
    std::span<delta_t> const& time_seq = {}) {
  auto const [t_idx, day] = t;

  auto const rt_t_idx = rtt.next_rt_transport_idx_++;
  rtt.static_trip_lookup_.emplace(t, rt_t_idx);
  rtt.rt_transport_static_transport_.emplace_back(t);

  rtt.bitfields_
      .emplace_back(rtt.bitfields_[rtt.transport_traffic_days_[t_idx]])
      .set(to_idx(day), false);
  rtt.transport_traffic_days_[t_idx] =
      bitfield_idx_t{rtt.bitfields_.size() - 1U};

  auto const location_seq =
      stop_seq.empty()
          ? std::span{tt.route_location_seq_[tt.transport_route_[t_idx]]}
          : stop_seq;
  rtt.rt_transport_location_seq_.emplace_back(location_seq);
  rtt.rt_transport_src_.emplace_back(src);
  rtt.rt_transport_train_nr_.emplace_back(0U);

  if (time_seq.empty()) {
    auto times =
        rtt.rt_transport_stop_times_.add_back_sized(location_seq.size() * 2U);
    auto i = 0U;
    auto stop_idx = stop_idx_t{0U};
    for (auto const [a, b] : utl::pairwise(location_seq)) {
      CISTA_UNUSED_PARAM(a)
      CISTA_UNUSED_PARAM(b)
      times[i++] =
          rtt.unix_to_delta(tt.event_time(t, stop_idx, event_type::kDep));
      times[i++] =
          rtt.unix_to_delta(tt.event_time(t, ++stop_idx, event_type::kArr));
    }
  } else {
    rtt.rt_transport_stop_times_.emplace_back(time_seq);
  }

  rtt.rt_transport_display_names_.add_back_sized(0U);
  rtt.rt_transport_section_clasz_.add_back_sized(0U);
  rtt.rt_transport_to_trip_section_.emplace_back(
      std::initializer_list<rt_merged_trips_idx_t>{
          rt_merged_trips_idx_t::invalid()});  // TODO(felix)

  assert(rtt.static_trip_lookup_.contains(t));
  assert(rtt.rt_transport_static_transport_[rt_t_idx] == t);
  assert(rtt.rt_transport_static_transport_.size() == to_idx(rt_t_idx) + 1U);
  assert(rtt.rt_transport_src_.size() == to_idx(rt_t_idx) + 1U);
  assert(rtt.rt_transport_stop_times_.size() == to_idx(rt_t_idx) + 1U);
  assert(rtt.rt_transport_location_seq_.size() == to_idx(rt_t_idx) + 1U);
  assert(rtt.rt_transport_display_names_.size() == to_idx(rt_t_idx) + 1U);
  assert(rtt.rt_transport_section_clasz_.size() == to_idx(rt_t_idx) + 1U);
  assert(rtt.rt_transport_to_trip_section_.size() == to_idx(rt_t_idx) + 1U);

  return rt_t_idx;
}

delta_t update_time(timetable const& tt,
                    rt_timetable& rtt,
                    run const& r,
                    stop_idx_t const stop_idx,
                    event_type const ev_type,
                    gtfsrt::TripUpdate_StopTimeEvent const& ev,
                    delta_t const min) {
  if (ev.has_time()) {
    rtt.update_time(*r.rt_, stop_idx, ev_type,
                    unixtime_t{std::chrono::duration_cast<unixtime_t::duration>(
                        std::chrono::seconds{ev.time()})});
  } else if (ev.has_delay()) {
    auto const static_time = tt.event_time(*r.t_, stop_idx, ev_type);
    rtt.update_time(
        *r.rt_, stop_idx, ev_type,
        static_time + std::chrono::duration_cast<unixtime_t::duration>(
                          std::chrono::seconds{ev.delay()}));
  }
  return rtt.event_time(*r.rt_, stop_idx, ev_type);
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
    r.rt_ = create_rt_run(src, tt, rtt, *r.t_);
  }

  auto const location_seq =
      tt.route_location_seq_[tt.transport_route_[r.t_->t_idx_]];
  auto const seq_numbers = ::nigiri::loader::gtfs::stop_seq_number_range{
      std::span{tt.transport_stop_seq_numbers_[r.t_->t_idx_]},
      static_cast<stop_idx_t>(location_seq.size())};

  auto min = delta_t{0U};
  auto prev_delay = std::optional<duration_t>{};
  auto stop_idx = stop_idx_t{0U};
  auto seq_it = begin(seq_numbers);
  auto upd_it = begin(stops);
  for (; upd_it != end(stops) && seq_it != end(seq_numbers) &&
         stop_idx != location_seq.size();
       ++stop_idx, ++seq_it) {
    if ((upd_it->has_stop_sequence() && upd_it->stop_sequence() == *seq_it) ||
        (upd_it->has_stop_id() &&
         upd_it->stop_id() ==
             tt.locations_.ids_[stop{location_seq[stop_idx]}.location_idx()]
                 .view())) {
      if (stop_idx != 0U && upd_it->has_arrival()) {
        min = update_time(tt, rtt, r, stop_idx, event_type::kArr,
                          upd_it->arrival());
      }
      if (stop_idx != location_seq.size() - 1U && upd_it->has_departure()) {
        min = update_time(tt, rtt, r, stop_idx, event_type::kDep,
                          upd_it->departure());
      }
      ++upd_it;
    } else {
      if (stop_idx != 0U) {
        min = update_time(tt, rtt, r, stop_idx, event_type::kArr, prev_delay,
                          min);
      }
      if (stop_idx != location_seq.size() - 1U) {
        min = update_time(tt, rtt, r, stop_idx, event_type::kDep, prev_delay,
                          min);
      }
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
      auto r = gtfsrt_resolve_run(today, tt, rtt, src, td);

      if (!r.valid()) {
        log(log_lvl::error, "rt.gtfs.resolve", "could not resolve {}",
            entity.trip_update().trip().DebugString());
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
          entity.DebugString(), e.what());
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