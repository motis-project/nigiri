#pragma once

#include "utl/pairwise.h"

#include <optional>
#include <string_view>

#include "nigiri/common/delta_t.h"
#include "nigiri/common/interval.h"
#include "nigiri/rt/run.h"
#include "nigiri/rt/service_alert.h"
#include "nigiri/stop.h"
#include "nigiri/string_store.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

using change_callback_t =
    std::function<void(transport const transport,
                       stop_idx_t const stop_idx,
                       event_type const ev_type,
                       std::optional<location_idx_t> const location_idx,
                       std::optional<bool> const in_out_allowed,
                       std::optional<duration_t> const delay)>;

// General note:
// - The real-time timetable does not use bitfields. It requires an initial copy
//   of the bitfields from the static timetable to be able to deactivate bits
//   for transports that are updated with delays, rerouting (incl. track
//   changes) or cancellations (without changing the static timetable).
// - RT transports represent departure and arrival times relative to a base day.
// - RT transports are currently not grouped into routes to simplify the code.
//   If this leads to performance issues during the routing, grouping into
//   routes can be introduced for real-time routing as well.
// - All RT transports can be resolved via their static transport if they were
//   already scheduled in the static timetable.
// - All RT transports that did not exist in the static timetable, can be looked
//   up with their trip_id in the RT timetable.
struct rt_timetable {
  rt_transport_idx_t add_rt_transport(source_idx_t,
                                      timetable const&,
                                      transport,
                                      std::span<stop::value_type> stop_seq = {},
                                      std::span<delta_t> time_seq = {},
                                      std::string_view new_trip_id = {},
                                      std::string_view route_id = {},
                                      std::string_view trip_short_name = {},
                                      delta_t = 0);

  delta_t unix_to_delta(unixtime_t const t) const {
    auto const d =
        (t - std::chrono::time_point_cast<unixtime_t::duration>(base_day_))
            .count();
    assert(d >= std::numeric_limits<delta_t>::min());
    assert(d <= std::numeric_limits<delta_t>::max());
    return clamp(d);
  }

  void update_time(rt_transport_idx_t const rt_t,
                   stop_idx_t const stop_idx,
                   event_type const ev_type,
                   unixtime_t const new_time) {
    auto const ev_idx = stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0);
    assert(ev_idx >= 0 && static_cast<stop_idx_t>(ev_idx) <
                              rt_transport_stop_times_[rt_t].size());
    rt_transport_stop_times_[rt_t][static_cast<std::size_t>(ev_idx)] =
        unix_to_delta(new_time);
  }

  void cancel_run(rt::run const&);

  void set_change_callback(change_callback_t callback) {
    change_callback_ = callback;
  }

  void reset_change_callback() { change_callback_ = nullptr; }

  void dispatch_event(rt::run const& r,
                      stop_idx_t const stop_idx,
                      event_type const ev_type,
                      std::optional<location_idx_t> const location_idx,
                      std::optional<bool> const in_out_allowed,
                      std::optional<duration_t> const delay) {
    if (change_callback_ &&
        ((ev_type == event_type::kArr && stop_idx != r.stop_range_.from_) ||
         (ev_type == event_type::kDep && stop_idx != r.stop_range_.to_ - 1))) {
      change_callback_(r.t_, stop_idx, ev_type, location_idx, in_out_allowed,
                       delay);
    }
  }

  void dispatch_delay(rt::run const& r,
                      stop_idx_t const stop_idx,
                      event_type const ev_type,
                      duration_t const delay) {
    dispatch_event(r, stop_idx, ev_type, std::nullopt, std::nullopt, delay);
  }

  void dispatch_stop_change(rt::run const& r,
                            stop_idx_t const stop_idx,
                            event_type const ev_type,
                            std::optional<location_idx_t> const location_idx,
                            std::optional<bool> const in_out_allowed) {
    dispatch_event(r, stop_idx, ev_type, location_idx, in_out_allowed,
                   std::nullopt);
  }

  unixtime_t unix_event_time(rt_transport_idx_t const rt_t,
                             stop_idx_t const stop_idx,
                             event_type const ev_type) const {
    return base_day_ +
           std::chrono::minutes{event_time(rt_t, stop_idx, ev_type)};
  }

  delta_t event_time(rt_transport_idx_t const rt_t,
                     stop_idx_t const stop_idx,
                     event_type const ev_type) const {
    auto const ev_idx = stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0);
    return rt_transport_stop_times_[rt_t][static_cast<unsigned>(ev_idx)];
  }

  std::string_view transport_name(timetable const& tt,
                                  rt_transport_idx_t const t) const {
    return trip_short_name(tt, t);
  }

  std::string_view trip_short_name(timetable const& tt,
                                   rt_transport_idx_t const t) const {
    if (rt_transport_trip_short_names_[t].empty()) {
      return rt_transport_static_transport_[t].apply(utl::overloaded{
          [&](transport const x) {
            auto const trip_idx =
                tt.merged_trips_[tt.transport_to_trip_section_[x.t_idx_]
                                     .front()]
                    .front();
            return tt.trip_display_names_[trip_idx].view();
          },
          [&](rt_add_trip_id_idx_t) { return std::string_view{"?"}; }});
    } else {
      return rt_transport_trip_short_names_[t].view();
    }
  }

  debug dbg(timetable const& tt, rt_transport_idx_t const t) const {
    return rt_transport_static_transport_[t].apply(
        utl::overloaded{[&](transport const x) { return tt.dbg(x.t_idx_); },
                        [&](rt_add_trip_id_idx_t) { return debug{"RT"}; }});
  }

  transport resolve_static(rt_transport_idx_t const rt_t) const noexcept {
    auto const t = rt_transport_static_transport_[rt_t];
    return holds_alternative<transport>(t) ? t.as<transport>() : transport{};
  }

  rt_transport_idx_t resolve_rt(transport const t) const noexcept {
    auto const it = static_trip_lookup_.find(t);
    return it == end(static_trip_lookup_) ? rt_transport_idx_t::invalid()
                                          : it->second;
  }

  std::uint32_t n_rt_transports() const noexcept {
    return rt_transport_src_.size();
  }

  array<bitvec_map<location_idx_t>, kNProfiles> has_td_footpaths_out_;
  array<bitvec_map<location_idx_t>, kNProfiles> has_td_footpaths_in_;
  array<vecvec<location_idx_t, td_footpath>, kNProfiles> td_footpaths_out_;
  array<vecvec<location_idx_t, td_footpath>, kNProfiles> td_footpaths_in_;

  // Updated transport traffic days from the static timetable.
  // Initial: 100% copy from static, then adapted according to real-time
  // updates
  vector_map<transport_idx_t, bitfield_idx_t> transport_traffic_days_;
  vector_map<bitfield_idx_t, bitfield> bitfields_;

  // Location -> RT transports that stop at this location
  mutable_fws_multimap<location_idx_t, rt_transport_idx_t>
      location_rt_transports_;

  // Base-day: all real-time timestamps (departures + arrivals in
  // rt_transport_stop_times_) are given relative to this base day.
  date::sys_days base_day_;
  day_idx_t base_day_idx_;

  // Lookup: static transport -> realtime transport
  // only works for transport that existed in the static timetable
  hash_map<transport, rt_transport_idx_t> static_trip_lookup_;

  // Lookup: additional trip index -> realtime transport
  vector_map<rt_add_trip_id_idx_t, rt_transport_idx_t> additional_trips_lookup_;

  // RT transport -> static transport (not for additional trips)
  vector_map<rt_transport_idx_t, variant<transport, rt_add_trip_id_idx_t>>
      rt_transport_static_transport_;

  string_store<rt_add_trip_id_idx_t> additional_trip_ids_;

  vector_map<rt_transport_idx_t, source_idx_t> rt_transport_src_;

  vector_map<rt_transport_idx_t, route_id_idx_t> rt_transport_route_id_;

  // RT transport -> direction for each section
  vecvec<trip_direction_string_idx_t, char> rt_transport_direction_strings_;
  vecvec<rt_transport_idx_t, trip_direction_string_idx_t>
      rt_transport_section_directions_;

  // RT transport -> event times (dep, arr, dep, arr, ...)
  vecvec<rt_transport_idx_t, delta_t> rt_transport_stop_times_;
  vecvec<rt_transport_idx_t, stop::value_type> rt_transport_location_seq_;

  // RT trip index -> display name (empty if not changed)
  vecvec<rt_transport_idx_t, char> rt_transport_trip_short_names_;
  vecvec<rt_transport_idx_t, char> rt_transport_line_;

  // RT transport -> vehicle clasz for each section
  vecvec<rt_transport_idx_t, clasz> rt_transport_section_clasz_;

  // RT transport -> canceled flag
  bitvec rt_transport_is_cancelled_;

  // RT transport * 2 -> bikes allowed along the transport
  // RT transport * 2 + 1 -> bikes along parts of the transport
  bitvec rt_transport_bikes_allowed_;

  // same for cars
  bitvec rt_transport_cars_allowed_;

  // RT transport -> bikes allowed for each section
  vecvec<rt_transport_idx_t, bool> rt_bikes_allowed_per_section_;

  // same for cars
  vecvec<rt_transport_idx_t, bool> rt_cars_allowed_per_section_;

  // Service alerts
  alerts alerts_;

  change_callback_t change_callback_;

  // TODO route colors?
};

}  // namespace nigiri
