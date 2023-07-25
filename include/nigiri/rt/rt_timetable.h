#pragma once

#include "utl/pairwise.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/common/interval.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

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
  rt_transport_idx_t add_rt_transport(
      source_idx_t const,
      timetable const&,
      transport const,
      std::span<stop::value_type> const& stop_seq = {},
      std::span<delta_t> const& time_seq = {});

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
    if (rt_transport_display_names_[t].empty()) {
      return rt_transport_static_transport_[t].apply(utl::overloaded{
          [&](transport const x) { return tt.transport_name(x.t_idx_); },
          [&](rt_add_trip_id_idx_t) { return std::string_view{"?"}; }});
    } else {
      return rt_transport_display_names_[t].view();
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

  // Updated transport traffic days from the static timetable.
  // Initial: 100% copy from static, then adapted according to real-time
  // updates
  vector_map<transport_idx_t, bitfield_idx_t> transport_traffic_days_;
  vector_map<bitfield_idx_t, bitfield> bitfields_;

  // Location -> RT transports that stop at this location
  vecvec<location_idx_t, rt_transport_idx_t> location_rt_transports_;

  // Base-day: all real-time timestamps (departures + arrivals in
  // rt_transport_stop_times_) are given relative to this base day.
  date::sys_days base_day_;
  day_idx_t base_day_idx_;

  // Lookup: static transport -> realtime transport
  // only works for transport that existed in the static timetable
  hash_map<transport, rt_transport_idx_t> static_trip_lookup_;

  // Lookup: additional trip index -> realtime transport
  hash_map<rt_add_trip_id_idx_t, rt_transport_idx_t> additional_trips_lookup_;

  // RT transport -> static transport (not for additional trips)
  vector_map<rt_transport_idx_t, variant<transport, rt_add_trip_id_idx_t>>
      rt_transport_static_transport_;

  // RT trip ID index -> ID strings + source
  vecvec<rt_add_trip_id_idx_t, char> trip_id_strings_;
  vector_map<rt_transport_idx_t, source_idx_t> rt_transport_src_;

  // RT trip ID index -> train number, if available (otherwise 0)
  vector_map<rt_transport_idx_t, std::uint32_t> rt_transport_train_nr_;

  // RT transport -> event times (dep, arr, dep, arr, ...)
  vecvec<rt_transport_idx_t, delta_t> rt_transport_stop_times_;
  vecvec<rt_transport_idx_t, stop::value_type> rt_transport_location_seq_;

  // RT trip index -> display name (empty if not changed)
  vecvec<rt_transport_idx_t, char> rt_transport_display_names_;
  vecvec<rt_transport_idx_t, char> rt_transport_line_;

  // RT transport -> vehicle clasz for each section
  vecvec<rt_transport_idx_t, clasz> rt_transport_section_clasz_;

  // RT transport -> canceled flag
  bitvec rt_transport_is_cancelled_;
};

}  // namespace nigiri