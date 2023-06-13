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
      source_idx_t const src,
      timetable const& tt,
      transport const t,
      std::span<stop::value_type> const& stop_seq = {},
      std::span<delta_t> const& time_seq = {}) {
    auto const [t_idx, day] = t;

    auto const rt_t_idx = rt_transport_src_.size();
    auto const rt_t = rt_transport_idx_t{rt_t_idx};
    static_trip_lookup_.emplace(t, rt_t_idx);
    rt_transport_static_transport_.emplace_back(t);

    auto const static_bf = bitfields_[transport_traffic_days_[t_idx]];
    bitfields_.emplace_back(static_bf).set(to_idx(day), false);
    transport_traffic_days_[t_idx] = bitfield_idx_t{bitfields_.size() - 1U};

    auto const location_seq =
        stop_seq.empty()
            ? std::span{tt.route_location_seq_[tt.transport_route_[t_idx]]}
            : stop_seq;
    rt_transport_location_seq_.emplace_back(location_seq);
    rt_transport_src_.emplace_back(src);
    rt_transport_train_nr_.emplace_back(0U);

    for (auto const s : location_seq) {
      auto rt_transports = location_rt_transports_[stop{s}.location_idx()];
      if (rt_transports.empty() || rt_transports.back() != rt_t) {
        rt_transports.push_back(rt_t);
      }
    }

    if (time_seq.empty()) {
      auto times = rt_transport_stop_times_.add_back_sized(
          location_seq.size() * 2U - 2U);
      auto i = 0U;
      auto stop_idx = stop_idx_t{0U};
      for (auto const [a, b] : utl::pairwise(location_seq)) {
        CISTA_UNUSED_PARAM(a)
        CISTA_UNUSED_PARAM(b)
        times[i++] =
            unix_to_delta(tt.event_time(t, stop_idx, event_type::kDep));
        times[i++] =
            unix_to_delta(tt.event_time(t, ++stop_idx, event_type::kArr));
      }
    } else {
      rt_transport_stop_times_.emplace_back(time_seq);
    }

    rt_transport_display_names_.add_back_sized(0U);
    rt_transport_section_clasz_.add_back_sized(0U);

    assert(static_trip_lookup_.contains(t));
    assert(rt_transport_static_transport_[rt_transport_idx_t{rt_t_idx}] == t);
    assert(rt_transport_static_transport_.size() == rt_t_idx + 1U);
    assert(rt_transport_src_.size() == rt_t_idx + 1U);
    assert(rt_transport_stop_times_.size() == rt_t_idx + 1U);
    assert(rt_transport_location_seq_.size() == rt_t_idx + 1U);
    assert(rt_transport_display_names_.size() == rt_t_idx + 1U);
    assert(rt_transport_section_clasz_.size() == rt_t_idx + 1U);

    return rt_transport_idx_t{rt_t_idx};
  }

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
    rt_transport_stop_times_[rt_t][stop_idx * 2 -
                                   (ev_type == event_type::kArr ? 1 : 0)] =
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

  // RT transport -> vehicle clasz for each section
  vecvec<rt_transport_idx_t, clasz> rt_transport_section_clasz_;
};

}  // namespace nigiri