#pragma once

#include "utl/pairwise.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string_view>
#include <variant>

#include "utl/visit.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/common/interval.h"
#include "nigiri/rt/run.h"
#include "nigiri/rt/service_alert.h"
#include "nigiri/stop.h"
#include "nigiri/string_store.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

// If set, the bitfield has to be looked up in the RT timetable.
constexpr auto const kRtBitfieldFlag = std::uint32_t{0x8000'0000U};

using change_callback_t =
    std::function<void(transport const transport,
                       stop_idx_t const stop_idx,
                       event_type const ev_type,
                       std::optional<location_idx_t> const location_idx,
                       std::optional<bool> const in_out_allowed,
                       std::optional<duration_t> const delay)>;

// General note:
// - To deactivate bits for static transports that are updated with delays,
//   rerouting (incl. track changes) or cancellations (without changing the
//   static timetable), it appends a modified copy of the static bitfield to its
//   own `bitfields_` and re-points the `transport_traffic_days_` entry (tagged
//   with `kRtBitfieldFlag`) at it.
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
                                      direction_id_t = {},
                                      std::string_view trip_short_name = {},
                                      delta_t = 0);

  delta_t unix_to_delta(unixtime_t const t) const {
    auto const d =
        (t - std::chrono::time_point_cast<unixtime_t::duration>(base_day_))
            .count();
    return clamp(d);
  }

  bool affects(interval<unixtime_t> const& itv,
               profile_idx_t const prf_idx) const noexcept {
    assert(itv.from_ < itv.to_);
    return prf_idx != 0U || itv.overlaps(deviation_interval_);
  }

  void extend_deviation_interval(unixtime_t const t) {
    extend_deviation_interval(interval{t, t + unixtime_t::duration{1}});
  }

  void extend_deviation_interval(interval<unixtime_t> const i) {
    if (i.from_ >= i.to_) {
      return;  // empty
    }
    deviation_interval_ =
        deviation_interval_.empty()
            ? i
            : interval{std::min(deviation_interval_.from_, i.from_),
                       std::max(deviation_interval_.to_, i.to_)};
  }

  // Interval covered by the event times of `rt_t`.
  interval<unixtime_t> rt_transport_interval(
      rt_transport_idx_t const rt_t) const {
    auto const times = rt_transport_stop_times_[rt_t];
    if (times.empty()) {
      return {};
    }
    auto const [min, max] = std::minmax_element(begin(times), end(times));
    return {delta_to_unix(base_day_, *min),
            delta_to_unix(base_day_, *max) + unixtime_t::duration{1}};
  }

  // Interval covered by the event times of the static transport `t`.
  interval<unixtime_t> static_transport_interval(timetable const&,
                                                 transport) const;

  interval<unixtime_t> static_transport_interval(transport const t) const {
    assert(tt_ != nullptr);
    return static_transport_interval(*tt_, t);
  }

  // Sets one event time. If this is the first event of `rt_t` to move off its
  // scheduled time, the transport is taken off the static scan and put on the
  // real-time scan in the same step -- see mark_deviating().
  void update_time(rt_transport_idx_t, stop_idx_t, event_type, unixtime_t);

  // Cancels one stop (no boarding, no alighting). Also a deviation: the stop
  // sequence no longer matches the static route's.
  void cancel_stop(rt_transport_idx_t, stop_idx_t);

  // Call after applying an update to `rt_t`, once its times are final.
  //
  // A real-time update that changes nothing -- a feed covering a trip and
  // reporting it exactly on time, which is most of a real feed -- still
  // materialises an rt_transport, and add_rt_transport() used to clear the
  // trip's static traffic day, moving it onto the per-transport real-time
  // scan. For a transport identical to its schedule the static scan gives the
  // same answer far more cheaply, so it stays there instead.
  //
  // Idempotent and authoritative in both directions: a transport that was
  // punctual and is now delayed gets its traffic day cleared and joins the
  // real-time scan; one that becomes punctual again is handed back.
  void finalize_rt_transport(timetable const&, rt_transport_idx_t);

  // Takes the transport's day off the static scan *and* puts it on the
  // real-time scan. These always happen together: a transport whose traffic
  // day is cleared while it is on no scan is invisible to the routing, which
  // is silent data loss rather than a degradation. Keeping the pair inside one
  // function makes that state unrepresentable.
  void mark_deviating(rt_transport_idx_t);

  // Sets or clears one day of `t` in the rt traffic days, creating the
  // rt-owned bitfield copy on first use.
  void set_rt_traffic_day(transport t, bool active);

  // Registers `rt_t` on location_rt_scan_, the list the routing builds its
  // marks from, for every location the transport currently stops at. Called
  // whenever the transport is (re)marked as deviating. The scheduled
  // locations are registered once; the current ones are reconciled on every
  // call, because a track change (gtfsrt_update.cc) moves a stop to another
  // platform, i.e. another location, and a later update can move it back.
  // A transport that becomes identical to schedule again stays registered
  // and is filtered out by is_unchanged() when the marks are built.
  void register_scan(rt_transport_idx_t);

  // True if `rt_t` is identical to its static counterpart *for routing*, and
  // therefore ridden on the static scan. Track-only changes still count as
  // unchanged: they do not affect routing, and frun still surfaces them.
  bool is_unchanged(rt_transport_idx_t const rt_t) const noexcept {
    return to_idx(rt_t) < rt_transport_is_unchanged_.size() &&
           rt_transport_is_unchanged_.test(to_idx(rt_t));
  }

  void set_unchanged(rt_transport_idx_t const rt_t, bool const x) {
    if (to_idx(rt_t) >= rt_transport_is_unchanged_.size()) {
      rt_transport_is_unchanged_.resize(to_idx(rt_t) + 1U);
    }
    rt_transport_is_unchanged_.set(to_idx(rt_t), x);
  }

  bool matches_schedule(timetable const&, rt_transport_idx_t) const;

  void update_lbs(timetable const& tt,
                  rt_transport_idx_t,
                  stop_idx_t,
                  vector_map<location_idx_t, std::vector<footpath>>&,
                  vector_map<location_idx_t, std::vector<footpath>>&);
  void update_lbs(timetable const& tt);

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

  void set_track(rt_transport_idx_t, stop_idx_t, event_type, std::string_view);
  std::optional<std::string_view> get_track(rt_transport_idx_t,
                                            stop_idx_t,
                                            event_type) const;

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

  std::variant<translation_idx_t, std::string_view> trip_short_name(
      timetable const& tt, rt_transport_idx_t const t) const {
    if (rt_transport_trip_short_names_[t].empty()) {
      return rt_transport_static_transport_[t].apply(utl::overloaded{
          [&](transport const x)
              -> std::variant<translation_idx_t, std::string_view> {
            auto const trip_idx =
                tt.merged_trips_[tt.transport_to_trip_section_[x.t_idx_]
                                     .front()]
                    .front();
            return tt.trip_display_names_[trip_idx];
          },
          [&](rt_add_trip_id_idx_t)
              -> std::variant<translation_idx_t, std::string_view> {
            return std::string_view{"?"};
          }});
    } else {
      return rt_transport_trip_short_names_[t].view();
    }
  }

  std::string_view default_trip_short_name(timetable const& tt,
                                           rt_transport_idx_t const t) const {
    return utl::visit(
        trip_short_name(tt, t),
        [&](translation_idx_t x) { return tt.get_default_translation(x); },
        [](std::string_view x) { return x; });
  }

  std::string_view transport_name(timetable const& tt,
                                  rt_transport_idx_t const t) const;

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

  bool is_flag_set(route_flag const f, rt_transport_idx_t const r) const {
    return rt_transport_flags_[f][to_idx(r) * 2U] ||
           rt_transport_flags_[f][to_idx(r) * 2U + 1U];
  }

  bitfield const& traffic_days(bitfield_idx_t const i) const {
    return (to_idx(i) & kRtBitfieldFlag) != 0U
               ? bitfields_[bitfield_idx_t{to_idx(i) & ~kRtBitfieldFlag}]
               : tt_->bitfields_[i];
  }

  bitfield_idx_t rt_bitfield_idx() const {
    return bitfield_idx_t{static_cast<bitfield_idx_t::value_t>(
        (bitfields_.size() - 1U) | kRtBitfieldFlag)};
  }

  bool is_transport_active(transport_idx_t const t, day_idx_t const day) const {
    return traffic_days(transport_traffic_days_[t]).test(to_idx(day));
  }

  timetable const* tt_{nullptr};

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

  // RT transport -> static transport (not for additional trips)
  vector_map<rt_transport_idx_t, variant<transport, rt_add_trip_id_idx_t>>
      rt_transport_static_transport_;

  struct additional_trips {
    string_store<rt_add_trip_id_idx_t> ids_;
    vector_map<rt_add_trip_id_idx_t, rt_transport_idx_t> transports_;
  };
  vector_map<source_idx_t, additional_trips> additional_trips_;

  vector_map<rt_transport_idx_t, source_idx_t> rt_transport_src_;

  bitvec_map<rt_transport_idx_t> rt_transport_direction_id_;
  vector_map<rt_transport_idx_t, route_id_idx_t> rt_transport_route_id_;

  // RT transport -> direction for each section
  vecvec<trip_direction_string_idx_t, char> rt_transport_direction_strings_;
  vecvec<rt_transport_idx_t, trip_direction_string_idx_t>
      rt_transport_section_directions_;

  // RT transport -> event times (dep, arr, dep, arr, ...)
  vecvec<rt_transport_idx_t, delta_t> rt_transport_stop_times_;
  vecvec<rt_transport_idx_t, stop::value_type> rt_transport_location_seq_;

  // RT transport -> real-time track (dep, arr, dep, arr, ...)
  // - empty if no overwrite exists at all
  // - track_idx_t::invalid for stops where no overwrite exists
  vecvec<rt_transport_idx_t, track_idx_t> rt_transport_track_sequence_;
  string_store<track_idx_t> track_strings_;

  // RT trip index -> display name (empty if not changed)
  vecvec<rt_transport_idx_t, char> rt_transport_trip_short_names_;
  vecvec<rt_transport_idx_t, char> rt_transport_line_;

  // RT transport -> vehicle clasz for each section
  vecvec<rt_transport_idx_t, clasz> rt_transport_section_clasz_;

  // RT transport -> canceled flag
  bitvec rt_transport_is_cancelled_;

  // Set for rt transports identical to their static counterpart, see
  // finalize_rt_transport(). Those ride the static scan and are on no list
  // below, so they cost the routing nothing.
  bitvec rt_transport_is_unchanged_;

  // The rt transports the routing has to scan, i.e. the ones that deviate.
  // Building marks from this instead of from location_rt_transports_ keeps the
  // punctual majority out of the mark-building walk entirely. Maintained by
  // register_scan(); must list a transport under every location it currently
  // stops at (platform changes included), otherwise the routing cannot board
  // it there.
  mutable_fws_multimap<location_idx_t, rt_transport_idx_t> location_rt_scan_;
  // Set once the scheduled locations of the transport are on location_rt_scan_.
  bitvec rt_transport_scan_registered_;

  // RT transport * 2 -> flags (bikes, cars, wheelchairs, reservtion) along the
  // transport RT transport * 2 + 1 -> flags along parts of the transport
  std::array<bitvec, kNumRouteFlags> rt_transport_flags_;

  // RT transport -> flags for each section
  std::array<vecvec<rt_transport_idx_t, bool>, kNumRouteFlags>
      rt_flags_per_section_;

  // Service alerts
  alerts alerts_;

  change_callback_t change_callback_;

  // Lower bound graph extension.
  bitvec_map<location_idx_t> fwd_search_lb_graph_has_edges_;
  bitvec_map<location_idx_t> bwd_search_lb_graph_has_edges_;
  vecvec<location_idx_t, footpath> fwd_search_lb_graph_;
  vecvec<location_idx_t, footpath> bwd_search_lb_graph_;

  // Incremental mode requires copyability of rt_timetable.
  // unique_ptr alone would make the rt_timetable move only (not copyable)
  // -> we need a copyable wrapper
  struct gpu_rtt_slot {
    gpu_rtt_slot() = default;
    gpu_rtt_slot(gpu_rtt_slot const&) { /* start empty, upload after update */ }
    gpu_rtt_slot(gpu_rtt_slot&&) = default;
    gpu_rtt_slot& operator=(gpu_rtt_slot&&) = default;
    std::unique_ptr<void, void (*)(void*)> ptr_{nullptr, [](void*) {}};
  };
  gpu_rtt_slot gpu_rtt_;

  // Interval real-time data is available for, i.e. the hull of the events of
  // all rt transports that deviate from their schedule (scheduled + real-time
  // times) - a covered but punctual trip rides the static scan and does not
  // count. Empty (`from_ == to_`) if nothing deviates. Never shrinks. See
  // `extend_deviation_interval()`, `mark_deviating()`.
  interval<unixtime_t> deviation_interval_;
};

}  // namespace nigiri
