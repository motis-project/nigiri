#include "nigiri/rt/rt_timetable.h"

#include <algorithm>
#include <span>

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
#include "utl/timer.h"
#include "utl/verify.h"

#include "nigiri/loader/gtfs/route.h"

namespace nigiri {

void rt_timetable::register_scan(rt_transport_idx_t const rt_t) {
  auto const t = resolve_static(rt_t);
  auto const static_seq =
      t.is_valid() && tt_ != nullptr
          ? std::span<
                stop::value_type const>{tt_->route_location_seq_
                                            [tt_->transport_route_[t.t_idx_]]}
          : std::span<stop::value_type const>{};

  // The scheduled locations are registered once (cheap: the list is only
  // appended to, so a back() check suffices while walking one sequence).
  if (to_idx(rt_t) >= rt_transport_scan_registered_.size()) {
    rt_transport_scan_registered_.resize(to_idx(rt_t) + 1U);
  }
  if (!rt_transport_scan_registered_.test(to_idx(rt_t))) {
    rt_transport_scan_registered_.set(to_idx(rt_t), true);
    for (auto const s : static_seq) {
      auto scan = location_rt_scan_[stop{s}.location_idx()];
      if (scan.empty() || scan.back() != rt_t) {
        scan.push_back(rt_t);
      }
    }
  }

  // The current locations are reconciled on every call: a GTFS-RT track
  // change moves a stop to another platform (= another location) and a later
  // update can move it back, so a snapshot taken at the first registration
  // goes stale. Only locations that differ from the scheduled one at the
  // same position need the (linear) membership test; for a scheduled copy
  // without track changes this loop does no work beyond the comparison.
  auto const cur_seq = rt_transport_location_seq_[rt_t];
  for (auto i = 0U; i != cur_seq.size(); ++i) {
    auto const l = stop{cur_seq[i]}.location_idx();
    if (i < static_seq.size() && stop{static_seq[i]}.location_idx() == l) {
      continue;  // registered with the scheduled locations above
    }
    auto scan = location_rt_scan_[l];
    if (utl::find(scan, rt_t) == end(scan)) {
      scan.push_back(rt_t);
    }
  }
}

void rt_timetable::set_rt_traffic_day(transport const t, bool const active) {
  if (!t.is_valid()) {
    return;
  }
  auto bf_idx = transport_traffic_days_[t.t_idx_];
  if ((to_idx(bf_idx) & kRtBitfieldFlag) == 0U) {
    // still pointing at the shared static bitfield: take a private copy
    bitfields_.emplace_back(traffic_days(bf_idx));
    transport_traffic_days_[t.t_idx_] = rt_bitfield_idx();
    bf_idx = transport_traffic_days_[t.t_idx_];
  }
  bitfields_[bitfield_idx_t{to_idx(bf_idx) & ~kRtBitfieldFlag}].set(
      to_idx(t.day_), active);
}

void rt_timetable::mark_deviating(rt_transport_idx_t const rt_t) {
  set_unchanged(rt_t, false);
  auto const t = resolve_static(rt_t);
  set_rt_traffic_day(t, false);
  register_scan(rt_t);

  // Deviation interval: only transports on the real-time scan can make
  // real-time data relevant to a query. A punctual covered trip rides the
  // static scan and does not extend the deviation interval; the moment a
  // transport deviates, its scheduled times are covered (the static day is
  // gone, so a query touching them sees a different world) and its real-time
  // events are covered as they are written by update_time().
  if (tt_ != nullptr && t.is_valid()) {
    extend_deviation_interval(static_transport_interval(*tt_, t));
  }
}

void rt_timetable::update_time(rt_transport_idx_t const rt_t,
                               stop_idx_t const stop_idx,
                               event_type const ev_type,
                               unixtime_t const new_time) {
  auto const ev_idx = stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0);
  assert(ev_idx >= 0 && static_cast<stop_idx_t>(ev_idx) <
                            rt_transport_stop_times_[rt_t].size());
  auto const v = unix_to_delta(new_time);
  rt_transport_stop_times_[rt_t][static_cast<std::size_t>(ev_idx)] = v;

  if (is_unchanged(rt_t)) {
    // O(1): only the event being written has to be compared, and only while
    // the transport still counts as identical to its schedule.
    auto const t = resolve_static(rt_t);
    if (tt_ == nullptr || !t.is_valid()) {
      mark_deviating(rt_t);  // nothing to compare against: assume it deviates
    } else {
      auto const r = tt_->transport_route_[t.t_idx_];
      auto const scheduled =
          (static_cast<int>(to_idx(t.day_)) -
           static_cast<int>(to_idx(base_day_idx_))) *
              1440 +
          tt_->event_mam(r, t.t_idx_, stop_idx, ev_type).count();
      if (static_cast<int>(v) != scheduled) {
        mark_deviating(rt_t);
      }
    }
  }

  if (!is_unchanged(rt_t)) {
    // On the real-time scan: this event is real-time data a query can see.
    // An event that left the transport on the static scan is its scheduled
    // time and changes nothing.
    extend_deviation_interval(delta_to_unix(base_day_, v));
  }
}

void rt_timetable::cancel_stop(rt_transport_idx_t const rt_t,
                               stop_idx_t const stop_idx) {
  auto& stp = rt_transport_location_seq_[rt_t][stop_idx];
  stp = stop{stop{stp}.location_idx(), false, false, false, false}.value();
  mark_deviating(rt_t);  // stop sequence no longer matches the static route
}

bool rt_timetable::matches_schedule(timetable const& tt,
                                    rt_transport_idx_t const rt_t) const {
  if (rt_transport_is_cancelled_.test(to_idx(rt_t))) {
    return false;
  }

  // additional trips have no static counterpart to fall back to
  auto const t = resolve_static(rt_t);
  if (!t.is_valid()) {
    return false;
  }

  auto const r = tt.transport_route_[t.t_idx_];
  auto const static_seq = tt.route_location_seq_[r];
  auto const rt_seq = rt_transport_location_seq_[rt_t];
  if (rt_seq.size() != static_seq.size() ||
      !std::equal(begin(rt_seq), end(rt_seq), begin(static_seq))) {
    return false;
  }

  // Everything the static scan reads has to agree, not just the times: it
  // filters on the route's flags and claszes, which add_rt_transport() copies
  // from the static route, so this also catches an update path that starts
  // changing them.
  auto const& rt_clasz = rt_transport_section_clasz_[rt_t];
  auto const& static_clasz = tt.route_section_clasz_[r];
  if (rt_clasz.size() != static_clasz.size() ||
      !std::equal(begin(rt_clasz), end(rt_clasz), begin(static_clasz))) {
    return false;
  }
  for (auto f = 0U; f != kNumRouteFlags; ++f) {
    if (rt_transport_flags_[f].test(to_idx(rt_t) * 2U) !=
            tt.route_flags_[f].test(r.v_ * 2U) ||
        rt_transport_flags_[f].test(to_idx(rt_t) * 2U + 1U) !=
            tt.route_flags_[f].test(r.v_ * 2U + 1U)) {
      return false;
    }
    auto const& rt_sections = rt_flags_per_section_[f][rt_t];
    auto const& static_sections = tt.route_flags_per_section_[f][r];
    if (rt_sections.size() != static_sections.size() ||
        !std::equal(begin(rt_sections), end(rt_sections),
                    begin(static_sections))) {
      return false;
    }
  }

  // rt event times are minutes from base_day_, static ones minutes from their
  // own day's midnight
  auto const n_stops = static_cast<stop_idx_t>(static_seq.size());
  auto const day_offset = (static_cast<int>(to_idx(t.day_)) -
                           static_cast<int>(to_idx(base_day_idx_))) *
                          1440;
  auto const same = [&](stop_idx_t const stop_idx, event_type const ev) {
    return static_cast<int>(event_time(rt_t, stop_idx, ev)) ==
           day_offset + tt.event_mam(r, t.t_idx_, stop_idx, ev).count();
  };
  for (auto stop_idx = stop_idx_t{0U}; stop_idx != n_stops; ++stop_idx) {
    if (stop_idx != 0U && !same(stop_idx, event_type::kArr)) {
      return false;
    }
    if (stop_idx != n_stops - 1U && !same(stop_idx, event_type::kDep)) {
      return false;
    }
  }

  return true;
}

void rt_timetable::finalize_rt_transport(timetable const& tt,
                                         rt_transport_idx_t const rt_t) {
  auto const unchanged = matches_schedule(tt, rt_t);
  set_unchanged(rt_t, unchanged);

  // An additional trip has no static counterpart and therefore no traffic day
  // to hand back, but it still has to be scanned.
  if (auto const t = resolve_static(rt_t); t.is_valid()) {
    // The day is on the static scan exactly while the transport is unchanged.
    // Guarded by the static traffic days so this can only ever *restore*
    // traffic the schedule already has, never invent it.
    set_rt_traffic_day(t,
                       unchanged && tt.is_transport_active(t.t_idx_, t.day_));
  }
  if (!unchanged) {
    register_scan(rt_t);
    if (auto const t = resolve_static(rt_t); t.is_valid()) {
      extend_deviation_interval(static_transport_interval(tt, t));
    }
  }
}

rt_transport_idx_t rt_timetable::add_rt_transport(
    source_idx_t const src,
    timetable const& tt,
    transport const t,
    std::span<stop::value_type> stop_seq,
    std::span<delta_t> time_seq,
    std::string_view new_trip_id,
    std::string_view route_id,
    direction_id_t const direction_id,
    std::string_view trip_short_name,
    delta_t const offset) {
  auto const [t_idx, day] = t;

  auto const rt_t_idx = rt_transport_src_.size();
  auto const rt_t = rt_transport_idx_t{rt_t_idx};
  if (new_trip_id.empty() && t.is_valid()) {
    static_trip_lookup_.emplace(t, rt_t_idx);
    rt_transport_static_transport_.emplace_back(t);

    // No traffic day is cleared here: the times below are copied verbatim from
    // the schedule, so until something changes them the static scan is the
    // correct -- and cheaper -- place for this transport. update_time(),
    // cancel_stop() and cancel_run() take it off there the moment it deviates.
    set_unchanged(rt_t, true);  // revisited below once the times are in
  } else {
    auto const rt_add_idx =
        rt_add_trip_id_idx_t{additional_trips_.at(src).transports_.size()};
    additional_trips_.at(src).ids_.store(new_trip_id);
    additional_trips_.at(src).transports_.emplace_back(rt_t_idx);
    rt_transport_static_transport_.emplace_back(rt_add_idx);
    set_unchanged(rt_t, false);  // no schedule to fall back on
  }

  auto const r =
      t.is_valid() ? tt.transport_route_[t_idx] : route_idx_t::invalid();
  auto const given_r = tt.route_ids_[src].ids_.find(route_id).value_or(
      route_id_idx_t::invalid());
  auto const location_seq = stop_seq.empty() && r != route_idx_t::invalid()
                                ? std::span{tt.route_location_seq_[r]}
                                : stop_seq;
  rt_transport_location_seq_.emplace_back(location_seq);
  rt_transport_src_.emplace_back(src);
  rt_transport_route_id_.emplace_back(given_r);
  alerts_.rt_transport_.emplace_back_empty();

  for (auto const s : location_seq) {
    auto rt_transports = location_rt_transports_[stop{s}.location_idx()];
    if (rt_transports.empty() || rt_transports.back() != rt_t) {
      rt_transports.push_back(rt_t);
    }
  }

  if (time_seq.empty() && r != route_idx_t::invalid()) {
    auto times =
        rt_transport_stop_times_.add_back_sized(location_seq.size() * 2U - 2U);
    auto i = 0U;
    auto const static_location_seq_len = tt.route_location_seq_[r].size();
    auto stop_idx = stop_idx_t{0U};
    for (auto const [a, b] : utl::pairwise(location_seq)) {
      CISTA_UNUSED_PARAM(a)
      CISTA_UNUSED_PARAM(b)
      times[i++] =
          unix_to_delta(tt.event_time(t, stop_idx, event_type::kDep)) + offset;
      times[i++] =
          unix_to_delta(tt.event_time(t, ++stop_idx, event_type::kArr)) +
          offset;
      if (stop_idx + 1U >= static_location_seq_len) {
        break;
      }
    }
  } else {
    // NOTE: `time_seq` may be a zero-initialized placeholder that is filled
    // in afterwards (added/replacement trips) - the deviation interval is
    // extended by the `update_time()` calls that write the times, not here.
    rt_transport_stop_times_.emplace_back(time_seq);
  }

  if (new_trip_id.empty() && t.is_valid() &&
      !(stop_seq.empty() && time_seq.empty() && offset == 0)) {
    // A replacement (own stop sequence / times) or a copy with shifted times
    // is not identical to its schedule from the start: off the static scan,
    // onto the real-time scan, and the deviation interval follows (see
    // mark_deviating()). A verbatim copy stays where it is until an update
    // moves one of its events.
    mark_deviating(rt_t);
    if (time_seq.empty()) {
      // shifted copy: the times are final and never pass through
      // update_time(), so cover them here
      extend_deviation_interval(rt_transport_interval(rt_t));
    }
  }

  rt_transport_track_sequence_.add_back_sized(0U);

  auto const flags_defaults = std::array{
      // TODO
      false,  // kBikesAllowed
      false,  // kCarsAllowed
      false,  // kWheelchairAccessible
      true,  // kReservationNotRequired
  };

  rt_transport_line_.add_back_sized(0U);
  rt_transport_is_cancelled_.resize(rt_transport_is_cancelled_.size() + 1U);
  for (auto i = 0U; i < kNumRouteFlags; ++i) {
    rt_transport_flags_[i].resize(rt_transport_flags_[i].size() + 2U);
  }
  rt_transport_section_directions_.add_back_sized(0U);  // TODO outside
  rt_transport_trip_short_names_.emplace_back(trip_short_name);

  rt_transport_direction_id_.resize(rt_transport_direction_id_.size() + 1U);
  rt_transport_direction_id_.set(rt_t, direction_id != direction_id_t{});

  if (r != route_idx_t::invalid()) {
    rt_transport_section_clasz_.emplace_back(tt.route_section_clasz_[r]);
    for (auto i = 0U; i < kNumRouteFlags; ++i) {
      rt_transport_flags_[i].set(rt_t_idx * 2, tt.route_flags_[i][r.v_ * 2]);
      rt_transport_flags_[i].set(rt_t_idx * 2 + 1,
                                 tt.route_flags_[i][r.v_ * 2 + 1]);
    }
  } else if (given_r != route_id_idx_t::invalid()) {
    rt_transport_section_clasz_.emplace_back(
        std::vector<clasz>{loader::gtfs::to_clasz(
            tt.route_ids_[src].route_id_type_.at(given_r).v_)});  // TODO

    for (auto i = 0U; i < kNumRouteFlags; ++i) {
      rt_transport_flags_[i].set(rt_t_idx * 2, flags_defaults[i]);
      rt_transport_flags_[i].set(rt_t_idx * 2 + 1, false);
    }
  } else {
    rt_transport_section_clasz_.emplace_back(std::vector<clasz>{clasz::kOther});

    for (auto i = 0U; i < kNumRouteFlags; ++i) {
      rt_transport_flags_[i].set(rt_t_idx * 2, flags_defaults[i]);
      rt_transport_flags_[i].set(rt_t_idx * 2 + 1, false);
    }
  }
  if (r != route_idx_t::invalid() && stop_seq.empty()) {
    for (auto i = 0U; i < kNumRouteFlags; ++i) {
      rt_flags_per_section_[i].emplace_back(tt.route_flags_per_section_[i][r]);
    }
  } else {
    for (auto i = 0U; i < kNumRouteFlags; ++i) {
      rt_flags_per_section_[i].emplace_back(
          std::vector<bool>{flags_defaults[i]});
      rt_transport_flags_[i].set(rt_t_idx * 2 + 1, false);
    }
  }

  assert(time_seq.empty() || time_seq.size() == location_seq.size() * 2U - 2U);
  assert(static_trip_lookup_.contains(t) ||
         additional_trips_.at(src).ids_.find(new_trip_id).has_value());
  assert(rt_transport_static_transport_[rt_transport_idx_t{rt_t_idx}] == t ||
         rt_transport_static_transport_[rt_transport_idx_t{rt_t_idx}] ==
             rt_add_trip_id_idx_t{additional_trips_.at(src).transports_.size() -
                                  1U});
  assert(additional_trips_.at(src).transports_.size() ==
         additional_trips_.at(src).ids_.strings_.size());
  assert(rt_transport_static_transport_.size() == rt_t_idx + 1U);
  assert(rt_transport_src_.size() == rt_t_idx + 1U);
  assert(rt_transport_route_id_.size() == rt_t_idx + 1U);
  assert(rt_transport_stop_times_.size() == rt_t_idx + 1U);
  assert(rt_transport_location_seq_.size() == rt_t_idx + 1U);
  assert(rt_transport_trip_short_names_.size() == rt_t_idx + 1U);
  assert(rt_transport_section_clasz_.size() == rt_t_idx + 1U);
  assert(rt_transport_line_.size() == rt_t_idx + 1U);
  assert(rt_flags_per_section_[kBikesAllowed].size() == rt_t_idx + 1U);
  assert(rt_flags_per_section_[kCarsAllowed].size() == rt_t_idx + 1U);
  assert(rt_flags_per_section_[kWheelchairAccessible].size() == rt_t_idx + 1U);
  assert(rt_flags_per_section_[kReservationNotRequired].size() ==
         rt_t_idx + 1U);

  return rt_transport_idx_t{rt_t_idx};
}

void rt_timetable::set_track(rt_transport_idx_t const rt_t,
                             stop_idx_t const stop_idx,
                             event_type const ev_type,
                             std::string_view const track) {
  if (track.empty()) {
    return;
  }
  utl::verify(stop_idx < rt_transport_location_seq_[rt_t].size(),
              "set_track: invalid stop_idx {} (n_stops={})", stop_idx,
              rt_transport_location_seq_[rt_t].size());
  auto bucket = rt_transport_track_sequence_[rt_t];
  if (bucket.empty()) {
    bucket.grow(rt_transport_stop_times_[rt_t].size(), track_idx_t::invalid());
  }
  auto const ev_idx = stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0);
  bucket[static_cast<std::size_t>(ev_idx)] = track_strings_.store(track);
}

std::optional<std::string_view> rt_timetable::get_track(
    rt_transport_idx_t const rt_t,
    stop_idx_t const stop_idx,
    event_type const ev_type) const {
  if (to_idx(rt_t) >= rt_transport_track_sequence_.size()) {
    return std::nullopt;
  }
  utl::verify(stop_idx < rt_transport_location_seq_[rt_t].size(),
              "get_track: invalid stop_idx {} (n_stops={})", stop_idx,
              rt_transport_location_seq_[rt_t].size());
  auto const bucket = rt_transport_track_sequence_[rt_t];
  if (bucket.empty()) {
    return std::nullopt;
  }
  auto const ev_idx = stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0);
  if (ev_idx < 0 || static_cast<std::size_t>(ev_idx) >= bucket.size()) {
    return std::nullopt;
  }
  return track_strings_.try_get(bucket[static_cast<std::size_t>(ev_idx)]);
}

interval<unixtime_t> rt_timetable::static_transport_interval(
    timetable const& tt, transport const t) const {
  if (!t.is_valid()) {
    return {};
  }
  auto const n_stops =
      tt.route_location_seq_[tt.transport_route_[t.t_idx_]].size();
  assert(n_stops >= 2U);
  return {tt.event_time(t, stop_idx_t{0U}, event_type::kDep),
          tt.event_time(t, static_cast<stop_idx_t>(n_stops - 1U),
                        event_type::kArr) +
              unixtime_t::duration{1}};
}

std::string_view rt_timetable::transport_name(
    timetable const& tt, rt_transport_idx_t const t) const {
  return std::visit(utl::overloaded{[&](translation_idx_t const idx) {
                                      return tt.get_default_translation(idx);
                                    },
                                    [](std::string_view const s) { return s; }},
                    trip_short_name(tt, t));
}

void rt_timetable::update_lbs(
    timetable const& tt,
    rt_transport_idx_t const rt_t,
    stop_idx_t const stop_idx,
    vector_map<location_idx_t, std::vector<footpath>>& tmp_fwd,
    vector_map<location_idx_t, std::vector<footpath>>& tmp_bwd) {
  auto const from_stop_idx = stop_idx;
  auto const to_stop_idx = static_cast<stop_idx_t>(stop_idx + 1U);

  auto const travel_time =
      duration_t{event_time(rt_t, to_stop_idx, event_type::kArr) -
                 event_time(rt_t, from_stop_idx, event_type::kDep)};

  auto const loc_seq = rt_transport_location_seq_[rt_t];
  if (travel_time < duration_t{0}) {
    log(log_lvl::error, "nigiri.rt.update_time",
        "travel_time < 0: {} -> {}: dep={} - arr={}",
        loc{tt, stop{loc_seq[from_stop_idx]}.location_idx()},
        loc{tt, stop{loc_seq[to_stop_idx]}.location_idx()},
        event_time(rt_t, from_stop_idx, event_type::kDep),
        event_time(rt_t, to_stop_idx, event_type::kArr));
    return;
  }

  auto const from =
      tt.locations_.get_root_idx(stop{loc_seq[from_stop_idx]}.location_idx());
  auto const to =
      tt.locations_.get_root_idx(stop{loc_seq[to_stop_idx]}.location_idx());

  if (from == to) {
    return;  // e.g. from one child to another within the same parent
  }

  auto const is_fastest = [](auto&& existing, footpath const& fp) {
    auto const it = utl::find_if(
        existing, [&](footpath const& x) { return x.target() == fp.target(); });
    return std::pair{it, it == end(existing) || it->duration() > fp.duration()};
  };

  auto const update =
      [&](vecvec<location_idx_t, footpath> const& tt_lbs,
          bitvec_map<location_idx_t>& rtt_has_lbs,
          vector_map<location_idx_t, std::vector<footpath>>& rtt_lbs,
          direction const dir) {
        auto const fwd = dir == direction::kForward;
        auto const src = fwd ? to : from;  // lbs are backwards!
        auto const tgt = fwd ? from : to;
        auto const new_fp = footpath{tgt, travel_time};

        if (!is_fastest(tt_lbs[src], new_fp).second) {
          return;  // Static timetable has faster/eq. Nothing to do.
        }

        if (!rtt_has_lbs[src]) {
          // There are no values stored in the real-time timetable. Push first.
          rtt_lbs[src].push_back(new_fp);
          rtt_has_lbs.set(src, true);
          return;
        }

        // There are already values stored in the real-time timetable.
        auto& lbs = rtt_lbs[src];
        auto const [it, is_fastest_rt] = is_fastest(lbs, new_fp);
        if (!is_fastest_rt) {
          // The value stored in the rt_timetable is faster/eq. Nothing to do.
          return;
        }

        if (it != end(lbs)) {
          // The same target did exist already. Update existing.
          it->duration_ = static_cast<location_idx_t::value_t>(
              std::min(footpath::kMaxDuration, travel_time).count());
        }

        // The same target did not exist yet. Push new.
        rtt_lbs[src].push_back(new_fp);
      };

  update(tt.fwd_search_lb_graph_[kDefaultProfile],
         fwd_search_lb_graph_has_edges_, tmp_fwd, direction::kForward);
  update(tt.bwd_search_lb_graph_[kDefaultProfile],
         bwd_search_lb_graph_has_edges_, tmp_bwd, direction::kBackward);
}

void rt_timetable::update_lbs(timetable const& tt) {
  auto timer = utl::scoped_timer{"update_lbs"};

  // --- UPDATE LBS ---
  auto tmp_fwd_lbs = vector_map<location_idx_t, std::vector<footpath>>{};
  auto tmp_bwd_lbs = vector_map<location_idx_t, std::vector<footpath>>{};
  tmp_fwd_lbs.resize(tt.n_locations());
  tmp_bwd_lbs.resize(tt.n_locations());
  for (auto rt_t = rt_transport_idx_t{0U}; rt_t != n_rt_transports(); ++rt_t) {
    auto const n_events = rt_transport_stop_times_[rt_t].size();
    auto const n_segments = static_cast<stop_idx_t>(n_events / 2U);
    for (auto i = stop_idx_t{0U}; i != n_segments; ++i) {
      update_lbs(tt, rt_t, i, tmp_fwd_lbs, tmp_bwd_lbs);
    }
  }

  // --- COPY TO RT_TIMETABLE ---
  auto const copy =
      [&](vecvec<location_idx_t, footpath>& to,
          vector_map<location_idx_t, std::vector<footpath>> const& from) {
        to.clear();
        for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
          to.emplace_back(from[l]);
        }
      };
  copy(fwd_search_lb_graph_, tmp_fwd_lbs);
  copy(bwd_search_lb_graph_, tmp_bwd_lbs);
}

void rt_timetable::cancel_run(rt::run const& r) {
  if (r.is_rt()) {
    rt_transport_is_cancelled_.set(to_idx(r.rt_), true);
    set_unchanged(r.rt_, false);
    register_scan(r.rt_);  // still scanned, just never identical to schedule
    extend_deviation_interval(rt_transport_interval(r.rt_));
  }
  if (r.is_scheduled()) {
    extend_deviation_interval(static_transport_interval(r.t_));
    set_rt_traffic_day(r.t_, false);

    for (auto i = r.stop_range_.from_; i != r.stop_range_.to_; ++i) {
      dispatch_stop_change(r, i, event_type::kArr, std::nullopt, false);
      dispatch_stop_change(r, i, event_type::kDep, std::nullopt, false);
    }
  }
}

}  // namespace nigiri
