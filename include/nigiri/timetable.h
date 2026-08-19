#pragma once

#include <compare>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>

#include "cista/cuda_check.h"
#include "cista/memory_holder.h"

#include "geo/box.h"
#include "geo/latlng.h"

#include "nigiri/common/interval.h"
#include "nigiri/fares.h"
#include "nigiri/footpath.h"
#include "nigiri/stop.h"
#include "nigiri/string_store.h"
#include "nigiri/td_footpath.h"
#include "nigiri/types.h"

namespace nigiri {

struct day_list;

struct location_id_hash {
  using is_transparent = void;

  cista::hash_t operator()(owning_location_id const& x) const {
    auto h = cista::BASE_HASH;
    h = cista::hash_combine(h, cista::hashing<source_idx_t>{}(x.src_));
    h = cista::hash_combine(h,
                            cista::hashing<std::string_view>{}(x.id_.view()));
    return h;
  }

  cista::hash_t operator()(location_id const& x) const {
    auto h = cista::BASE_HASH;
    h = cista::hash_combine(h, cista::hashing<source_idx_t>{}(x.src_));
    h = cista::hash_combine(h, cista::hashing<std::string_view>{}(x.id_));
    return h;
  }
};

struct location_id_equals {
  using is_transparent = void;

  cista::hash_t operator()(owning_location_id const& a,
                           owning_location_id const& b) const {
    return std::tie(a.src_, a.id_) == std::tie(b.src_, b.id_);
  }

  cista::hash_t operator()(location_id const& b,
                           owning_location_id const& a) const {
    return std::tie(a.src_, a.id_) == std::tie(b.src_, b.id_);
  }
};

// Transfer point recommended by the input data (transfers.txt type 0 =
// recommended / type 1 = timed), optionally scoped to a trip or route pair
// (invalid = unscoped).
struct preferred_transfer {
  location_idx_t to_{location_idx_t::invalid()};
  trip_idx_t from_trip_{trip_idx_t::invalid()};
  trip_idx_t to_trip_{trip_idx_t::invalid()};
  route_id_idx_t from_route_{route_id_idx_t::invalid()};
  route_id_idx_t to_route_{route_id_idx_t::invalid()};
};

struct timetable {
  struct locations {
    // Virtual locations (location_type::kVirt) have no attributes of their own
    // (name, platform code, stop code, description, timezone): they are taken
    // from the location they were generated for.
    location_idx_t get_attribute_idx(location_idx_t const l) const {
      return types_[l] == location_type::kVirt ? parents_[l] : l;
    }

    location_idx_t get_root_idx(location_idx_t const idx) const {
      auto l = idx;
      auto i = 0;
      for (auto p = parents_[l]; p != location_idx_t::invalid();
           p = parents_[l]) {
        if (p == idx || i > 20) {
          l = parents_[idx];
          break;
        }
        l = p;
        ++i;
      }
      return l;
    }

    // Same-stop minimum transfer time as seen by `prf`. Profiles other than
    // `kDefaultProfile` ignore the qualified transfers.txt rules and the
    // virtual locations these created (see raptor's `ProjectVirts`), but still
    // honor the plain minimum transfer time of the stop.
    u8_minutes min_transfer_time(profile_idx_t const prf,
                                 location_idx_t const l) const {
      return prf == kDefaultProfile
                 ? transfer_time_[l]
                 : base_transfer_time_[types_[l] == location_type::kVirt
                                           ? parents_[l]
                                           : l];
    }

    // Extends `base_transfer_time_` to cover every location added since the
    // last call, capturing its transfer time before transfers.txt same-stop
    // rules are applied on top of `transfer_time_`.
    void sync_base_transfer_time() {
      for (auto l = location_idx_t{base_transfer_time_.size()};
           l != location_idx_t{transfer_time_.size()}; ++l) {
        base_transfer_time_.push_back(transfer_time_[l]);
      }
    }

    hash_map<owning_location_id,
             location_idx_t,
             location_id_hash,
             location_id_equals>
        location_id_to_idx_;
    vector_map<location_idx_t, translation_idx_t> names_;
    vector_map<location_idx_t, translation_idx_t> platform_codes_;
    vector_map<location_idx_t, translation_idx_t> stop_codes_;
    vector_map<location_idx_t, translation_idx_t> descriptions_;
    vecvec<location_idx_t, char> ids_;
    vector_map<location_idx_t, geo::latlng> coordinates_;
    vector_map<location_idx_t, source_idx_t> src_;
    // Minimum transfer time at a location. `transfer_time_` has the same-stop
    // rules from transfers.txt applied, `base_transfer_time_` is the value
    // before any such rule. Profiles that ignore transfers.txt (everything
    // except `kDefaultProfile`, see raptor's `ProjectVirts`) read the base.
    vector_map<location_idx_t, u8_minutes> transfer_time_;
    vector_map<location_idx_t, u8_minutes> base_transfer_time_;
    vector_map<location_idx_t, location_type> types_;
    vector_map<location_idx_t, location_idx_t> parents_;
    vector_map<location_idx_t, timezone_idx_t> location_timezones_;
    mutable_fws_multimap<location_idx_t, location_idx_t> equivalences_;
    mutable_fws_multimap<location_idx_t, location_idx_t> children_;
    mutable_fws_multimap<location_idx_t, footpath> preprocessing_footpaths_out_;
    array<vecvec<location_idx_t, footpath>, kNProfiles> footpaths_out_;
    array<vecvec<location_idx_t, footpath>, kNProfiles> footpaths_in_;
    vector_map<location_idx_t, std::uint32_t> location_importance_;
    std::uint32_t max_importance_{0U};
    rtree<location_idx_t> rtree_;
    bitvec_map<location_idx_t> ticketing_unavailable_;

    // Authoritative transfer edges from transfers.txt. These override any
    // (re)computed footpath between their endpoints (e.g. after street
    // routing) - a rule fixes the transfer time, which may be shorter or
    // longer than the walking time.
    mutable_fws_multimap<location_idx_t, footpath> transfer_rule_fps_;

    // Preferred transfer points (transfers.txt type 0 = recommended,
    // type 1 = timed). Soft preference used by optimize_footpaths to move
    // transfers here. Matching includes children, so a station pair covers
    // its platforms.
    mutable_fws_multimap<location_idx_t, preferred_transfer>
        preferred_transfers_;

    // Transfer hubs: aggregation nodes with an in and an out member list that
    // derive the default-valued transfer cells of a stop with kVirt children
    // instead of materializing them. Every pair u -> h -> v derived by hub h
    // has the same duration hub_time_[h], so the weight is stored once per hub
    // and not per member. Relaxation happens within one footpath phase (raptor
    // expand_hubs: gather marked locations into hub minima, scatter hub minima
    // over the out members at hub_time_). The backward search swaps the two
    // lists, so a hub's pair set is derivable in both directions by
    // construction; reconstruct walks hub_out_by_loc_ x hub_in_. All
    // classification happens in build_hubs and is encoded purely in list
    // membership. Transfer rules do not depend on the street routing profile,
    // so one set of hubs serves all profiles.
    // Per profile: a hub belongs to exactly one. The default profile's hubs
    // come from transfers.txt; every other profile ignores the rules - a
    // wheelchair may not manage the stated time or the stairs at all, a car
    // transfer has nothing to do with them - and gets hubs built from its own
    // routed walks, plus one per stop to hold its virtual locations together,
    // which they still need since the transports are bound to them.
    array<vecvec<hub_idx_t, location_idx_t>, kNProfiles> hub_in_;
    array<vecvec<hub_idx_t, location_idx_t>, kNProfiles> hub_out_;
    array<vector_map<hub_idx_t, duration_t>, kNProfiles> hub_time_;
    array<vecvec<location_idx_t, hub_idx_t>, kNProfiles> hub_in_by_loc_;
    array<vecvec<location_idx_t, hub_idx_t>, kNProfiles> hub_out_by_loc_;
    // hubs [0, n_rule_hubs_) of the default profile are rule-derived and hold
    // in every profile; the rest are that profile's walks
    std::uint32_t n_rule_hubs_{0U};


  } locations_;

  struct transport {
    bitfield_idx_t bitfield_idx_;
    route_idx_t route_idx_;
    delta first_dep_offset_;
    basic_string<merged_trips_idx_t> const& external_trip_ids_;
    basic_string<attribute_combination_idx_t> const& section_attributes_;
    basic_string<provider_idx_t> const& section_providers_;
    basic_string<translation_idx_t> const& section_directions_;
  };

  timezone_idx_t register_timezone(timezone tz);

  std::string_view translate(lang_t const&, translation_idx_t) const;
  translated_str_t get(translation_idx_t) const;
  std::string_view get_default_translation(translation_idx_t) const;
  std::string_view get_default_name(location_idx_t) const;

  translation_idx_t register_translation(std::string const&);
  translation_idx_t register_translation(std::string_view);
  translation_idx_t register_translation(translated_str_t const&);

  std::optional<location_idx_t> find(location_id const& id) const;

  void resolve();

  bitfield_idx_t register_bitfield(bitfield const& b);
  route_idx_t register_route(
      basic_string<stop::value_type> const& stop_seq,
      basic_string<clasz> const& clasz_sections,
      std::array<bitvec, route_flag::kNumRouteFlags> const& flags_per_section);
  void finish_route();

  provider_idx_t get_provider_idx(std::string_view id, source_idx_t) const;

  merged_trips_idx_t register_merged_trip(basic_string<trip_idx_t> const&);

  source_file_idx_t register_source_file(std::string_view path);

  void add_transport(transport&& t);

  transport_idx_t next_transport_idx() const;

  bool is_transport_active(transport_idx_t const t, day_idx_t const day) const {
    return bitfields_[transport_traffic_days_[t]].test(to_idx(day));
  }

  bool is_route_active(route_idx_t const r, day_idx_t const day) const {
    return bitfields_[route_traffic_days_[r]].test(to_idx(day));
  }

  std::span<delta const> event_times_at_stop(route_idx_t const r,
                                             stop_idx_t const stop_idx,
                                             event_type const ev_type) const {
    auto const n_transports =
        static_cast<unsigned>(route_transport_ranges_[r].size());
    auto const idx = static_cast<unsigned>(
        route_stop_time_ranges_[r].from_ +
        n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
    return std::span<delta const>{&route_stop_times_[idx], n_transports};
  }

  delta event_mam(route_idx_t const r,
                  transport_idx_t t,
                  stop_idx_t const stop_idx,
                  event_type const ev_type) const {
    auto const range = route_transport_ranges_[r];
    auto const n_transports = static_cast<unsigned>(range.size());
    auto const route_stop_begin = static_cast<unsigned>(
        route_stop_time_ranges_[r].from_ +
        n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
    auto const t_idx_in_route = to_idx(t) - to_idx(range.from_);
    return route_stop_times_[route_stop_begin + t_idx_in_route];
  }

  delta event_mam(transport_idx_t t,
                  stop_idx_t const stop_idx,
                  event_type const ev_type) const {
    return event_mam(transport_route_[t], t, stop_idx, ev_type);
  }

  unixtime_t event_time(nigiri::transport t,
                        stop_idx_t const stop_idx,
                        event_type const ev_type) const {
    return unixtime_t{internal_interval_days().from_ + to_idx(t.day_) * 1_days +
                      event_mam(t.t_idx_, stop_idx, ev_type).as_duration()};
  }

  day_idx_t day_idx(date::year_month_day const day) const {
    return day_idx(date::sys_days{day});
  }

  day_idx_t day_idx(date::sys_days const day) const {
    return day_idx_t{(day - (date_range_.from_ - kTimetableOffset)).count()};
  }

  std::pair<day_idx_t, minutes_after_midnight_t> day_idx_mam(
      unixtime_t const t) const {
    auto const minutes_since_timetable_begin =
        (t - internal_interval().from_).count();
    auto const d =
        static_cast<day_idx_t::value_t>(minutes_since_timetable_begin / 1440);
    auto const m = minutes_since_timetable_begin % 1440;
    return {day_idx_t{d}, minutes_after_midnight_t{m}};
  }

  unixtime_t to_unixtime(day_idx_t const d,
                         minutes_after_midnight_t const m = 0_minutes) const {
    return internal_interval_days().from_ + to_idx(d) * 1_days + m;
  }

  cista::base_t<trip_idx_t> n_trips() const { return trip_short_names_.size(); }

  cista::base_t<location_idx_t> n_locations() const {
    return locations_.names_.size();
  }

  cista::base_t<route_idx_t> n_routes() const {
    return route_location_seq_.size();
  }

  cista::base_t<source_idx_t> n_sources() const { return n_sources_; }

  cista::base_t<provider_idx_t> n_agencies() const { return providers_.size(); }

  interval<unixtime_t> external_interval() const {
    return {std::chrono::time_point_cast<i32_minutes>(date_range_.from_),
            std::chrono::time_point_cast<i32_minutes>(date_range_.to_)};
  }

  interval<date::sys_days> internal_interval_days() const {
    return {date_range_.from_ - kTimetableOffset,
            date_range_.to_ + date::days{1}};
  }

  day_list days(bitfield const&) const;

  interval<unixtime_t> internal_interval() const {
    return {
        std::chrono::time_point_cast<i32_minutes>(date_range_.from_ -
                                                  kTimetableOffset),
        std::chrono::time_point_cast<i32_minutes>(date_range_.to_ + 1_days)};
  }

  std::string_view transport_name(transport_idx_t const t) const {
    auto const trip_idx =
        merged_trips_[transport_to_trip_section_[t].front()].front();
    return get_default_translation(trip_display_names_[trip_idx]);
  }

  debug dbg(transport_idx_t const t) const {
    auto const trip_idx =
        merged_trips_[transport_to_trip_section_[t].front()].front();
    return debug{
        source_file_names_[trip_debug_[trip_idx].front().source_file_idx_]
            .view(),
        trip_debug_[trip_idx].front().line_number_from_,
        trip_debug_[trip_idx].front().line_number_to_};
  }

  friend std::ostream& operator<<(std::ostream&, timetable const&);

  void write(std::filesystem::path const&) const;
  static cista::wrapped<timetable> read(std::filesystem::path const&);

  bool is_flag_set(route_flag const f, route_idx_t const r) const {
    return route_flags_[f][to_idx(r) * 2U] ||
           route_flags_[f][to_idx(r) * 2U + 1U];
  }

  // Schedule range.
  interval<date::sys_days> date_range_;

  // Timezones.
  vector_map<timezone_idx_t, timezone> timezones_;

  // Source -> feed end date
  vector_map<source_idx_t, date::sys_days> src_end_date_;

  // Trip access: external trip id -> internal trip index
  vector<pair<trip_id_idx_t, trip_idx_t>> trip_id_to_idx_;

  // Trip index -> list of external trip ids
  mutable_fws_multimap<trip_idx_t, trip_id_idx_t> trip_ids_;

  // Storage for trip id strings + source
  vecvec<trip_id_idx_t, char> trip_id_strings_;
  vector_map<trip_id_idx_t, source_idx_t> trip_id_src_;

  // Trip -> direction (valid options 0 or 1)
  bitvec_map<trip_idx_t> trip_direction_id_;

  // Trip train number, if available (otherwise 0)
  vector_map<trip_id_idx_t, std::uint32_t> trip_train_nr_;

  // Trip -> route name
  vector_map<trip_idx_t, route_id_idx_t> trip_route_id_;

  // Categories
  vector_map<category_idx_t, category> categories_;

  // External route id
  struct route_ids {
    vector_map<route_id_idx_t, category_idx_t> route_id_category_;
    vector_map<route_id_idx_t, translation_idx_t> route_id_short_names_;
    vector_map<route_id_idx_t, translation_idx_t> route_id_long_names_;
    vector_map<route_id_idx_t, translation_idx_t> route_id_url_;
    vector_map<route_id_idx_t, route_type_t> route_id_type_;
    vector_map<route_id_idx_t, provider_idx_t> route_id_provider_;
    vector_map<route_id_idx_t, route_color> route_id_colors_;
    vector_map<route_id_idx_t, ticketing_link_idx_t> route_id_ticketing_link_;
    paged_vecvec<route_id_idx_t, trip_idx_t> route_id_trips_;
    string_store<route_id_idx_t> ids_;
  };
  vector_map<source_idx_t, route_ids> route_ids_;

  // Trip index -> all transports with a stop interval
  paged_vecvec<trip_idx_t, transport_range_t> trip_transport_ranges_;

  // Transport -> stop sequence numbers (relevant for GTFS-RT stop matching)
  // Compaction:
  // - empty = zero-based sequence 0,1,2,...
  // - only one '1' entry = one-based sequence 1,2,3,...
  // - only one '10' entry = 10-based sequence 10,20,30,...
  // - more than one entry: exact sequence number for each stop
  vecvec<trip_idx_t, stop_idx_t> trip_stop_seq_numbers_;

  // Trip -> debug info
  mutable_fws_multimap<trip_idx_t, trip_debug> trip_debug_;
  vecvec<source_file_idx_t, char, std::uint32_t> source_file_names_;

  // Trip index -> trip name
  vector_map<trip_idx_t, translation_idx_t> trip_short_names_;

  // Trip index -> display name
  vector_map<trip_idx_t, translation_idx_t> trip_display_names_;

  // Route -> range of transports in this route (from/to transport_idx_t)
  vector_map<route_idx_t, interval<transport_idx_t>> route_transport_ranges_;

  // Route -> list of stops
  vecvec<route_idx_t, stop::value_type> route_location_seq_;

  // Route -> clasz
  vector_map<route_idx_t, clasz> route_clasz_;

  // Route -> clasz per section
  vecvec<route_idx_t, clasz> route_section_clasz_;

  // Route * 2 -> flag along the entire route
  // Route * 2 + 1 -> flag along parts of the route
  std::array<bitvec, route_flag::kNumRouteFlags> route_flags_;

  // Route -> flag per section
  // Only set for routes where the entry in route_flag_
  // is set to "flag along parts of the route"
  std::array<vecvec<route_idx_t, bool>, route_flag::kNumRouteFlags>
      route_flags_per_section_;

  // Location -> list of routes
  vecvec<location_idx_t, route_idx_t> location_routes_;

  // Route 1:
  //   stop-1-dep: [trip1, trip2, ..., tripN]
  //   stop-2-arr: [trip1, trip2, ..., tripN]
  //   ...
  // Route 2:
  //  stop-1-dep: [...]
  // ...
  // RouteN: ...
  vector_map<route_idx_t, interval<std::uint32_t>> route_stop_time_ranges_;
  vector<delta> route_stop_times_;

  // Offset between the stored time and the time given in the GTFS timetable
  // Required to match GTFS-RT with GTFS-static trips.
  vector_map<transport_idx_t, delta> transport_first_dep_offset_;

  // Services in GTFS can start with a first departure time > 24:00:00
  // The loader transforms this into a time <24:00:00 and shifts the bits in
  // the bitset accordingly. To still be able to match the traffic day from
  // the corresponding service_id, it's necessary to store the number of days
  // which is floor(stop_times.txt:departure_time/1440)
  vector_map<transport_idx_t, std::uint8_t> initial_day_offset_;

  // Trip index -> traffic day bitfield
  vector_map<transport_idx_t, bitfield_idx_t> transport_traffic_days_;

  // Route -> traffic day bitfield
  vector_map<route_idx_t, bitfield_idx_t> route_traffic_days_;

  // Unique bitfields
  vector_map<bitfield_idx_t, bitfield> bitfields_;

  // For each trip the corresponding route
  vector_map<transport_idx_t, route_idx_t> transport_route_;

  // Trip index -> merged trips
  vecvec<transport_idx_t, merged_trips_idx_t> transport_to_trip_section_;

  // Merged trips info
  vecvec<merged_trips_idx_t, trip_idx_t> merged_trips_;

  // Section meta infos:
  vector_map<attribute_idx_t, attribute> attributes_;
  vecvec<attribute_combination_idx_t, attribute_idx_t> attribute_combinations_;
  vector_map<provider_idx_t, provider> providers_;
  vector<provider_idx_t> provider_id_to_idx_;

  // Transport to section meta infos; Compaction:
  // - only one value = value is valid for the whole run
  // - multiple values = one value for each section
  vecvec<transport_idx_t, attribute_combination_idx_t>
      transport_section_attributes_;
  vecvec<transport_idx_t, provider_idx_t> transport_section_providers_;
  vecvec<transport_idx_t, translation_idx_t> transport_section_directions_;

  // Lower bound graph.
  std::array<vecvec<location_idx_t, footpath>, kNProfiles> fwd_search_lb_graph_;
  std::array<vecvec<location_idx_t, footpath>, kNProfiles> bwd_search_lb_graph_;

  // profile name -> profile_idx_t
  hash_map<string, profile_idx_t> profiles_;

  // Fares
  vector_map<source_idx_t, fares> fares_;
  vector_map<area_idx_t, area> areas_;
  vecvec<location_idx_t, area_idx_t> location_areas_;

  // Flex
  paged_vecvec<location_group_idx_t, location_idx_t> location_group_locations_;
  paged_vecvec<location_idx_t, location_group_idx_t> location_location_groups_;
  vector_map<location_group_idx_t, translation_idx_t> location_group_name_;
  vector_map<location_group_idx_t, string_idx_t> location_group_id_;
  vector_map<flex_area_idx_t, geo::box> flex_area_bbox_;
  vector_map<flex_area_idx_t, string_idx_t> flex_area_id_;
  vector_map<flex_area_idx_t, source_idx_t> flex_area_src_;
  vecvec<flex_area_idx_t, location_idx_t> flex_area_locations_;
  nvec<flex_area_idx_t, geo::latlng, 2U> flex_area_outers_;
  nvec<flex_area_idx_t, geo::latlng, 3U> flex_area_inners_;
  vector_map<flex_area_idx_t, translation_idx_t> flex_area_name_;
  vector_map<flex_area_idx_t, translation_idx_t> flex_area_desc_;
  rtree<flex_area_idx_t> flex_area_rtree_;
  paged_vecvec<location_group_idx_t, flex_transport_idx_t>
      location_group_transports_;
  paged_vecvec<flex_area_idx_t, flex_transport_idx_t> flex_area_transports_;
  vector_map<flex_transport_idx_t, bitfield_idx_t> flex_transport_traffic_days_;
  vector_map<flex_transport_idx_t, trip_idx_t> flex_transport_trip_;
  vecvec<flex_transport_idx_t, interval<duration_t>>
      flex_transport_stop_time_windows_;
  vector_map<flex_transport_idx_t, flex_stop_seq_idx_t>
      flex_transport_stop_seq_;
  vecvec<flex_stop_seq_idx_t, flex_stop_t> flex_stop_seq_;
  vecvec<flex_transport_idx_t, booking_rule_idx_t>
      flex_transport_pickup_booking_rule_;
  vecvec<flex_transport_idx_t, booking_rule_idx_t>
      flex_transport_drop_off_booking_rule_;
  vector_map<booking_rule_idx_t, booking_rule> booking_rules_;

  // Strings
  string_store<string_idx_t> strings_;

  // Translated strings
  nvec<translation_idx_t, char, 2U> translations_;
  vecvec<translation_idx_t, language_idx_t> translation_language_;
  string_store<language_idx_t> languages_;

  cista::base_t<source_idx_t> n_sources_{};

  // Ticketing
  vecvec<location_idx_t, pair<provider_idx_t, string_idx_t>>
      location_ticketing_identifier_;
  vecvec<trip_idx_t, string_idx_t> trip_ticketing_identifier_;
  bitvec_map<trip_idx_t> trip_ticketing_unavailable_;

  struct ticketing_links {
    vecvec<ticketing_link_idx_t, char> web_;
    vecvec<ticketing_link_idx_t, char> andoid_;
    vecvec<ticketing_link_idx_t, char> ios_;
  };

  ticketing_links ticketing_links_;
};

struct loc {
  timetable const& tt_;
  location_idx_t l_;
};

inline auto format_as(loc const& l)
    -> std::pair<std::string_view, std::string_view> {
  if (l.l_ == location_idx_t::invalid()) {
    return {};
  }
  return {l.tt_.get_default_name(l.l_), l.tt_.locations_.ids_[l.l_].view()};
}

inline std::ostream& operator<<(std::ostream& out, loc const& l) {
  auto const [id, name] = format_as(l);
  return out << '(' << id << ", " << name << ')';
}

}  // namespace nigiri
