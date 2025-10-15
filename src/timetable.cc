#include <cassert>
#include <ranges>

#include "nigiri/timetable.h"

#include "cista/io.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/overloaded.h"

#include "nigiri/common/day_list.h"
#include "nigiri/rt/frun.h"

namespace vw = std::views;

namespace nigiri {

std::string reverse(std::string s) {
  std::reverse(s.begin(), s.end());
  return s;
}

void timetable::resolve() {
  for (auto& tz : locations_.timezones_) {
    if (holds_alternative<pair<string, void const*>>(tz)) {
      auto& [name, ptr] = tz.as<pair<string, void const*>>();
      ptr = date::locate_zone(name);
    }
  }
}

std::ostream& operator<<(std::ostream& out, timetable const& tt) {
  for (auto const [id, idx] : tt.trip_id_to_idx_) {
    auto const str = tt.trip_id_strings_[id].view();
    out << str << ":\n";
    for (auto const& t : tt.trip_transport_ranges_.at(idx)) {
      out << "  " << t.first << ": " << t.second << " active="
          << day_list{tt.bitfields_[tt.transport_traffic_days_[t.first]],
                      tt.internal_interval_days().from_}
          << "\n";
    }
  }

  auto const internal = tt.internal_interval_days();
  auto const num_days =
      static_cast<size_t>((internal.to_ - internal.from_ + 1_days) / 1_days);
  for (auto i = 0U; i != tt.transport_traffic_days_.size(); ++i) {
    auto const transport_idx = transport_idx_t{i};
    auto const num_stops =
        tt.route_location_seq_[tt.transport_route_[transport_idx]].size();
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    out << "TRANSPORT=" << transport_idx << ", TRAFFIC_DAYS="
        << reverse(traffic_days.to_string().substr(kMaxDays - num_days))
        << "\n";
    for (auto d = internal.from_; d != internal.to_;
         d += std::chrono::days{1}) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t::value_t>((d - internal.from_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(out, "%F", d);
        out << " (day_idx=" << day_idx << ")\n";
        out << rt::frun{
            tt,
            nullptr,
            {.t_ = transport{transport_idx, day_idx},
             .stop_range_ = {0U, static_cast<stop_idx_t>(num_stops)}}};
        out << "\n";
      }
    }
    out << "---\n\n";
  }
  return out;
}

day_list timetable::days(bitfield const& bf) const {
  return day_list{bf, internal_interval_days().from_};
}

cista::wrapped<timetable> timetable::read(std::filesystem::path const& p) {
  return cista::read<timetable>(p);
}

void timetable::write(std::filesystem::path const& p) const {
  return cista::write(p, *this);
}

struct index_mapping {
  alt_name_idx_t const alt_name_idx_offset_;
  area_idx_t const area_idx_offset_;
  attribute_idx_t const attribute_idx_offset_;
  vector_map<bitfield_idx_t, bitfield_idx_t> const bitfield_idx_map_;
  booking_rule_idx_t const booking_rule_idx_offset_;
  flex_stop_seq_idx_t const flex_stop_seq_idx_offset_;
  flex_transport_idx_t const flex_transport_idx_offset_;
  language_idx_t const language_idx_offset_;
  location_group_idx_t const location_group_idx_offset_;
  location_idx_t const location_idx_offset_;
  merged_trips_idx_t const merged_trips_idx_offset_;
  provider_idx_t const provider_idx_offset_;
  route_idx_t const route_idx_offset_;
  source_file_idx_t const source_file_idx_offset_;
  source_idx_t const source_idx_offset_;
  vector_map<string_idx_t, string_idx_t> const string_idx_map_;
  timezone_idx_t const timezone_idx_offset_;
  transport_idx_t const transport_idx_offset_;
  trip_direction_string_idx_t const trip_direction_string_idx_offset_;
  trip_id_idx_t const trip_id_idx_offset_;
  trip_idx_t const trip_idx_offset_;
  trip_line_idx_t const trip_line_idx_offset_;

  index_mapping(
      timetable const& first_tt,
      vector_map<bitfield_idx_t, bitfield_idx_t> const& bitfield_idx_map,
      vector_map<string_idx_t, string_idx_t> const& string_idx_map)
      : alt_name_idx_offset_{first_tt.locations_.alt_name_strings_.size()},
        area_idx_offset_{first_tt.areas_.size()},
        attribute_idx_offset_{first_tt.attributes_.size()},
        bitfield_idx_map_{bitfield_idx_map},
        booking_rule_idx_offset_{first_tt.booking_rules_.size()},
        flex_stop_seq_idx_offset_{first_tt.flex_stop_seq_.size()},
        flex_transport_idx_offset_{
            first_tt.flex_transport_traffic_days_.size()},
        language_idx_offset_{first_tt.languages_.size()},
        location_group_idx_offset_{first_tt.location_group_name_.size()},
        location_idx_offset_{first_tt.n_locations()},
        merged_trips_idx_offset_{first_tt.merged_trips_.size()},
        provider_idx_offset_{first_tt.providers_.size()},
        route_idx_offset_{first_tt.n_routes()},
        source_file_idx_offset_{first_tt.source_file_names_.size()},
        source_idx_offset_{first_tt.src_end_date_.size()},
        string_idx_map_{string_idx_map},
        timezone_idx_offset_{first_tt.locations_.timezones_.size()},
        transport_idx_offset_{first_tt.transport_traffic_days_.size()},
        trip_direction_string_idx_offset_{
            first_tt.trip_direction_strings_.size()},
        trip_id_idx_offset_{first_tt.trip_id_strings_.size()},
        trip_idx_offset_{first_tt.trip_ids_.size()},
        trip_line_idx_offset_{first_tt.trip_lines_.size()} {}

  auto map(alt_name_idx_t const& i) const {
    return i != alt_name_idx_t::invalid() ? i + alt_name_idx_offset_
                                          : alt_name_idx_t::invalid();
  }
  auto map(area_idx_t const& i) const {
    return i != area_idx_t::invalid() ? i + area_idx_offset_
                                      : area_idx_t::invalid();
  }
  auto map(attribute_idx_t const& i) const {
    return i != attribute_idx_t::invalid() ? i + attribute_idx_offset_
                                           : attribute_idx_t::invalid();
  }
  auto map(bitfield_idx_t const& i) const {
    return i != bitfield_idx_t::invalid() ? bitfield_idx_map_[i]
                                          : bitfield_idx_t::invalid();
  }
  auto map(booking_rule_idx_t const& i) const {
    return i != booking_rule_idx_t::invalid() ? i + booking_rule_idx_offset_
                                              : booking_rule_idx_t::invalid();
  }
  auto map(flex_stop_seq_idx_t const& i) const {
    return i != flex_stop_seq_idx_t::invalid() ? i + flex_stop_seq_idx_offset_
                                               : flex_stop_seq_idx_t::invalid();
  }
  auto map(flex_transport_idx_t const& i) const {
    return i != flex_transport_idx_t::invalid()
               ? i + flex_transport_idx_offset_
               : flex_transport_idx_t::invalid();
  }
  auto map(language_idx_t const& i) const {
    return i != language_idx_t::invalid() ? i + language_idx_offset_
                                          : language_idx_t::invalid();
  }
  auto map(location_group_idx_t const& i) const {
    return i != location_group_idx_t::invalid()
               ? i + location_group_idx_offset_
               : location_group_idx_t::invalid();
  }
  auto map(location_idx_t const& i) const {
    return i != location_idx_t::invalid() ? i + location_idx_offset_
                                          : location_idx_t::invalid();
  }
  auto map(merged_trips_idx_t const& i) const {
    return i != merged_trips_idx_t::invalid() ? i + merged_trips_idx_offset_
                                              : merged_trips_idx_t::invalid();
  }
  auto map(provider_idx_t const& i) const {
    return i != provider_idx_t::invalid() ? i + provider_idx_offset_
                                          : provider_idx_t::invalid();
  }
  auto map(route_idx_t const& i) const {
    return i != route_idx_t::invalid() ? i + route_idx_offset_
                                       : route_idx_t::invalid();
  }
  auto map(source_file_idx_t const& i) const {
    return i != source_file_idx_t::invalid() ? i + source_file_idx_offset_
                                             : source_file_idx_t::invalid();
  }
  auto map(source_idx_t const& i) const {
    return i != source_idx_t::invalid() ? i + source_idx_offset_
                                        : source_idx_t::invalid();
  }
  auto map(string_idx_t const& i) const {
    return i != string_idx_t::invalid() ? string_idx_map_[i]
                                        : string_idx_t::invalid();
  }
  auto map(timezone_idx_t const& i) const {
    return i != timezone_idx_t::invalid() ? i + timezone_idx_offset_
                                          : timezone_idx_t::invalid();
  }
  auto map(transport_idx_t const& i) const {
    return i != transport_idx_t::invalid() ? i + transport_idx_offset_
                                           : transport_idx_t::invalid();
  }
  auto map(trip_direction_string_idx_t const& i) const {
    return i != trip_direction_string_idx_t::invalid()
               ? i + trip_direction_string_idx_offset_
               : trip_direction_string_idx_t::invalid();
  }
  auto map(trip_id_idx_t const& i) const {
    return i != trip_id_idx_t::invalid() ? i + trip_id_idx_offset_
                                         : trip_id_idx_t::invalid();
  }
  auto map(trip_idx_t const& i) const {
    return i != trip_idx_t::invalid() ? i + trip_idx_offset_
                                      : trip_idx_t::invalid();
  }
  auto map(trip_line_idx_t const& i) const {
    return i != trip_line_idx_t::invalid() ? i + trip_line_idx_offset_
                                           : trip_line_idx_t::invalid();
  }

  auto map(stop const& i) const {
    return stop{map(i.location_idx()), i.in_allowed_, i.out_allowed_,
                i.in_allowed_wheelchair_, i.out_allowed_wheelchair_};
  }
  auto map(trip_direction_t const& i) const {
    return i.apply([&](auto const& d) -> trip_direction_t {
      return trip_direction_t{map(d)};
    });
  }

  auto map(area const& i) const {
    return area{.id_ = map(i.id_), .name_ = map(i.name_)};
  }
  auto map(booking_rule const& i) const {
    return booking_rule{.id_ = map(i.id_),
                        .type_ = i.type_,
                        .message_ = map(i.message_),
                        .pickup_message_ = map(i.pickup_message_),
                        .drop_off_message_ = map(i.drop_off_message_),
                        .phone_number_ = map(i.phone_number_),
                        .info_url_ = map(i.info_url_),
                        .booking_url_ = map(i.booking_url_)};
  }
  auto map(fares::fare_leg_join_rule const& i) const {
    return fares::fare_leg_join_rule{i.from_network_, i.to_network_,
                                     map(i.from_stop_), map(i.to_stop_)};
  }
  auto map(fares::fare_leg_rule const& i) const {
    return fares::fare_leg_rule{
        .rule_priority_ = i.rule_priority_,
        .network_ = i.network_,
        .from_area_ = map(i.from_area_),
        .to_area_ = map(i.to_area_),
        .from_timeframe_group_ = i.from_timeframe_group_,
        .to_timeframe_group_ = i.to_timeframe_group_,
        .fare_product_ = i.fare_product_,
        .leg_group_idx_ = i.leg_group_idx_,
        .contains_exactly_area_set_id_ = i.contains_exactly_area_set_id_,
        .contains_area_set_id_ = i.contains_area_set_id_};
  }
  auto map(fares::fare_media const& i) const {
    return fares::fare_media{.name_ = map(i.name_), .type_ = i.type_};
  }
  auto map(fares::fare_product const& i) const {
    return fares::fare_product{.amount_ = i.amount_,
                               .name_ = map(i.name_),
                               .media_ = i.media_,
                               .currency_code_ = map(i.currency_code_),
                               .rider_category_ = i.rider_category_};
  }
  auto map(fares::rider_category const& i) const {
    return fares::rider_category{
        .name_ = map(i.name_),
        .eligibility_url_ = map(i.eligibility_url_),
        .is_default_fare_category_ = i.is_default_fare_category_};
  }
  auto map(fares::timeframe const& i) const {
    return fares::timeframe{.start_time_ = i.start_time_,
                            .end_time_ = i.end_time_,
                            .service_ = i.service_,
                            .service_id_ = map(i.service_id_)};
  }
  auto map(fares::network const& i) const {
    return fares::network{.id_ = map(i.id_), .name_ = map(i.name_)};
  }
  auto map(footpath const& i) const {
    return footpath{map(i.target()), i.duration()};
  }
  auto map(location_id const& i) const {
    return location_id{i.id_, map(i.src_)};
  }
  auto map(provider const& i) const {
    return provider{.id_ = map(i.id_),
                    .name_ = map(i.name_),
                    .url_ = map(i.url_),
                    .tz_ = map(i.tz_),
                    .src_ = map(i.src_)};
  }
  auto map(trip_debug const& i) const {
    return trip_debug{map(i.source_file_idx_), i.line_number_from_,
                      i.line_number_to_};
  }

  template <typename T>
  auto map(interval<T> const& i) const {
    return interval{map(i.from_), map(i.to_)};
  }

  template <typename T1, typename T2>
  auto map(pair<T1, T2> const& i) const {
    return pair<T1, T2>{map(i.first), map(i.second)};
  }

  template <typename T>
  auto map(T const& i) const {
    return i;
  }
};

void assert_unhandled_fields_are_empty(timetable const& tt) {
  // Fields not used during loading, thus not handled
  assert(tt.locations_.footpaths_out_.size() == kNProfiles);
  for (auto const& i : tt.locations_.footpaths_out_) {
    assert(i.size() == 0);
  }
  assert(tt.locations_.footpaths_in_.size() == kNProfiles);
  for (auto const& i : tt.locations_.footpaths_in_) {
    assert(i.size() == 0);
  }
  assert(tt.fwd_search_lb_graph_.size() == kNProfiles);
  for (auto const& i : tt.fwd_search_lb_graph_) {
    assert(i.size() == 0);
  }
  assert(tt.bwd_search_lb_graph_.size() == kNProfiles);
  for (auto const& i : tt.bwd_search_lb_graph_) {
    assert(i.size() == 0);
  }
  assert(tt.flex_area_locations_.size() == 0);
  assert(tt.trip_train_nr_.size() == 0);
  assert(tt.initial_day_offset_.size() == 0);
  assert(tt.profiles_.size() == 0);
}

void timetable::merge(timetable const& other_tt) {
  assert_unhandled_fields_are_empty(*this);
  assert_unhandled_fields_are_empty(other_tt);
  assert(this->date_range_ == other_tt.date_range_);

  /* Save new data */
  auto const new_bitfields = other_tt.bitfields_;
  auto const new_source_end_date = other_tt.src_end_date_;
  auto const new_trip_id_to_idx = other_tt.trip_id_to_idx_;
  auto const new_trip_ids = other_tt.trip_ids_;
  auto const new_trip_id_strings = other_tt.trip_id_strings_;
  auto const new_trip_id_src = other_tt.trip_id_src_;
  auto const new_trip_direction_id = other_tt.trip_direction_id_;
  auto const new_trip_route_id = other_tt.trip_route_id_;
  auto const new_route_ids = other_tt.route_ids_;
  auto const new_trip_transport_ranges = other_tt.trip_transport_ranges_;
  auto const new_trip_stop_seq_numbers = other_tt.trip_stop_seq_numbers_;
  auto const new_source_file_names = other_tt.source_file_names_;
  auto const new_trip_debug = other_tt.trip_debug_;
  auto const new_trip_short_names = other_tt.trip_short_names_;
  auto const new_trip_display_names = other_tt.trip_display_names_;
  auto const new_route_transport_ranges = other_tt.route_transport_ranges_;
  auto const new_route_location_seq = other_tt.route_location_seq_;
  auto const new_route_clasz = other_tt.route_clasz_;
  auto const new_route_section_clasz = other_tt.route_section_clasz_;
  auto const new_route_bikes_allowed = other_tt.route_bikes_allowed_;
  auto const new_route_cars_allowed = other_tt.route_cars_allowed_;
  auto const new_route_bikes_allowed_per_section =
      other_tt.route_bikes_allowed_per_section_;
  auto const new_route_cars_allowed_per_section =
      other_tt.route_cars_allowed_per_section_;
  auto const new_route_stop_time_ranges = other_tt.route_stop_time_ranges_;
  auto const new_route_stop_times = other_tt.route_stop_times_;
  auto const new_transport_first_dep_offset =
      other_tt.transport_first_dep_offset_;
  auto const new_transport_traffic_days = other_tt.transport_traffic_days_;
  auto const new_transport_route = other_tt.transport_route_;
  auto const new_transport_to_trip_section =
      other_tt.transport_to_trip_section_;
  auto const new_languages = other_tt.languages_;
  auto const new_locations = other_tt.locations_;
  auto const new_merged_trips = other_tt.merged_trips_;
  auto const new_attributes = other_tt.attributes_;
  auto const new_attribute_combinations = other_tt.attribute_combinations_;
  auto const new_trip_direction_strings = other_tt.trip_direction_strings_;
  auto const new_trip_directions = other_tt.trip_directions_;
  auto const new_trip_lines = other_tt.trip_lines_;
  auto const new_transport_section_attributes =
      other_tt.transport_section_attributes_;
  auto const new_transport_section_providers =
      other_tt.transport_section_providers_;
  auto const new_transport_section_directions =
      other_tt.transport_section_directions_;
  auto const new_transport_section_lines = other_tt.transport_section_lines_;
  auto const new_transport_section_route_colors =
      other_tt.transport_section_route_colors_;
  auto const new_location_routes = other_tt.location_routes_;
  auto const new_providers = other_tt.providers_;
  auto const new_provider_id_to_idx = other_tt.provider_id_to_idx_;
  auto const new_fares = other_tt.fares_;
  auto const new_areas = other_tt.areas_;
  auto const new_location_areas = other_tt.location_areas_;
  auto const new_location_location_groups = other_tt.location_location_groups_;
  auto const new_location_group_locations = other_tt.location_group_locations_;
  auto const new_location_group_name = other_tt.location_group_name_;
  auto const new_location_group_id = other_tt.location_group_id_;
  auto const new_flex_area_bbox = other_tt.flex_area_bbox_;
  auto const new_flex_area_id = other_tt.flex_area_id_;
  auto const new_flex_area_src = other_tt.flex_area_src_;
  auto const new_flex_area_outers = other_tt.flex_area_outers_;
  auto const new_flex_area_inners = other_tt.flex_area_inners_;
  auto const new_flex_area_name = other_tt.flex_area_name_;
  auto const new_flex_area_desc = other_tt.flex_area_desc_;
  auto const new_flex_area_rtree = other_tt.flex_area_rtree_;
  auto const new_location_group_transports =
      other_tt.location_group_transports_;
  auto const new_flex_area_transports = other_tt.flex_area_transports_;
  auto const new_flex_transport_traffic_days =
      other_tt.flex_transport_traffic_days_;
  auto const new_flex_transport_trip = other_tt.flex_transport_trip_;
  auto const new_flex_transport_stop_time_windows =
      other_tt.flex_transport_stop_time_windows_;
  auto const new_flex_transport_stop_seq = other_tt.flex_transport_stop_seq_;
  auto const new_flex_stop_seq = other_tt.flex_stop_seq_;
  auto const new_flex_transport_pickup_booking_rule =
      other_tt.flex_transport_pickup_booking_rule_;
  auto const new_flex_transport_drop_off_booking_rule =
      other_tt.flex_transport_drop_off_booking_rule_;
  auto const new_booking_rules = other_tt.booking_rules_;
  auto const new_strings = other_tt.strings_;
  auto const new_n_sources = other_tt.n_sources_;
  /* Add new data and adjust references */
  /*	  bitfields	*/
  auto bitfield_idx_map = vector_map<bitfield_idx_t, bitfield_idx_t>{};
  auto bitfields_ = hash_map<bitfield, bitfield_idx_t>{};
  for (auto const [idx_, bf] : utl::enumerate(this->bitfields_)) {
    auto new_idx = utl::get_or_create(bitfields_, bf, [&]() { return idx_; });
    assert(new_idx == idx_);  // bitfields must be unique in the timetable
  }
  for (auto const& [idx_, bf] : utl::enumerate(new_bitfields)) {
    auto adjusted_idx = utl::get_or_create(
        bitfields_, bf, [&]() { return this->register_bitfield(bf); });
    bitfield_idx_map.emplace_back(adjusted_idx);
  }
  /*    string_idx_t	*/
  auto string_idx_map = vector_map<string_idx_t, string_idx_t>{};
  for (auto const& [idx_, s] : utl::enumerate(new_strings.strings_)) {
    auto new_idx = this->strings_.store(s.view());
    string_idx_map.push_back(new_idx);
  }
  auto const im = index_mapping(*this, bitfield_idx_map, string_idx_map);
  /*	   sources	*/
  for (auto const& i : new_source_end_date) {
    this->src_end_date_.push_back(i);
  }
  for (auto const& i : new_source_file_names) {
    this->source_file_names_.emplace_back(i);
  }
  for (auto const& i : new_trip_debug) {
    auto entry = this->trip_debug_.emplace_back();
    for (auto const& j : i) {
      entry.emplace_back(im.map(j));
    }
  }
  /*	 languages	*/
  for (auto const& i : new_languages) {
    this->languages_.emplace_back(i);
  }
  /*   location_idx_t	*/
  {  // merge locations struct
    auto&& loc = this->locations_;
    for (auto const& i : new_locations.location_id_to_idx_) {
      auto const loc_id = im.map(i.first);
      auto const loc_idx = im.map(i.second);
      auto const [it, is_new] =
          loc.location_id_to_idx_.emplace(loc_id, loc_idx);
      if (!is_new) {
        log(log_lvl::error, "loader.load", "duplicate station {}", loc_id.id_);
      }
    }
    for (auto const& i : new_locations.names_) {
      loc.names_.emplace_back(i);
    }
    for (auto const& i : new_locations.platform_codes_) {
      loc.platform_codes_.emplace_back(i);
    }
    for (auto const& i : new_locations.descriptions_) {
      loc.descriptions_.emplace_back(i);
    }
    for (auto const& i : new_locations.ids_) {
      loc.ids_.emplace_back(i);
    }
    for (auto const& i : new_locations.alt_names_) {
      auto vec = loc.alt_names_.add_back_sized(0U);
      for (auto const& j : i) {
        vec.push_back(im.map(j));
      }
    }
    for (auto const& i : new_locations.coordinates_) {
      loc.coordinates_.push_back(i);
    }
    for (auto const& i : new_locations.src_) {
      loc.src_.push_back(im.map(i));
    }
    for (auto const& i : new_locations.transfer_time_) {
      loc.transfer_time_.push_back(i);
    }
    for (auto const& i : new_locations.types_) {
      loc.types_.push_back(i);
    }
    for (auto const& i : new_locations.parents_) {
      loc.parents_.push_back(im.map(i));
    }
    for (auto const& i : new_locations.location_timezones_) {
      loc.location_timezones_.push_back(im.map(i));
    }
    for (auto const& i : new_locations.equivalences_) {
      auto entry = loc.equivalences_.emplace_back();
      for (auto const& j : i) {
        entry.emplace_back(im.map(j));
      }
    }
    for (auto const& i : new_locations.children_) {
      auto entry = loc.children_.emplace_back();
      for (auto const& j : i) {
        entry.emplace_back(im.map(j));
      }
    }
    for (auto const& i : new_locations.preprocessing_footpaths_out_) {
      auto entry = loc.preprocessing_footpaths_out_.emplace_back();
      for (auto const& j : i) {
        entry.emplace_back(im.map(j));
      }
    }
    for (auto const& i : new_locations.preprocessing_footpaths_in_) {
      auto entry = loc.preprocessing_footpaths_in_.emplace_back();
      for (auto const& j : i) {
        entry.emplace_back(im.map(j));
      }
    }
    /*
      loc.footpaths_out_ and loc.footpaths_in_ don't get used during loading
      and are thus skipped
    */
    assert(new_locations.footpaths_out_.size() == kNProfiles);
    for (auto const& i : new_locations.footpaths_out_) {
      assert(i.size() == 0);
    }
    assert(new_locations.footpaths_in_.size() == kNProfiles);
    for (auto const& i : new_locations.footpaths_in_) {
      assert(i.size() == 0);
    }
    for (auto const& i : new_locations.timezones_) {
      loc.timezones_.push_back(i);
    }
    /*
      loc.location_importance_ doesn't get used during loading and is thus
      skipped
    */
    assert(loc.location_importance_.size() == 0);
    for (auto const& i : new_locations.alt_name_strings_) {
      loc.alt_name_strings_.emplace_back(i);
    }
    for (auto const& i : new_locations.alt_name_langs_) {
      loc.alt_name_langs_.push_back(im.map(i));
    }
    /*
      loc.max_importance_ and loc.rtree_ don't get used during loading
      and are thus skipped
    */
    assert(loc.max_importance_ == 0U);
  }  // end of locations struct
  for (auto const& i : new_location_routes) {
    auto vec = this->location_routes_.add_back_sized(0U);
    for (auto const& j : i) {
      vec.push_back(im.map(j));
    }
  }
  for (auto const& i : new_location_areas) {
    auto vec = this->location_areas_.add_back_sized(0U);
    for (auto const& j : i) {
      vec.push_back(im.map(j));
    }
  }
  for (location_idx_t i = location_idx_t{0};
       i < location_idx_t{new_location_location_groups.size()}; ++i) {
    this->location_location_groups_.emplace_back_empty();
    for (auto const& j : new_location_location_groups[i]) {
      this->location_location_groups_.back().push_back(im.map(j));
    }
  }
  for (location_group_idx_t i = location_group_idx_t{0};
       i < location_group_idx_t{new_location_group_locations.size()}; ++i) {
    this->location_group_locations_.emplace_back_empty();
    for (auto const& j :
         new_location_group_locations[location_group_idx_t{i}]) {
      this->location_group_locations_.back().push_back(im.map(j));
    }
  }
  for (auto const& i : new_location_group_name) {
    this->location_group_name_.emplace_back(im.map(i));
  }
  for (auto const& i : new_location_group_id) {
    this->location_group_id_.emplace_back(im.map(i));
  }
  // tt.fwd_search_lb_graph_ not used during loading
  assert(this->fwd_search_lb_graph_.size() == kNProfiles);
  for (auto const& i : this->fwd_search_lb_graph_) {
    assert(i.size() == 0);
  }
  // bwd_search_lb_graph_ not used during loading
  assert(this->bwd_search_lb_graph_.size() == kNProfiles);
  for (auto const& i : this->bwd_search_lb_graph_) {
    assert(i.size() == 0);
  }
  /*     route_idx_t	*/
  for (auto const& i : new_route_transport_ranges) {
    this->route_transport_ranges_.push_back(im.map(i));
  }
  for (auto const& i : new_route_location_seq) {
    auto vec = this->route_location_seq_.add_back_sized(0U);
    for (auto const& j : i) {
      vec.push_back(im.map(stop{j}).value());
    }
  }
  for (auto const& i : new_route_clasz) {
    this->route_clasz_.emplace_back(i);
  }
  for (auto const& i : new_route_section_clasz) {
    this->route_section_clasz_.emplace_back(i);
  }
  for (auto const& i : new_route_bikes_allowed_per_section) {
    this->route_bikes_allowed_per_section_.emplace_back(i);
  }
  for (auto const& i : new_route_cars_allowed_per_section) {
    this->route_cars_allowed_per_section_.emplace_back(i);
  }
  auto const new_route_bikes_allowed_size = new_route_bikes_allowed.size();
  auto const route_bikes_allowed_size = this->route_bikes_allowed_.size();
  this->route_bikes_allowed_.resize(route_bikes_allowed_size +
                                    new_route_bikes_allowed_size);
  for (auto const& i : vw::iota(0U, new_route_bikes_allowed_size)) {
    this->route_bikes_allowed_.set(route_bikes_allowed_size + i,
                                   new_route_bikes_allowed.test(i));
  }
  auto const new_route_cars_allowed_size = new_route_cars_allowed.size();
  auto const route_cars_allowed_size = this->route_cars_allowed_.size();
  this->route_cars_allowed_.resize(route_cars_allowed_size +
                                   new_route_cars_allowed_size);
  for (auto const& i : vw::iota(0U, new_route_cars_allowed_size)) {
    this->route_cars_allowed_.set(route_cars_allowed_size + i,
                                  new_route_cars_allowed.test(i));
  }
  auto const route_stop_times_offset = this->route_stop_times_.size();
  for (auto const& i : new_route_stop_time_ranges) {
    this->route_stop_time_ranges_.push_back(interval{
        i.from_ + route_stop_times_offset, i.to_ + route_stop_times_offset});
  }
  for (auto const& i : new_route_stop_times) {
    this->route_stop_times_.push_back(i);
  }
  for (auto const& i : new_transport_route) {
    this->transport_route_.push_back(im.map(i));
  }
  /*        fares	*/
  for (auto const& i : new_fares) {
    auto mapped_leg_group_name = vector_map<leg_group_idx_t, string_idx_t>{};
    for (auto const& j : i.leg_group_name_) {
      mapped_leg_group_name.push_back(im.map(j));
    }
    auto mapped_fare_media = vector_map<fare_media_idx_t, fares::fare_media>{};
    for (auto const& j : i.fare_media_) {
      mapped_fare_media.push_back(im.map(j));
    }
    auto mapped_fare_products =
        vecvec<fare_product_idx_t, fares::fare_product>{};
    for (auto const& j : i.fare_products_) {
      auto vec = mapped_fare_products.add_back_sized(0U);
      for (auto const& k : j) {
        vec.push_back(im.map(k));
      }
    }
    auto mapped_fare_product_id =
        vector_map<fare_product_idx_t, string_idx_t>{};
    for (auto const& j : i.fare_product_id_) {
      mapped_fare_product_id.push_back(im.map(j));
    }
    auto mapped_fare_leg_rules = vector<fares::fare_leg_rule>{};
    for (auto const& j : i.fare_leg_rules_) {
      mapped_fare_leg_rules.push_back(im.map(j));
    }
    auto mapped_fare_leg_join_rules = vector<fares::fare_leg_join_rule>{};
    for (auto const& j : i.fare_leg_join_rules_) {
      mapped_fare_leg_join_rules.push_back(im.map(j));
    }
    auto mapped_rider_categories =
        vector_map<rider_category_idx_t, fares::rider_category>{};
    for (auto const& j : i.rider_categories_) {
      mapped_rider_categories.push_back(im.map(j));
    }
    auto mapped_timeframes = vecvec<timeframe_group_idx_t, fares::timeframe>{};
    for (auto const& j : i.timeframes_) {
      auto vec = mapped_timeframes.add_back_sized(0U);
      for (auto const& k : j) {
        vec.push_back(im.map(k));
      }
    }
    auto mapped_timeframe_id =
        vector_map<timeframe_group_idx_t, string_idx_t>{};
    for (auto const& j : i.timeframe_id_) {
      mapped_timeframe_id.push_back(im.map(j));
    }
    auto mapped_networks = vector_map<network_idx_t, fares::network>{};
    for (auto const& j : i.networks_) {
      mapped_networks.push_back(im.map(j));
    }
    auto mapped_area_sets = vecvec<area_set_idx_t, area_idx_t>{};
    for (auto const& j : i.area_sets_) {
      auto vec = mapped_area_sets.add_back_sized(0U);
      for (auto const& k : j) {
        vec.push_back(im.map(k));
      }
    }
    auto mapped_area_set_ids = vector_map<area_set_idx_t, string_idx_t>{};
    for (auto const& j : i.area_set_ids_) {
      mapped_area_set_ids.push_back(im.map(j));
    }
    auto const mapped_fares =
        fares{.leg_group_name_ = mapped_leg_group_name,
              .fare_media_ = mapped_fare_media,
              .fare_products_ = mapped_fare_products,
              .fare_product_id_ = mapped_fare_product_id,
              .fare_leg_rules_ = mapped_fare_leg_rules,
              .fare_leg_join_rules_ = mapped_fare_leg_join_rules,
              .fare_transfer_rules_ = i.fare_transfer_rules_,
              .rider_categories_ = mapped_rider_categories,
              .timeframes_ = mapped_timeframes,
              .timeframe_id_ = mapped_timeframe_id,
              .route_networks_ = i.route_networks_,
              .networks_ = mapped_networks,
              .area_sets_ = mapped_area_sets,
              .area_set_ids_ = mapped_area_set_ids,
              .has_priority_ = i.has_priority_};
    this->fares_.push_back(mapped_fares);
  }
  /*   provider_idx_t	*/
  for (auto const& i : new_providers) {
    this->providers_.push_back(im.map(i));
  }
  for (auto const& i : new_provider_id_to_idx) {
    this->provider_id_to_idx_.push_back(im.map(i));
  }
  /*	    Flex	*/
  for (auto const& i : new_flex_area_bbox) {
    this->flex_area_bbox_.push_back(i);
  }
  for (auto const& i : new_flex_area_id) {
    this->flex_area_id_.push_back(im.map(i));
  }
  for (auto const& i : new_flex_area_src) {
    this->flex_area_src_.push_back(im.map(i));
  }
  // tt.flex_area_locations_ not used during loading
  assert(this->flex_area_locations_.size() == 0);
  for (auto const& i : new_flex_area_outers) {
    this->flex_area_outers_.emplace_back(i);
  }
  for (auto const& i : new_flex_area_inners) {
    this->flex_area_inners_.emplace_back(i);
  }
  for (auto const& i : new_flex_area_name) {
    this->flex_area_name_.emplace_back(i);
  }
  for (auto const& i : new_flex_area_desc) {
    this->flex_area_desc_.emplace_back(i);
  }
  for (auto const& n : new_flex_area_rtree.nodes_) {
    if (n.kind_ == rtree<flex_area_idx_t>::kind::kLeaf) {
      for (size_t i = 0; i < n.count_; ++i) {
        this->flex_area_rtree_.insert(n.rects_[i].min_, n.rects_[i].max_,
                                      n.data_[i]);
      }
    }
  }
  for (location_group_idx_t i = location_group_idx_t{0};
       i < location_group_idx_t{new_location_group_transports.size()}; ++i) {
    this->location_group_transports_.emplace_back_empty();
    for (auto const& j : new_location_group_transports[i]) {
      this->location_group_transports_.back().push_back(im.map(j));
    }
  }
  for (flex_area_idx_t i = flex_area_idx_t{0};
       i < flex_area_idx_t{new_flex_area_transports.size()}; ++i) {
    this->flex_area_transports_.emplace_back_empty();
    for (auto const& j : new_flex_area_transports[i]) {
      this->flex_area_transports_.back().push_back(im.map(j));
    }
  }
  for (auto const& i : new_flex_transport_traffic_days) {
    this->flex_transport_traffic_days_.push_back(im.map(i));
  }
  for (auto const& i : new_flex_transport_trip) {
    this->flex_transport_trip_.push_back(im.map(i));
  }
  for (auto const& i : new_flex_transport_stop_time_windows) {
    this->flex_transport_stop_time_windows_.emplace_back(i);
  }
  for (auto const& i : new_flex_transport_stop_seq) {
    this->flex_transport_stop_seq_.push_back(im.map(i));
  }
  for (auto const& i : new_flex_stop_seq) {
    this->flex_stop_seq_.emplace_back(i);
  }
  for (auto const& i : new_flex_transport_pickup_booking_rule) {
    auto vec = this->flex_transport_pickup_booking_rule_.add_back_sized(0U);
    for (auto const& j : i) {
      vec.push_back(im.map(j));
    }
  }
  for (auto const& i : new_flex_transport_drop_off_booking_rule) {
    auto vec = this->flex_transport_drop_off_booking_rule_.add_back_sized(0U);
    for (auto const& j : i) {
      vec.push_back(im.map(j));
    }
  }
  for (auto const& i : new_booking_rules) {
    this->booking_rules_.push_back(im.map(i));
  }
  /*    trip_id_idx_t	*/
  for (auto const& i : new_trip_id_to_idx) {
    this->trip_id_to_idx_.push_back(im.map(i));
  }
  for (auto const& i : new_trip_ids) {
    auto entry = this->trip_ids_.emplace_back();
    for (auto const& j : i) {
      auto trip_id = trip_id_idx_t{im.map(j)};
      entry.emplace_back(trip_id);
    }
  }
  for (auto const& i : new_trip_id_src) {
    this->trip_id_src_.push_back(im.map(i));
  }
  for (auto const& i : new_trip_id_strings) {
    this->trip_id_strings_.emplace_back(i);
  }
  // tt.trip_train_nr_ not used during loading
  assert(this->trip_train_nr_.size() == 0);
  /*      trip_idx_t	*/
  auto const add_size = trip_idx_t{new_trip_direction_id.size()};
  this->trip_direction_id_.resize(to_idx(im.map(add_size)));
  for (auto const& i : vw::iota(0U, to_idx(add_size))) {
    auto const idx = trip_idx_t{i};
    this->trip_direction_id_.set(im.map(idx), new_trip_direction_id.test(idx));
  }
  for (auto const& i : new_trip_route_id) {
    this->trip_route_id_.push_back(i);
  }
  for (auto i = trip_idx_t{0}; i < trip_idx_t{new_trip_transport_ranges.size()};
       ++i) {
    this->trip_transport_ranges_.emplace_back(new_trip_transport_ranges[i]);
  }
  for (auto const& i : new_trip_stop_seq_numbers) {
    this->trip_stop_seq_numbers_.emplace_back(i);
  }
  for (auto const& i : new_trip_short_names) {
    this->trip_short_names_.emplace_back(i);
  }
  for (auto const& i : new_trip_display_names) {
    this->trip_display_names_.emplace_back(i);
  }
  for (auto const& i : new_merged_trips) {
    auto vec = this->merged_trips_.add_back_sized(0U);
    for (auto const& j : i) {
      vec.push_back(im.map(j));
    }
  }
  /*    route_id_idx_t	 */
  for (auto const& i : new_route_ids) {
    auto vec = paged_vecvec<route_id_idx_t, trip_idx_t>{};
    for (route_id_idx_t j = route_id_idx_t{0};
         j < route_id_idx_t{i.route_id_trips_.size()}; ++j) {
      vec.emplace_back_empty();
      for (auto const& k : i.route_id_trips_[j]) {
        vec.back().push_back(im.map(k));
      }
    }
    auto mapped_providers = vector_map<route_id_idx_t, provider_idx_t>{};
    for (auto const& j : i.route_id_provider_) {
      mapped_providers.push_back(im.map(j));
    }
    auto const mapped_route_ids =
        timetable::route_ids{.route_id_short_names_ = i.route_id_short_names_,
                             .route_id_long_names_ = i.route_id_long_names_,
                             .route_id_type_ = i.route_id_type_,
                             .route_id_provider_ = mapped_providers,
                             .route_id_colors_ = i.route_id_colors_,
                             .route_id_trips_ = vec,
                             .ids_ = i.ids_};
    this->route_ids_.push_back(mapped_route_ids);
  }
  /*   transport_idx_t	*/
  for (auto const& i : new_transport_first_dep_offset) {
    this->transport_first_dep_offset_.push_back(i);
  }
  // tt.initial_day_offset_ not used during loading
  assert(this->initial_day_offset_.size() == 0);
  for (auto const& i : new_transport_traffic_days) {
    this->transport_traffic_days_.push_back(im.map(i));
  }
  for (auto const& i : new_transport_to_trip_section) {
    auto vec = this->transport_to_trip_section_.add_back_sized(0U);
    for (auto const& j : i) {
      vec.push_back(im.map(j));
    }
  }
  for (auto const& i : new_transport_section_attributes) {
    this->transport_section_attributes_.emplace_back(i);
  }
  for (auto const& i : new_transport_section_providers) {
    auto vec = this->transport_section_providers_.add_back_sized(0U);
    for (auto const& j : i) {
      vec.push_back(im.map(j));
    }
  }
  for (auto const& i : new_transport_section_directions) {
    this->transport_section_directions_.emplace_back(i);
  }
  for (auto const& i : new_transport_section_lines) {
    auto vec = this->transport_section_lines_.add_back_sized(0U);
    for (auto const& j : i) {
      vec.push_back(im.map(j));
    }
  }
  for (auto const& i : new_transport_section_route_colors) {
    this->transport_section_route_colors_.emplace_back(i);
  }
  /*     Meta infos	*/
  for (auto const& i : new_trip_lines) {
    this->trip_lines_.emplace_back(i);
  }
  /*     area_idx_t	*/
  for (auto const& i : new_areas) {
    this->areas_.push_back(im.map(i));
  }
  /*   attribute_idx_t	*/
  for (auto const& i : new_attributes) {
    this->attributes_.push_back(i);
  }
  for (auto const& i : new_attribute_combinations) {
    auto vec = this->attribute_combinations_.add_back_sized(0U);
    for (auto const& j : i) {
      vec.push_back(im.map(j));
    }
  }
  /*    trip_direction_string_idx_t	*/
  for (auto const& i : new_trip_direction_strings) {
    this->trip_direction_strings_.emplace_back(i);
  }
  for (auto const& i : new_trip_directions) {
    this->trip_directions_.push_back(im.map(i));
  }
  /*    Other	*/
  this->n_sources_ += new_n_sources;
  assert(this->n_sources_ == this->src_end_date_.size());
  // tt.profiles_ not used during loading
  assert(this->profiles_.size() == 0);
  // tt.date_range_ not changed
  assert(this->date_range_ == other_tt.date_range_);
}

}  // Namespace nigiri
