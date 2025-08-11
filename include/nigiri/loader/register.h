#pragma once

#include <span>

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

struct agency {
  agency(timetable const&, provider_idx_t);

  std::string_view get_name() const;
  void set_name(std::string_view);

  std::string_view get_url() const;
  void set_url(std::string_view);

  std::optional<std::string_view> get_timezone() const;
  void set_timezone(std::string_view);

  std::string_view id_;
  cista::raw::generic_string name_;
  cista::raw::generic_string url_;
  timezone_idx_t timezone_idx_;

  timetable* tt_{nullptr};
  hash_map<std::string, timezone_idx_t>* tz_map_{nullptr};
};

struct location {
  location(timetable const&, location_idx_t);

  std::string_view get_name() const;
  void set_name(std::string_view);

  std::string_view get_platform_code() const;
  void set_platform_code(std::string_view);

  std::string_view get_description() const;
  void set_description(std::string_view);

  geo::latlng get_pos() const;
  void set_pos(geo::latlng);

  std::optional<location> get_parent() const;

  std::optional<std::string_view> get_timezone() const;
  void set_timezone(std::string_view);

  duration_t::rep get_transfer_time() const;
  void set_transfer_time(duration_t::rep);

  source_idx_t src_;
  std::string_view id_;
  cista::raw::generic_string name_;
  cista::raw::generic_string platform_code_;
  cista::raw::generic_string description_;
  geo::latlng pos_;
  location_type type_;
  location_idx_t parent_;
  timezone_idx_t timezone_idx_;
  duration_t transfer_time_;
  std::span<location_idx_t const> equivalences_;

  timetable* tt_{nullptr};
  hash_map<std::string, timezone_idx_t>* tz_map_{nullptr};
};

struct route {
  route(timetable const&, source_idx_t, route_id_idx_t);

  std::string_view get_short_name() const;
  void set_short_name(std::string_view);

  std::string_view get_long_name() const;
  void set_long_name(std::string_view);

  std::uint16_t get_route_type() const;
  void set_route_type(std::uint16_t);

  color_t::value_t get_color() const;
  void set_color(color_t::value_t);

  color_t::value_t get_text_color() const;
  void set_text_color(color_t::value_t);

  provider const& get_agency() const;

  source_idx_t src_;
  std::string_view id_;
  cista::raw::generic_string short_name_;
  cista::raw::generic_string long_name_;
  route_type_t route_type_;
  route_color color_;
  provider_idx_t agency_;
};

struct trip {
  std::string_view get_id() const;

  std::string_view get_headsign() const;
  void set_headsign(std::string_view);

  std::string_view get_short_name() const;
  void set_short_name(std::string_view);

  std::string_view get_display_name() const;
  void set_display_name(std::string_view);

  route get_route() const;

  source_idx_t src_;
  std::string_view id_;
  cista::raw::generic_string headsign_;
  cista::raw::generic_string short_name_;
  cista::raw::generic_string display_name_;
  std::span<stop_idx_t> seq_numbers_;
  direction_id_t direction_;
  trip_debug dbg_;
  route_id_idx_t route_;

  timetable* tt_{nullptr};
  hash_map<std::string, timezone_idx_t>* tz_map_{nullptr};
};

location_idx_t register_location(timetable&, location const&);
route_id_idx_t register_route(timetable&, route const&);
trip_idx_t register_trip(timetable&, trip const&);

}  // namespace nigiri::loader