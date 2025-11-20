#pragma once

#include <span>

#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

extern gtfs::tz_map dummy_tz_map;

struct agency {
  agency(source_idx_t,
         std::string_view id,
         std::string_view name,
         std::string_view url,
         timezone_idx_t,
         timetable&,
         gtfs::tz_map& = dummy_tz_map);
  agency(timetable const&, provider_idx_t);

  std::string_view get_id() const;

  std::string_view get_name() const;
  void set_name(std::string_view);

  std::string_view get_url() const;
  void set_url(std::string_view);

  std::optional<std::string_view> get_timezone() const;
  void set_timezone(std::string_view);

  source_idx_t src_;

  std::string_view id_;
  cista::raw::generic_string name_;
  cista::raw::generic_string url_;
  timezone_idx_t timezone_idx_;

  timetable* tt_{nullptr};
  hash_map<std::string, timezone_idx_t>* tz_map_{nullptr};
};

struct location {
  location(std::string_view id,
           std::string_view name,
           std::string_view platform_code,
           std::string_view desc,
           geo::latlng pos,
           source_idx_t,
           location_type,
           location_idx_t parent,
           timezone_idx_t,
           duration_t transfer_time,
           timetable&,
           gtfs::tz_map& = dummy_tz_map);
  location(timetable const&, location_idx_t);

  std::string_view get_id() const;

  std::string_view get_name() const;
  void set_name(std::string_view);

  std::string_view get_platform_code() const;
  void set_platform_code(std::string_view);

  std::string_view get_description() const;
  void set_description(std::string_view);

  geo::latlng get_pos() const;
  void set_pos(geo::latlng);

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

  timetable* tt_{nullptr};
  gtfs::tz_map* tz_map_{nullptr};
};

struct route {
  route(timetable const&,
        source_idx_t,
        std::string_view id,
        std::string_view short_name,
        std::string_view long_name,
        route_type_t,
        route_color,
        provider_idx_t);
  route(timetable const&, source_idx_t, route_id_idx_t);

  std::string_view get_id() const;

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

  clasz get_clasz() const;
  void set_clasz(clasz);

  agency get_agency() const;

  source_idx_t src_;
  std::string_view id_;
  cista::raw::generic_string short_name_;
  cista::raw::generic_string long_name_;
  route_type_t route_type_;
  route_color color_;
  clasz clasz_;
  provider_idx_t agency_;

  timetable const& tt_;
};

struct trip {
  trip(source_idx_t,
       std::string_view id,
       std::string_view headsign,
       std::string_view short_name,
       std::string_view display_name,
       direction_id_t,
       route_id_idx_t,
       timetable&);

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
  direction_id_t direction_;
  route_id_idx_t route_;

  timetable* tt_{nullptr};
};

struct script_runner {
  script_runner();
  explicit script_runner(std::string const&);
  ~script_runner();

  struct impl;
  std::unique_ptr<impl> impl_;
};

bool process_location(script_runner const&, location&);
bool process_agency(script_runner const&, agency&);
bool process_route(script_runner const&, route&);
bool process_trip(script_runner const&, trip&);

provider_idx_t register_agency(timetable&, agency const&);
location_idx_t register_location(timetable&, location const&);
route_id_idx_t register_route(timetable&, route const&);
trip_idx_t register_trip(timetable&, trip const&);

}  // namespace nigiri::loader