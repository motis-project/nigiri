#pragma once

#include <span>

#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

extern gtfs::tz_map dummy_tz_map;

struct attribute {
  attribute(timetable*, std::string_view code, translation_idx_t text);

  std::string_view get_code() const;
  void set_code(std::string_view);

  std::string_view get_text() const;
  translated_str_t get_text_translations() const;
  void set_text(translated_str_t);

  generic_string code_;
  translation_idx_t text_;
  timetable* tt_{nullptr};
};

struct agency {
  agency(timetable&,
         source_idx_t,
         std::string_view id,
         translation_idx_t name,
         translation_idx_t url,
         timezone_idx_t,
         gtfs::tz_map& = dummy_tz_map);
  agency(timetable&, provider_idx_t);

  std::string_view get_id() const;

  std::string_view get_name() const;
  translated_str_t get_name_translations() const;
  void set_name(translated_str_t);

  std::string_view get_url() const;
  translated_str_t get_url_translations() const;
  void set_url(translated_str_t);

  std::optional<std::string_view> get_timezone() const;
  void set_timezone(std::string_view);

  source_idx_t src_;

  std::string_view id_;
  translation_idx_t name_;
  translation_idx_t url_;
  timezone_idx_t timezone_idx_;

  timetable* tt_{nullptr};
  hash_map<std::string, timezone_idx_t>* tz_map_{nullptr};
};

struct location {
  location(timetable&,
           source_idx_t,
           std::string_view id,
           translation_idx_t name,
           translation_idx_t platform_code,
           translation_idx_t desc,
           geo::latlng pos,
           location_type,
           location_idx_t parent,
           timezone_idx_t,
           duration_t transfer_time,
           gtfs::tz_map& = dummy_tz_map);
  location(timetable const&, location_idx_t);

  std::string_view get_id() const;

  std::string_view get_name() const;
  translated_str_t get_name_translations() const;
  void set_name(translated_str_t);

  std::string_view get_platform_code() const;
  translated_str_t get_platform_code_translations() const;
  void set_platform_code(translated_str_t);

  std::string_view get_description() const;
  translated_str_t get_description_translations() const;
  void set_description(translated_str_t);

  geo::latlng get_pos() const;
  void set_pos(geo::latlng);

  std::optional<std::string_view> get_timezone() const;
  void set_timezone(std::string_view);

  duration_t::rep get_transfer_time() const;
  void set_transfer_time(duration_t::rep);

  source_idx_t src_;
  std::string_view id_;
  translation_idx_t name_;
  translation_idx_t platform_code_;
  translation_idx_t description_;
  geo::latlng pos_;
  location_type type_;
  location_idx_t parent_;
  timezone_idx_t timezone_idx_;
  duration_t transfer_time_;

  timetable* tt_{nullptr};
  gtfs::tz_map* tz_map_{nullptr};
};

struct route {
  route(timetable&,
        source_idx_t,
        std::string_view id,
        translation_idx_t short_name,
        translation_idx_t long_name,
        route_type_t,
        route_color,
        provider_idx_t);
  route(timetable&, source_idx_t, route_id_idx_t);

  std::string_view get_id() const;

  std::string_view get_short_name() const;
  std::vector<translation> get_short_name_translations() const;
  void set_short_name(translated_str_t);

  std::string_view get_long_name() const;
  translated_str_t get_long_name_translations() const;
  void set_long_name(translated_str_t);

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
  translation_idx_t short_name_;
  translation_idx_t long_name_;
  route_type_t route_type_;
  route_color color_;
  clasz clasz_;
  provider_idx_t agency_;

  timetable* tt_{nullptr};
};

struct trip {
  trip(timetable&,
       source_idx_t,
       std::string_view id,
       translation_idx_t headsign,
       translation_idx_t short_name,
       translation_idx_t display_name,
       std::string_view vehicle_type_name,
       std::string_view vehicle_type_short_name,
       direction_id_t,
       route_id_idx_t,
       trip_debug);

  std::string_view get_id() const;

  std::string_view get_headsign() const;
  translated_str_t get_headsign_translations() const;
  void set_headsign(translated_str_t);

  std::string_view get_short_name() const;
  translated_str_t get_short_name_translations() const;
  void set_short_name(translated_str_t);

  std::string_view get_vehicle_type_name() const;
  std::string_view get_vehicle_type_short_name() const;

  std::string_view get_display_name() const;
  translated_str_t get_display_name_translations() const;
  void set_display_name(translated_str_t);

  std::vector<attribute> get_attributes() const;
  void set_attributes(std::vector<attribute> const&);

  route get_route() const;

  source_idx_t src_;
  std::string_view id_;
  translation_idx_t headsign_;
  translation_idx_t short_name_;
  translation_idx_t display_name_;
  std::string_view vehicle_type_name_;
  std::string_view vehicle_type_short_name_;
  direction_id_t direction_;
  route_id_idx_t route_;
  trip_debug dbg_;

  timetable* tt_{nullptr};
};

struct script_runner {
  script_runner();
  explicit script_runner(std::string const&);
  ~script_runner();

  struct impl;
  std::unique_ptr<impl> impl_;
};

bool process_attribute(script_runner const&, attribute&);
bool process_location(script_runner const&, location&);
bool process_agency(script_runner const&, agency&);
bool process_route(script_runner const&, route&);
bool process_trip(script_runner const&, trip&);

attribute_idx_t register_attribute(timetable&, attribute const&);
provider_idx_t register_agency(timetable&, agency const&);
location_idx_t register_location(timetable&, location const&);
route_id_idx_t register_route(timetable&, route const&);
trip_idx_t register_trip(timetable&, trip const&);

}  // namespace nigiri::loader
