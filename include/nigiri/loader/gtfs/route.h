#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/translations.h"
#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/loader/register.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

struct route {
  route_id_idx_t route_id_idx_;
  std::string network_;
  std::string ticketing_deep_link_id_;
};

using route_map_t = hash_map<std::string, std::unique_ptr<route>>;

clasz to_clasz(std::uint16_t);
clasz to_clasz(route_type_t);

inline display_type parse_display_type(std::uint8_t val) {
  switch (val) {
    case 0: return display_type::kUnset;
    case 1: return display_type::kRequired;
    case 2: return display_type::kOptional;
    case 3: return display_type::kDetailsOnly;
  }

  log(log_lvl::error, "gtfs.route", "Unknown display type {}", val);

  return display_type::kUnset;
}

route_map_t read_routes(source_idx_t,
                        timetable&,
                        translator&,
                        tz_map&,
                        agency_map_t&,
                        std::string_view file_content,
                        std::string_view default_tz,
                        script_runner const& = script_runner{});

}  // namespace nigiri::loader::gtfs
