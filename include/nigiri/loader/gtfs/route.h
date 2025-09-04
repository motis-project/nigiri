#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/loader/register.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

struct route {
  route_id_idx_t route_id_idx_;
  provider_idx_t agency_;
  std::string id_;
  std::string short_name_;
  std::string long_name_;
  std::string network_;
  clasz clasz_;
  color_t color_;
  color_t text_color_;
};

using route_map_t = hash_map<std::string, std::unique_ptr<route>>;

clasz to_clasz(std::uint16_t route_type);

route_map_t read_routes(source_idx_t,
                        timetable&,
                        tz_map&,
                        agency_map_t&,
                        std::string_view file_content,
                        std::string_view default_tz,
                        script_runner const& = script_runner{});

}  // namespace nigiri::loader::gtfs
