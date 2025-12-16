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
};

using route_map_t = hash_map<std::string, std::unique_ptr<route>>;

clasz to_clasz(std::uint16_t);
clasz to_clasz(route_type_t);

route_map_t read_routes(source_idx_t,
                        timetable&,
                        translator&,
                        tz_map&,
                        agency_map_t&,
                        std::string_view file_content,
                        std::string_view default_tz,
                        script_runner const& = script_runner{});

}  // namespace nigiri::loader::gtfs
