#pragma once

#include <string>

#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

using locations_map = hash_map<std::string, location_idx_t>;

struct gtfs_seated_transfer {
  location_idx_t from_{location_idx_t::invalid()};
  location_idx_t to_{location_idx_t::invalid()};
  std::string to_trip_id_;
};

using seated_transfers_map =
    hash_map<std::string /* from_trip_id */, std::vector<gtfs_seated_transfer>>;

std::pair<locations_map, seated_transfers_map> read_stops(
    source_idx_t,
    timetable&,
    tz_map&,
    std::string_view stops_file_content,
    std::string_view transfers_file_content);

}  // namespace nigiri::loader::gtfs
