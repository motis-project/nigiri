#pragma once
#include <nigiri/types.h>

namespace nigiri::loader::gtfs_flex {

struct td_trip {
  std::string route_id_;
  std::string service_id_;
  std::string shape_id_;
  std::string trip_headsign_;
  std::string trip_short_name_;
  uint8_t direction_id_;
  std::string block_id_;
  uint8_t wheelchair_accessible_;
  uint8_t bikes_allowed_;
  std::string trip_note_;
  std::string route_direction_;
};

using td_trip_map_t = hash_map<std::string, std::unique_ptr<td_trip>>;

td_trip_map_t read_td_trips(std::string_view file_content);

}  // namespace nigiri::loader::gtfs_flex