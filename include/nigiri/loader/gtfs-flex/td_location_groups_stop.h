#pragma once

#include <nigiri/types.h>

namespace nigiri::loader::gtfs_flex {

  using td_location_group_stop_map_t = hash_map<std::string, std::vector<std::string>>;

  td_location_group_stop_map_t read_td_location_group_stops(std::string_view file_content);

}  // namespace nigiri::loader::gtfs_flex