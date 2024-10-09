#pragma once

#include <nigiri/types.h>

namespace nigiri::loader::gtfs_flex {

using td_stop_area_map_t = hash_map<std::string, std::vector<std::string>>;

td_stop_area_map_t read_td_stop_areas(std::string_view file_content);

}  // namespace nigiri::loader::gtfs_flex