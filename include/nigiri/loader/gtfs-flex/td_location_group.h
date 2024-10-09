#pragma once

#include <nigiri/types.h>

namespace nigiri::loader::gtfs_flex {

struct td_location_group {
  std::vector<std::string> location_ids_; // Optional
  std::string location_group_name_;       // Optional
};

using td_location_group_map_t = hash_map<std::string, td_location_group>;

td_location_group_map_t read_td_location_groups(std::string_view file_content);

}  // namespace nigiri::loader::gtfs_flex