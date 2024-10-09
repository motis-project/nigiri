#pragma once

#include <nigiri/types.h>

namespace nigiri::loader::gtfs_flex {
const uint8_t LOCATION_STOP = 0;
const uint8_t LOCATION_STATION = 1;
const uint8_t LOCATION_ENTRANCE_EXIT = 2;
const uint8_t LOCATION_GENERIC_NODE = 3;
const uint8_t LOCATION_BOARDING_AREA = 4;

struct td_stop {
  std::string code_;                // Optional
  std::string name_;                // Conditionally Required If location_type_ <= 2
  std::string tts_name_;            // Optional
  std::string desc_;                // Optional
  std::string lat_;                 // Conditionally Required If location_type_ <= 2
  std::string lon_;                 // Conditionally Required If location_type_ <= 2
  std::string zone_id_;             // Optional
  std::string url_;                 // Optional
  uint8_t location_type_;           // Optional
  std::string parent_station_;      // Conditionally Required If location_type_ >= 2
  std::string timezone_;            // Optional
  uint8_t wheelchair_boarding_;     // Optional
  std::string level_id_;            // Optional
  std::string platform_code_;       // Optional
};

using td_stop_map_t = hash_map<std::string, std::unique_ptr<td_stop>>;

td_stop_map_t read_td_stops(std::string_view file_content);

}  // namespace nigiri::loader::gtfs_flex