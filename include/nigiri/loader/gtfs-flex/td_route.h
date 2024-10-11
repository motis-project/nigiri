#pragma once

#include <nigiri/types.h>

namespace nigiri::loader::gtfs_flex {

struct td_route {
  std::string agency_id_;       // Conditionally Required If Multiple Agencies Are Defined in agency.txt
  std::string short_name_;      // Conditionally Required If long_name_ Is Empty
  std::string long_name_;       // Conditionally Required If short_name_ Is Empty
  std::string desc_;            // Optional
  uint16_t type_;               // Required
  std::string url_;             // Optional
  std::string color_;           // Optional
  std::string text_color_;      // Optional
  std::string sort_order_;      // Optional
  uint8_t continuous_pickup_;   // Conditionally Forbidden If stop_times.start_pickup_drop_off_window Or stop_times.end_pickup_drop_off_window Are Defined
  uint8_t continuous_drop_off_; // Conditionally Forbidden If stop_times.start_pickup_drop_off_window Or stop_times.end_pickup_drop_off_window Are Defined
  std::string network_id_;      // Conditionally Forbidden If route_networks.txt Exists
};

using td_route_map_t = hash_map<std::string, std::unique_ptr<td_route>>;

td_route_map_t read_td_routes(std::string_view file_content);

}  // namespace nigiri::loader::gtfs_flex