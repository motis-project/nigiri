#pragma once

#include <nigiri/types.h>

namespace nigiri::loader::gtfs_flex {

struct td_stop_time {
  uint16_t stop_sequence_;                    // Required
  std::string stop_headsign_;                 // Optional
  duration_t arrival_time_;                  // Conditionally Required If timepoint == 1
  duration_t departure_time_;                // Conditionally Required If timepoint == 1
  duration_t start_pickup_drop_off_window_;  // Conditionally Required If location_group_id Or location_id Is Defined
  duration_t end_pickup_drop_off_window_;    // Conditionally Required If location_group_id Or location_id Is Defined
  double_t shape_dist_traveled_;              // Optional
  uint8_t timepoint_;                         // Optional
  std::string stop_note_;                     // Optional
  double_t mean_duration_factor_;             // Conditionally Forbidden
  double_t mean_duration_offset_;             // Conditionally Forbidden
  double_t safe_duration_factor_;             // Conditionally Forbidden
  double_t safe_duration_offset_;             // Conditionally Forbidden
  uint8_t pickup_type_;                       // Conditionally Forbidden
  uint8_t drop_off_type_;                     // Conditionally Forbidden
  std::string pickup_booking_rule_id_;        // Optional
  std::string drop_off_booking_rule_id_;      // Optional
  uint8_t continuous_pickup_;                 // Conditionally Forbidden If stop_times.start_pickup_drop_off_window Or stop_times.end_pickup_drop_off_window Are Defined
  uint8_t continuous_drop_off_;               // Conditionally Forbidden If stop_times.start_pickup_drop_off_window Or stop_times.end_pickup_drop_off_window Are Defined
};

using td_stop_id_t = std::string;
using td_trip_id_t = std::string;
using td_stop_time_id_t = std::pair<td_stop_id_t, td_trip_id_t>;

using td_stop_time_map_t = hash_map<td_stop_time_id_t, std::unique_ptr<td_stop_time>>;

td_stop_time_map_t read_td_stop_times(std::string_view file_content);

}  // namespace nigiri::loader::gtfs_flex