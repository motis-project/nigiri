#include "nigiri/routing/placeholder_create_td_offset.h"

// namespace nigiri {
// hash_map<location_idx_t, std::vector<routing::td_offset>> create_td_offsets(
//     timetable& tt,
//     location_idx_t location_idx,
//     unixtime_t const& start_time,
//     std::function<duration_t(geo::latlng, geo::latlng)> get_duration,
//     uint8_t extra_days,
//     direction const search_dir) {
//   hash_map<location_idx_t, std::vector<routing::td_offset>> schedule{};
//
//   // First Step: Find Geometries containing location_idx
//   auto const start_coordinate = tt.locations_.coordinates_.at(location_idx);
//   auto const geometries = tt.lookup_td_stops(start_coordinate);
//
//   uint8_t counter = 0;
//   // floor<date::days>(time);
//
//   for (auto g_id : geometries) {
//     auto const geometry = tt.geometry_.at(g_id);
//     auto const trip_ids = tt.geometry_idx_to_trip_idxs_.at(g_id);
//
//     auto stops_in_geometry = tt.lookup_stops(geometry);
//     unixtime_t now = std::chrono::floor<std::chrono::minutes>(
//         std::chrono::system_clock::now());
//
//     for (auto const stop : stops_in_geometry) {
//       auto stop_coordinate = tt.locations_.coordinates_.at(stop);
//       auto duration = get_duration(start_coordinate, stop_coordinate);
//
//       for (auto t_id : trip_ids) {
//         auto const gt_id =
//             tt.geometry_trip_idxs_.at(geometry_trip_idx{t_id, g_id});
//         auto const stop_window = tt.window_times_.at(gt_id);
//         auto const pickup_booking_rules =
//             tt.booking_rules_.at(tt.pickup_booking_rules_.at(gt_id));
//         auto const dropoff_booking_rules =
//             tt.booking_rules_.at(tt.dropoff_booking_rules_.at(gt_id));
//
//         if (schedule.count(stop) == 0) {
//           if () {
//             continue;
//           }
//           std::vector<routing::td_offset> offset{};
//           offset.emplace_back(routing::td_offset{
//               std::max(pickup_booking_rules.prior_notice_duration_min_ +),
//           });
//           schedule.emplace(stop, offset);
//         } else {
//         }
//
//         for (; counter < extra_days; counter++) {
//         }
//         counter = 0;
//       }
//     }
//   }
//
//   std::vector<routing::td_offset> td_offsets;
//   routing::td_offset{.valid_from_ =, .duration_ =, t_id};
// }
// }  // namespace nigiri
