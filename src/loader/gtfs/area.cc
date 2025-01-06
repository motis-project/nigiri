#include "nigiri/loader/gtfs/area.h"

#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/for_each.h>
#include <utl/progress_tracker.h>

#include <nigiri/timetable.h>
#include <utl/get_or_create.h>

namespace nigiri::loader::gtfs {
area_map_t read_areas(timetable& tt,
                      locations_map const& locations_map,
                      std::string_view const stop_areas_content,
                      std::string_view const location_groups_content,
                      std::string_view const location_group_stops_content) {
  auto const timer = scoped_timer{"read areas"};
  auto stop_areas_map = read_areas(tt, locations_map, stop_areas_content);
  auto location_groups_map =
      read_areas(tt, locations_map, location_groups_content);
  auto location_group_stops_map =
      read_areas(tt, locations_map, location_group_stops_content);

  auto merged_map = area_map_t{};
  merged_map.reserve(stop_areas_map.size() + location_groups_map.size() +
                     location_group_stops_map.size());
  merged_map.insert(stop_areas_map.begin(), stop_areas_map.end());
  merged_map.insert(location_groups_map.begin(), location_groups_map.end());
  merged_map.insert(location_group_stops_map.begin(),
                    location_group_stops_map.end());

  return merged_map;
}

area_map_t read_areas(timetable& tt,
                      locations_map const& locations_map,
                      std::string_view const file_content) {
  struct csv_area {
    utl::csv_col<utl::cstr, UTL_NAME("area_id")> area_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_group_id")> location_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_id")> location_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse area")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  area_map_t area_map;

  hash_map<std::string, std::vector<location_idx_t>> area_id_to_location_ids;
  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_area>()  //
      | utl::for_each([&](csv_area const& a) {
          auto area_id = a.area_id_->view();
          if (area_id.empty()) {
            area_id = a.location_group_id_->view();
          }
          if (area_id.empty()) {
            log(log_lvl::error, "loader.gtfs.area",
                "area_id and location_group_id are empty!");
            return;
          }
          auto& location_ids = utl::get_or_create(
              area_id_to_location_ids, area_id,
              []() { return std::vector<location_idx_t>{}; });
          std::string id;
          if (!a.stop_id_->empty()) {
            id = a.stop_id_->to_str();
          } else if (!a.location_id_->empty()) {
            id = a.location_id_->to_str();
          } else {
            log(log_lvl::error, "loader.gtfs.area",
                "area {}: stop_id and location_id are empty", area_id);
            return;
          }

          auto const l_it = locations_map.find(id);
          if (l_it == locations_map.end()) {
            log(log_lvl::error, "loader.gtfs.area",
                "area {}: stop_id \"{}\" is unkown", area_id,
                a.location_id_->to_str());
            return;
          }
          location_ids.push_back(l_it->second);
        });
  for (auto const& a_to_l : area_id_to_location_ids) {
    auto const area_idx = tt.register_area(a_to_l.first, a_to_l.second);
    area_map.emplace(a_to_l.first, area_idx);
  }
  return area_map;
}

}  // namespace nigiri::loader::gtfs