#include "nigiri/loader/gtfs/area.h"

#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/for_each.h>
#include <utl/pipes/transform.h>
#include <utl/progress_tracker.h>

#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {
area_map_t read_areas(const source_idx_t src,
                      timetable& tt,
                      const locations_map& location_id_to_idx,
                      const location_geojson_map_t& geojson_id_to_idx,
                      std::string_view stop_areas_content,
                      std::string_view location_groups_content,
                      std::string_view location_group_stops_content) {
  auto stop_areas_map = read_areas(src, tt, location_id_to_idx,
                                   geojson_id_to_idx, stop_areas_content);
  auto location_groups_map = read_areas(
      src, tt, location_id_to_idx, geojson_id_to_idx, location_groups_content);
  auto location_group_stops_map =
      read_areas(src, tt, location_id_to_idx, geojson_id_to_idx,
                 location_group_stops_content);

  auto merged_map = area_map_t{};
  merged_map.reserve(stop_areas_map.size() + location_groups_map.size() +
                     location_group_stops_map.size());
  merged_map.insert(stop_areas_map.begin(), stop_areas_map.end());
  merged_map.insert(location_groups_map.begin(), location_groups_map.end());
  merged_map.insert(location_group_stops_map.begin(),
                    location_group_stops_map.end());

  return merged_map;
}

area_map_t read_areas(source_idx_t src,
                      timetable& tt,
                      const locations_map& location_id_to_idx,
                      const location_geojson_map_t& geojson_id_to_idx,
                      std::string_view file_content) {
  struct csv_area {
    utl::csv_col<utl::cstr, UTL_NAME("area_id")> area_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_group_id")> location_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_id")> location_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Agencies")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  area_map_t area_map;

  hash_map<std::string, std::vector<std::string>> area_id_to_location_ids;

  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_area>()  //
      | utl::for_each([&](csv_area const& a) {
          // TODO assert?

          auto area_id = a.area_id_->view();
          if (a.area_id_->empty()) {
            area_id = a.location_group_id_->view();
          }

          auto location_ids =
              utl::get_or_create(area_id_to_location_ids, area_id,
                                 []() { return std::vector<std::string>{}; });
          if (!a.location_id_->empty()) {
            location_ids.push_back(a.location_id_->to_str());
          } else if (!a.stop_id_->empty()) {
            location_ids.push_back(a.stop_id_->to_str());
          } else {
            // TODO error log
          }
        });
  for (auto pair : area_id_to_location_ids) {
    const auto area_idx = tt.areas_.register_area(
        src, pair.first, pair.second, location_id_to_idx, geojson_id_to_idx);
    area_map.emplace(pair.first, area_idx);
  }
}

}  // namespace nigiri::loader::gtfs