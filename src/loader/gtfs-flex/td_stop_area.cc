#include "nigiri/loader/gtfs-flex/td_stop_area.h"

#include <utl/get_or_create.h>
#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/pipes/vec.h>
#include <utl/progress_tracker.h>

namespace nigiri::loader::gtfs_flex {

td_stop_area_map_t read_td_stop_areas(std::string_view file_content) {
  struct csv_td_stop_area {
    utl::csv_col<utl::cstr, UTL_NAME("area_id")> area_id_;  // Required
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;  // Required
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse td_Stop_Areas")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  td_stop_area_map_t map{};
  utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
  | utl::csv<csv_td_stop_area>()  //
  | utl::transform([&](csv_td_stop_area const& s) {
    auto value = utl::get_or_create(map, s.area_id_->to_str(), [&](std::vector<std::string>){return std::vector<std::string>{};});
    value.push_back(s.stop_id_->to_str());
  });  //
  return map;
}

}  // namespace nigiri::loader::gtfs