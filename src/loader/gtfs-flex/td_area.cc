#include "nigiri/loader/gtfs-flex/td_area.h"

#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/pipes/vec.h>
#include <utl/progress_tracker.h>

namespace nigiri::loader::gtfs_flex {

td_area_map_t read_td_areas(std::string_view file_content) {
  struct csv_td_area {
    utl::csv_col<utl::cstr, UTL_NAME("area_id")> area_id_;      // Required
    utl::csv_col<utl::cstr, UTL_NAME("area_name")> area_name_;  // Optional
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse td_Areas")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
  | utl::csv<csv_td_area>()  //
  | utl::transform([&](csv_td_area const& a) {
    return std::pair{
        a.area_id_->to_str(),
        a.area_name_->to_str()
  };
  })  //
  | utl::to<td_area_map_t>();
}

}  // namespace nigiri::loader::gtfs