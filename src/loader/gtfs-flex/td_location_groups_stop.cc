#include "nigiri/loader/gtfs-flex/td_location_groups_stop.h"

#include <utl/get_or_create.h>
#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/pipes/vec.h>
#include <utl/progress_tracker.h>

namespace nigiri::loader::gtfs_flex {

td_location_group_stop_map_t read_td_location_group_stops(std::string_view file_content) {
  struct csv_td_location_group_stop {
    utl::csv_col<utl::cstr, UTL_NAME("location_group_id")> location_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse td_Location_Group_Stops")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  td_location_group_stop_map_t map{};
  utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
  | utl::csv<csv_td_location_group_stop>()  //
  | utl::transform([&](csv_td_location_group_stop const& l) {
    auto value = utl::get_or_create(map, l.location_group_id_->to_str(), [&](std::vector<std::string>){return std::vector<std::string>{};});
    value.push_back(l.stop_id_->to_str());
  });  //
  return map;
}

}  // namespace nigiri::loader::gtfs