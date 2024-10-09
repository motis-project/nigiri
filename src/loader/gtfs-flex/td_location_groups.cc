#include <utl/get_or_create.h>

#include "nigiri/loader/gtfs-flex/td_location_group.h"

#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/pipes/vec.h>
#include <utl/progress_tracker.h>

namespace nigiri::loader::gtfs_flex {

td_location_group_map_t read_td_location_groups(std::string_view file_content) {
  struct csv_td_location_group {
    utl::csv_col<utl::cstr, UTL_NAME("location_group_id")> location_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_id")> location_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_group_name")> location_group_name_;
  };
  td_location_group_map_t map{};
  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse td_Location_Groups")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
  | utl::csv<csv_td_location_group>()  //
  | utl::transform([&](csv_td_location_group const& l) {
    auto value = utl::get_or_create(map, l.location_group_id_->to_str(), [&](td_location_group) {
      return td_location_group{};
    });
    if(value.location_group_name_.empty()) {
      value.location_group_name_ = l.location_group_name_->to_str();
    }
    value.location_ids_.push_back(l.location_group_id_->to_str());

  });
  return map;
  //

}

}  // namespace nigiri::loader::gtfs