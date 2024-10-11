#include "nigiri/loader/gtfs-flex/td_trip.h"

#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/pipes/vec.h>
#include <utl/progress_tracker.h>

namespace nigiri::loader::gtfs_flex {

td_trip_map_t read_td_trips(std::string_view file_content) {
  struct csv_td_trip {
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("service_id")> service_id_;
    utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;
    utl::csv_col<utl::cstr, UTL_NAME("shape_id")> shape_id_;
    utl::csv_col<utl::cstr, UTL_NAME("trip_headsign")> trip_headsign_;
    utl::csv_col<utl::cstr, UTL_NAME("trip_short_name")> trip_short_name_;
    utl::csv_col<uint8_t, UTL_NAME("direction_id")> direction_id_;
    utl::csv_col<utl::cstr, UTL_NAME("block_id")> block_id_;
    utl::csv_col<uint8_t, UTL_NAME("wheelchair_accessible")> wheelchair_accessible_;
    utl::csv_col<uint8_t, UTL_NAME("bikes_allowed")> bikes_allowed_;
    utl::csv_col<utl::cstr, UTL_NAME("trip_note")> trip_note_;
    utl::csv_col<utl::cstr, UTL_NAME("route_direction")> route_direction_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse td_Agencies")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
  | utl::csv<csv_td_trip>()  //
  | utl::transform([&](csv_td_trip const& t) {
    return std::pair{
        t.trip_id_->to_str(),
        std::make_unique<td_trip>(td_trip{
          .route_id_ = t.route_id_->to_str(),
          .service_id_ = t.service_id_->to_str(),
          .shape_id_ = t.shape_id_->to_str(),
          .trip_headsign_ = t.trip_headsign_->to_str(),
          .trip_short_name_ = t.trip_short_name_->to_str(),
          .direction_id_ = t.direction_id_.val(),
          .block_id_ = t.block_id_->to_str(),
          .trip_note_ = t.trip_note_->to_str(),
          .route_direction_ = t.route_direction_->to_str(),
          .wheelchair_accessible_ = t.wheelchair_accessible_.val(),
          .bikes_allowed_ = t.bikes_allowed_.val(),
        })
  };
  })  //
  | utl::to<td_trip_map_t>();
}

}  // namespace nigiri::loader::gtfs