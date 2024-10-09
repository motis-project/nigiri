#include "nigiri/loader/gtfs-flex/td_stop_time.h"

#include <string>
#include <nigiri/loader/gtfs/parse_time.h>
#include <nigiri/loader/hrd/util.h>
#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/pipes/vec.h>
#include <utl/progress_tracker.h>

namespace nigiri::loader::gtfs_flex {

td_stop_time_map_t read_td_stop_times(std::string_view file_content) {
  struct csv_td_stop_time {
    utl::csv_col<utl::cstr, UTL_NAME("trip_id")> trip_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_id")> location_id_;
    utl::csv_col<utl::cstr, UTL_NAME("location_group_id")> location_group_id_;
    utl::csv_col<uint16_t, UTL_NAME("stop_sequence")> stop_sequence_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_headsign")> stop_headsign_;
    utl::csv_col<utl::cstr, UTL_NAME("arrival_time")> arrival_time_;
    utl::csv_col<utl::cstr, UTL_NAME("departure_time")> departure_time_;
    utl::csv_col<utl::cstr, UTL_NAME("start_pickup_drop_off_window")> start_pickup_drop_off_window_;
    utl::csv_col<utl::cstr, UTL_NAME("end_pickup_drop_off_window")> end_pickup_drop_off_window_;
    utl::csv_col<double_t, UTL_NAME("shape_dist_traveled")> shape_dist_traveled_;
    utl::csv_col<utl::cstr, UTL_NAME("timepoint")> timepoint_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_note")> stop_note_;
    utl::csv_col<utl::cstr, UTL_NAME("mean_duration_factor")> mean_duration_factor_;
    utl::csv_col<double_t, UTL_NAME("mean_duration_offset")> mean_duration_offset_;
    utl::csv_col<utl::cstr, UTL_NAME("safe_duration_factor")> safe_duration_factor_;
    utl::csv_col<double_t, UTL_NAME("safe_duration_offset")> safe_duration_offset_;
    utl::csv_col<uint8_t, UTL_NAME("pickup_type")> pickup_type_;
    utl::csv_col<uint8_t, UTL_NAME("drop_off_type")> drop_off_type_;
    utl::csv_col<utl::cstr, UTL_NAME("pickup_booking_rule_id")> pickup_booking_rule_id_;
    utl::csv_col<utl::cstr, UTL_NAME("drop_off_booking_rule_id")> drop_off_booking_rule_id_;
    utl::csv_col<utl::cstr, UTL_NAME("continuous_pickup")> continuous_pickup_;
    utl::csv_col<utl::cstr, UTL_NAME("continuous_drop_off")> continuous_drop_off_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse td_Agencies")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
  | utl::csv<csv_td_stop_time>()  //
  | utl::transform([&](csv_td_stop_time const& s) {
    assert(!s.trip_id_->empty());
    assert(!s.stop_id_->empty() || !s.location_id_->empty() || !s.location_group_id_->empty());

    std::string final_id = "";
    if(!s.stop_id_->empty()) {
      final_id = s.stop_id_->to_str();
    }
    else if(!s.location_id_->empty()) {
      final_id = s.location_id_->to_str();
    }
    else if(!s.location_group_id_->empty()) {
      final_id = s.location_group_id_->to_str();
    }

    return std::pair{
        td_stop_time_id_t{final_id, s.trip_id_->to_str()},
        std::make_unique<td_stop_time>{
            .stop_sequence_ = s.stop_sequence_.val(),
            .stop_headsign_ = s.stop_headsign_->to_str(),
            .arrival_time_ = gtfs::hhmm_to_min(*s.arrival_time_),
            .departure_time_ = gtfs::hhmm_to_min(*s.departure_time_),
            .start_pickup_drop_off_window_ = gtfs::hhmm_to_min(*s.start_pickup_drop_off_window_),
            .end_pickup_drop_off_window_ = gtfs::hhmm_to_min(*s.end_pickup_drop_off_window_),
            .shape_dist_traveled_ = s.shape_dist_traveled_.val(),
            .timepoint_ = s.timepoint_->view().empty() ? DEFAULT_TIMEPOINT : atoi(s.timepoint_->c_str()),
            .stop_note_ = s.stop_note_->view(),
            .mean_duration_factor_ = s.mean_duration_factor_->view().empty() ? DEFAULT_FACTOR : std::stod(s.mean_duration_factor_->c_str()),
            .mean_duration_offset_ = s.mean_duration_offset_.val(),
            .safe_duration_factor_ = s.safe_duration_factor_->view().empty() ? DEFAULT_FACTOR : std::stod(s.safe_duration_factor_->c_str()),
            .safe_duration_offset_ = s.safe_duration_offset_.val(),
            .pickup_type_ = s.pickup_type_.val(),
            .drop_off_type_ = s.drop_off_type_.val(),
            .pickup_booking_rule_id_ = s.pickup_booking_rule_id_->to_str(),
            .drop_off_booking_rule_id_ = s.drop_off_booking_rule_id_->to_str(),
            .continuous_pickup_ = s.continuous_pickup_->to_str().empty() ? DEFAULT_CONTINOUS_STOPPING_PICKUP_DROPOFF : atoi(s.continuous_pickup_->c_str()),
            .continuous_drop_off_ = s.continuous_drop_off_->to_str().empty() ? DEFAULT_CONTINOUS_STOPPING_PICKUP_DROPOFF : atoi(s.continuous_drop_off_->c_str())
        }
  };
  })  //
  | utl::to<td_stop_time_map_t>();
}

}  // namespace nigiri::loader::gtfs