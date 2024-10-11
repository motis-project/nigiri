#include "nigiri/loader/gtfs-flex/td_stop.h"

#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/pipes/vec.h>
#include <utl/progress_tracker.h>

namespace nigiri::loader::gtfs_flex {

td_stop_map_t read_td_stops(std::string_view file_content) {
  struct csv_td_stop {
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_code")> stop_code_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_name")> stop_name_;
    utl::csv_col<utl::cstr, UTL_NAME("tts_stop_name")> tts_stop_name_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_desc")> stop_desc_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_lat")> stop_lat_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_lon")> stop_lon_;
    utl::csv_col<utl::cstr, UTL_NAME("zone_id")> zone_id_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_url")> stop_url_;
    utl::csv_col<uint8_t, UTL_NAME("location_type")> location_type_;
    utl::csv_col<utl::cstr, UTL_NAME("parent_station")> parent_station_;
    utl::csv_col<utl::cstr, UTL_NAME("stop_timezone")> stop_timezone_;
    utl::csv_col<uint8_t, UTL_NAME("wheelchair_boarding")> wheelchair_boarding_;
    utl::csv_col<utl::cstr, UTL_NAME("level_id")> level_id_;
    utl::csv_col<utl::cstr, UTL_NAME("platform_code")> platform_code_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse td_Agencies")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
  | utl::csv<csv_td_stop>()  //
  | utl::transform([&](csv_td_stop const& s) {
    assert(!s.stop_id_->to_str().empty());
    const auto location_type = s.location_type_.val();
    if(location_type == LOCATION_STOP || location_type == LOCATION_STATION) {
        assert(!s.stop_name_->to_str().empty());
        assert(!s.stop_lat_->to_str().empty());
        assert(!s.stop_lon_->to_str().empty());
    }
    else if(location_type == LOCATION_ENTRANCE_EXIT) {
        assert(!s.stop_name_->to_str().empty());
        assert(!s.stop_lat_->to_str().empty());
        assert(!s.stop_lon_->to_str().empty());

        assert(!s.parent_station_->to_str().empty());
    }
    else {
      assert(location_type == LOCATION_GENERIC_NODE || location_type == LOCATION_BOARDING_AREA);
      assert(!s.parent_station_->to_str().empty());
    }

    return std::pair{
        s.stop_id_->to_str(),
        std::make_unique<td_stop>(td_stop{
          .code_ = s.stop_code_->to_str(),
          .name_ = s.stop_name_->to_str(),
          .tts_name_ = s.tts_stop_name_->to_str(),
          .desc_ = s.stop_desc_->to_str(),
          .lat_ = s.stop_lat_->to_str(),
          .lon_ = s.stop_lon_->to_str(),
          .zone_id_ = s.zone_id_->to_str(),
          .url_ = s.stop_url_->to_str(),
          .location_type_ = s.location_type_.val(),
          .parent_station_ = s.parent_station_->to_str(),
          .timezone_ = s.stop_timezone_->to_str(),
          .wheelchair_boarding_ = s.wheelchair_boarding_.val(),
          .level_id_ = s.level_id_->to_str(),
          .platform_code_ = s.platform_code_->to_str()
        })
  };
  })  //
  | utl::to<td_stop_map_t>();
}

}  // namespace nigiri::loader::gtfs