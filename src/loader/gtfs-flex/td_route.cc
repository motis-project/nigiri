#include "nigiri/loader/gtfs-flex/td_route.h"

#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/pipes/vec.h>
#include <utl/progress_tracker.h>

namespace nigiri::loader::gtfs_flex {

td_route_map_t read_td_routes(std::string_view file_content) {
  struct csv_td_route {
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> agency_id_;
    utl::csv_col<utl::cstr, UTL_NAME("route_short_name")> route_short_name_;
    utl::csv_col<utl::cstr, UTL_NAME("route_long_name")> route_long_name_;
    utl::csv_col<utl::cstr, UTL_NAME("route_desc")> route_desc_;
    utl::csv_col<utl::cstr, UTL_NAME("route_type")> route_type_;
    utl::csv_col<utl::cstr, UTL_NAME("route_url")> route_url_;
    utl::csv_col<utl::cstr, UTL_NAME("route_color")> route_color_;
    utl::csv_col<utl::cstr, UTL_NAME("route_text_color")> route_text_color_;
    utl::csv_col<utl::cstr, UTL_NAME("route_sort_order")> route_sort_order_;
    utl::csv_col<utl::cstr, UTL_NAME("continuous_pickup")> continuous_pickup_;
    utl::csv_col<utl::cstr, UTL_NAME("continuous_drop_off")> continuous_drop_off_;
    utl::csv_col<utl::cstr, UTL_NAME("network_id")> network_id_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse td_Agencies")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
  | utl::csv<csv_td_route>()  //
  | utl::transform([&](csv_td_route const& r) {
    assert(!r.route_id_->view().empty());
    assert(!r.route_type_->view().empty());

    return std::pair{
        r.route_id_->to_str(),
        std::make_unique<td_route>{
          .agency_id_ = r.agency_id_->to_str(),
          .short_name_ = r.route_short_name_->to_str(),
          .long_name_ = r.route_long_name_->to_str(),
          .desc_ = r.route_desc_->to_str(),
          .type_ = strtoul(r.route_type_->c_str(), NULL, 10),
          .url_ = r.route_url_->to_str(),
          .color_ = r.route_color_->to_str(),
          .text_color_ = r.route_text_color_->to_str(),
          .sort_order_ = r.route_sort_order_->to_str(),
          .continuous_pickup_ = r.continuous_pickup_->to_str().empty() ? DEFAULT_CONTINOUS_STOPPING_PICKUP_DROPOFF : atoi(r.continuous_pickup_->c_str()),
          .continuous_drop_off_ = r.continuous_pickup_->to_str().empty() ? DEFAULT_CONTINOUS_STOPPING_PICKUP_DROPOFF : atoi(r.continuous_pickup_->c_str()),
          .network_id_ = r.network_id_->to_str()
        }
  };
  })  //
  | utl::to<td_route_map_t>();
}

}  // namespace nigiri::loader::gtfs_flex