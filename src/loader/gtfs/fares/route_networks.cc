#include "nigiri/loader/gtfs/fares/route_networks.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs::fares {

std::vector<route_network> read_route_networks(std::string_view file_content) {
  struct route_network_record {
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("network_id")> network_id_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Route Networks")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                            progress_tracker->update_fn())} |
         utl::csv<route_network_record>() |
         utl::transform([](route_network_record const& r) {
           auto rn = route_network{};
           rn.route_id_ = r.route_id_->view();
           rn.network_id_ = r.network_id_->view();
           return rn;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares