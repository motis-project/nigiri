#include "nigiri/loader/gtfs/fares/stop_areas.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs::fares {

std::vector<stop_area> read_stop_areas(std::string_view file_content) {
  struct stop_area_record {
    utl::csv_col<utl::cstr, UTL_NAME("stop_id")> stop_id_;
    utl::csv_col<utl::cstr, UTL_NAME("area_id")> area_id_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Stop Areas")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                            progress_tracker->update_fn())} |
         utl::csv<stop_area_record>() |
         utl::transform([](stop_area_record const& r) {
           auto sa = stop_area{};
           sa.stop_id_ = r.stop_id_->view();
           sa.area_id_ = r.area_id_->view();
           return sa;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares