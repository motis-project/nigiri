#include "nigiri/loader/assistance.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"

namespace nigiri::loader {

void read_availability(timetable const& tt, std::string_view content) {
  struct assistance {
    utl::csv_col<utl::cstr, UTL_NAME("name")> name_;
    utl::csv_col<double, UTL_NAME("lat")> lat_;
    utl::csv_col<double, UTL_NAME("lng")> lng_;
    utl::csv_col<utl::cstr, UTL_NAME("time")> time_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Agencies")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
         | utl::csv<agency>()  //
         | utl::transform([&](agency const& a) {
             return std::pair{
                 a.id_->to_str(),
                 tt.register_provider(
                     {a.id_->view(), a.name_->view(),
                      get_tz_idx(tt, timezones, a.tz_name_->trim().view())})};
           })  //
         | utl::to<agency_map_t>();

  return {};
}

}  // namespace nigiri::loader