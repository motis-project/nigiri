#include "nigiri/loader/assistance.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

namespace nigiri::loader {

void read_availability(timetable const& tt, std::string_view file_content) {
  struct assistance {
    utl::csv_col<utl::cstr, UTL_NAME("name")> name_;
    utl::csv_col<double, UTL_NAME("lat")> lat_;
    utl::csv_col<double, UTL_NAME("lng")> lng_;
    utl::csv_col<utl::cstr, UTL_NAME("time")> time_;
  };

  auto const a = assistance{};
  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Assistance")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
         | utl::csv<assistance>()  //
         | utl::for_each([&](assistance const& a) {

           });
}

}  // namespace nigiri::loader