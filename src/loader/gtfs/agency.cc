#include "nigiri/loader/gtfs/agency.h"

#include "date/tz.h"

#include "utl/get_or_create.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

agency_map_t read_agencies(timetable& tt,
                           tz_map& timezones,
                           std::string_view file_content) {
  struct agency {
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> id_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_name")> name_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_timezone")> tz_name_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Agencies")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
         | utl::csv<agency>()  //
         | utl::transform([&](agency const& a) {
             return cista::pair{
                 a.id_->to_str(),
                 tt.register_provider(
                     {a.id_->view(), a.name_->view(),
                      get_tz_idx(tt, timezones, a.tz_name_->trim().view())})};
           })  //
         | utl::to<agency_map_t>();
}

}  // namespace nigiri::loader::gtfs
