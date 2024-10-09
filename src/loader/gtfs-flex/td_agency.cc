#include "nigiri/loader/gtfs-flex/td_agency.h"

#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/pipes/vec.h>
#include <utl/progress_tracker.h>

namespace nigiri::loader::gtfs_flex {

td_agency_map_t read_td_agencies(std::string_view file_content) {
  struct csv_td_agency {
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> id_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_name")> name_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_timezone")> tz_name_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_lang")> language_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_phone")> phone_number_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_url")> url_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse td_Agencies")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());
  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
  | utl::csv<csv_td_agency>()  //
  | utl::transform([&](csv_td_agency const& a) {
    return std::pair{
        a.id_->to_str(),
        std::make_unique<td_agency>{
          .name_ = a.name_->to_str(),
          .tz_name_ = a.tz_name_->to_str(),
          .language_ = a.language_->to_str(),
          .phone_number_ = a.phone_number_->to_str(),
          .url_ = a.url_->to_str()
        }
  };
  })  //
  | utl::to<td_agency_map_t>();
}

}  // namespace nigiri::loader::gtfs