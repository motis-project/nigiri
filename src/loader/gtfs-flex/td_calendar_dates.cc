#include "nigiri/loader/gtfs-flex/td_calendar_dates.h"

#include <nigiri/common/cached_lookup.h>
#include <nigiri/loader/gtfs/parse_date.h>
#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/for_each.h>
#include <utl/progress_tracker.h>

namespace nigiri::loader::gtfs_flex {

td_calendar_date_map_t read_calendar_date(
    std::string_view file_content) {
  struct csv_td_calendar_date {
    utl::csv_col<utl::cstr, UTL_NAME("service_id")> id_;
    utl::csv_col<unsigned, UTL_NAME("date")> date_;
    utl::csv_col<unsigned, UTL_NAME("exception_type")> exception_type_;
  };

  auto services = td_calendar_date_map_t{};
  auto lookup_service = cached_lookup{services};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Calendar Date")
      .out_bounds(31.F, 33.F)
      .in_high(file_content.size());
  utl::line_range{
    utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
  | utl::csv<csv_td_calendar_date>()  //
  | utl::for_each([&](csv_td_calendar_date const& c) {
      lookup_service(c.id_->view())
          .emplace_back(calendar_date{
              .type_ = (*c.exception_type_ == 1 ? calendar_date::kAdd
                                                : calendar_date::kRemove),
              .day_ = gtfs::parse_date(*c.date_)});
    });
  return services;
}

}  // namespace nigiri::loader::gtfs