#include "nigiri/loader/gtfs/calendar_date.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/parse_date.h"

namespace nigiri::loader::gtfs {

hash_map<std::string, std::vector<calendar_date>> read_calendar_date(
    std::string_view file_content) {
  struct entry {
    utl::csv_col<utl::cstr, UTL_NAME("service_id")> id_;
    utl::csv_col<unsigned, UTL_NAME("date")> date_;
    utl::csv_col<unsigned, UTL_NAME("exception_type")> exception_type_;
  };

  auto services = hash_map<std::string, std::vector<calendar_date>>{};
  auto cached_lookup = [&, prev_it =
                               std::optional<decltype(services)::iterator>{}](
                           auto&& key) mutable {
    if (!prev_it.has_value() || (*prev_it)->first != key) {
      if (auto const it = services.find(key); it == end(services)) {
        prev_it = services
                      .emplace(decltype(services)::key_type{key},
                               std::vector<calendar_date>{})
                      .first;
      } else {
        prev_it = it;
      }
    }
    return *prev_it;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Calendar Date")
      .out_bounds(34.F, 36.F)
      .in_high(file_content.size());
  utl::line_range{utl::buf_reader{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}}  //
      | utl::csv<entry>()  //
      | utl::for_each([&](entry const& e) {
          cached_lookup(e.id_->view())
              ->second.emplace_back(calendar_date{
                  .type_ = (*e.exception_type_ == 1 ? calendar_date::kAdd
                                                    : calendar_date::kRemove),
                  .day_ = parse_date(*e.date_)});
        });
  return services;
}

}  // namespace nigiri::loader::gtfs
