#include "nigiri/loader/gtfs/fares/fare_rules.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

namespace nigiri::loader::gtfs::fares {

std::vector<fare_rule> read_fare_rules(std::string_view file_content) {
  if (file_content.empty()) {
    return {};
  }

  struct fare_rule_record {
    utl::csv_col<utl::cstr, UTL_NAME("fare_id")> fare_id_;
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("origin_id")> origin_id_;
    utl::csv_col<utl::cstr, UTL_NAME("destination_id")> destination_id_;
    utl::csv_col<utl::cstr, UTL_NAME("contains_id")> contains_id_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Fare Rules")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                              progress_tracker->update_fn())} |
         utl::csv<fare_rule_record>() |
         utl::transform([](fare_rule_record const& r) {
           auto rule = fare_rule{};
           rule.fare_id_ = r.fare_id_->view();
           rule.route_id_ = r.route_id_->trim().empty()
                                ? std::nullopt
                                : std::optional{r.route_id_->view()};
           rule.origin_id_ = r.origin_id_->trim().empty()
                                 ? std::nullopt
                                 : std::optional{r.origin_id_->view()};
           rule.destination_id_ =
               r.destination_id_->trim().empty()
                   ? std::nullopt
                   : std::optional{r.destination_id_->view()};
           rule.contains_id_ = r.contains_id_->trim().empty()
                                   ? std::nullopt
                                   : std::optional{r.contains_id_->view()};
           return rule;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares