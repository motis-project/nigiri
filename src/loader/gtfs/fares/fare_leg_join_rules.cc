#include "nigiri/loader/gtfs/fares/fare_leg_join_rules.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs::fares {

std::vector<fare_leg_join_rule> read_fare_leg_join_rules(
    std::string_view file_content) {
  struct fare_leg_join_rule_record {
    utl::csv_col<utl::cstr, UTL_NAME("fare_leg_rule_id")> fare_leg_rule_id_;
    utl::csv_col<utl::cstr, UTL_NAME("from_leg_price_group_id")> from_leg_price_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("to_leg_price_group_id")> to_leg_price_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_leg_rule_sequence")> fare_leg_rule_sequence_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Fare Leg Join Rules")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                            progress_tracker->update_fn())} |
         utl::csv<fare_leg_join_rule_record>() |
         utl::transform([](fare_leg_join_rule_record const& r) {
           auto rule = fare_leg_join_rule{};
           rule.fare_leg_rule_id_ = r.fare_leg_rule_id_->view();
           
           // Handle optional fields
           if (!r.from_leg_price_group_id_->trim().empty()) {
             rule.from_leg_price_group_id_ = r.from_leg_price_group_id_->view();
           }
           
           if (!r.to_leg_price_group_id_->trim().empty()) {
             rule.to_leg_price_group_id_ = r.to_leg_price_group_id_->view();
           }
           
           if (!r.fare_leg_rule_sequence_->trim().empty()) {
             rule.fare_leg_rule_sequence_ = utl::parse<unsigned>(*r.fare_leg_rule_sequence_);
           }
           
           return rule;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares