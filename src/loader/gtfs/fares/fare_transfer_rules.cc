#include "nigiri/loader/gtfs/fares/fare_transfer_rules.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs::fares {

std::vector<fare_transfer_rule> read_fare_transfer_rules(
    std::string_view file_content) {
  struct fare_transfer_rule_record {
    utl::csv_col<utl::cstr, UTL_NAME("from_leg_group_id")> from_leg_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("to_leg_group_id")> to_leg_group_id_;
    utl::csv_col<unsigned, UTL_NAME("transfer_type")> transfer_type_;
    utl::csv_col<utl::cstr, UTL_NAME("transfer_fare_product_id")> transfer_fare_product_id_;
    utl::csv_col<utl::cstr, UTL_NAME("transfer_amount")> transfer_amount_;
    utl::csv_col<utl::cstr, UTL_NAME("transfer_count")> transfer_count_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Fare Transfer Rules")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                            progress_tracker->update_fn())} |
         utl::csv<fare_transfer_rule_record>() |
         utl::transform([](fare_transfer_rule_record const& r) {
           auto rule = fare_transfer_rule{};
           rule.from_leg_group_id_ = r.from_leg_group_id_->view();
           rule.to_leg_group_id_ = r.to_leg_group_id_->view();
           
           // Handle transfer type enum
           switch (*r.transfer_type_) {
             case 0U: rule.transfer_type_ = transfer_type::kDefaultCost; break;
             case 1U: rule.transfer_type_ = transfer_type::kNotPermitted; break;
             case 2U: rule.transfer_type_ = transfer_type::kPermittedWithCost; break;
             case 3U: rule.transfer_type_ = transfer_type::kPermittedWithFareProduct; break;
             default: rule.transfer_type_ = transfer_type::kDefaultCost; break;
           }
           
           // Handle optional fields
           if (!r.transfer_fare_product_id_->trim().empty()) {
             rule.transfer_fare_product_id_ = r.transfer_fare_product_id_->view();
           }
           
           if (!r.transfer_amount_->trim().empty()) {
             rule.transfer_amount_ = utl::parse<double>(*r.transfer_amount_);
           }
           
           if (!r.transfer_count_->trim().empty()) {
             rule.transfer_count_ = utl::parse<unsigned>(*r.transfer_count_);
           }
           
           return rule;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares