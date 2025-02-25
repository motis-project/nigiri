#include "nigiri/loader/gtfs/fares/fare_leg_rules.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs::fares {

std::vector<fare_leg_rule> read_fare_leg_rules(std::string_view file_content) {
  struct fare_leg_rule_record {
    utl::csv_col<utl::cstr, UTL_NAME("fare_leg_rule_id")> fare_leg_rule_id_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_product_id")> fare_product_id_;
    utl::csv_col<utl::cstr, UTL_NAME("leg_group_id")> leg_group_id_;
    utl::csv_col<utl::cstr, UTL_NAME("network_id")> network_id_;
    utl::csv_col<utl::cstr, UTL_NAME("from_area_id")> from_area_id_;
    utl::csv_col<utl::cstr, UTL_NAME("to_area_id")> to_area_id_;
    utl::csv_col<utl::cstr, UTL_NAME("route_id")> route_id_;
    utl::csv_col<utl::cstr, UTL_NAME("contains_area_id")> contains_area_id_;
    utl::csv_col<utl::cstr, UTL_NAME("contains_area_type")> contains_area_type_;
    utl::csv_col<utl::cstr, UTL_NAME("contains_route_id")> contains_route_id_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Fare Leg Rules")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                            progress_tracker->update_fn())} |
         utl::csv<fare_leg_rule_record>() |
         utl::transform([](fare_leg_rule_record const& r) {
           auto rule = fare_leg_rule{};
           rule.fare_leg_rule_id_ = r.fare_leg_rule_id_->view();
           rule.fare_product_id_ = r.fare_product_id_->view();
           
           // Handle optional fields
           if (!r.leg_group_id_->trim().empty()) {
             rule.leg_group_id_ = r.leg_group_id_->view();
           }
           
           if (!r.network_id_->trim().empty()) {
             rule.network_id_ = r.network_id_->view();
           }
           
           if (!r.from_area_id_->trim().empty()) {
             rule.from_area_id_ = r.from_area_id_->view();
           }
           
           if (!r.to_area_id_->trim().empty()) {
             rule.to_area_id_ = r.to_area_id_->view();
           }
           
           if (!r.route_id_->trim().empty()) {
             rule.route_id_ = r.route_id_->view();
           }
           
           if (!r.contains_area_id_->trim().empty()) {
             rule.contains_area_id_ = r.contains_area_id_->view();
           }
           
           if (!r.contains_area_type_->trim().empty()) {
             auto type_value = utl::parse<unsigned>(*r.contains_area_type_);
             switch (type_value) {
               case 0U: rule.contains_area_type_ = contains_area_type::kAny; break;
               case 1U: rule.contains_area_type_ = contains_area_type::kAll; break;
               default: rule.contains_area_type_ = contains_area_type::kAny; break;
             }
           }
           
           if (!r.contains_route_id_->trim().empty()) {
             rule.contains_route_id_ = r.contains_route_id_->view();
           }
           
           return rule;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares