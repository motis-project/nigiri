#include "nigiri/loader/gtfs/fares/rider_categories.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs::fares {

std::vector<rider_category> read_rider_categories(std::string_view file_content) {
  struct rider_category_record {
    utl::csv_col<utl::cstr, UTL_NAME("rider_category_id")> rider_category_id_;
    utl::csv_col<utl::cstr, UTL_NAME("rider_category_name")> rider_category_name_;
    utl::csv_col<utl::cstr, UTL_NAME("eligible_for_fare_product_id")> eligible_for_fare_product_id_;
    utl::csv_col<utl::cstr, UTL_NAME("min_age")> min_age_;
    utl::csv_col<utl::cstr, UTL_NAME("max_age")> max_age_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Rider Categories")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                            progress_tracker->update_fn())} |
         utl::csv<rider_category_record>() |
         utl::transform([](rider_category_record const& r) {
           auto rc = rider_category{};
           rc.rider_category_id_ = r.rider_category_id_->view();
           
           // Handle optional fields
           if (!r.rider_category_name_->trim().empty()) {
             rc.rider_category_name_ = r.rider_category_name_->view();
           }
           
           if (!r.eligible_for_fare_product_id_->trim().empty()) {
             rc.eligible_for_fare_product_id_ = r.eligible_for_fare_product_id_->view();
           }
           
           if (!r.min_age_->trim().empty()) {
             rc.min_age_ = utl::parse<unsigned>(*r.min_age_);
           }
           
           if (!r.max_age_->trim().empty()) {
             rc.max_age_ = utl::parse<unsigned>(*r.max_age_);
           }
           
           return rc;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares