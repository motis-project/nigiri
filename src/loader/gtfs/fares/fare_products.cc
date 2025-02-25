#include "nigiri/loader/gtfs/fares/fare_products.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs::fares {

std::vector<fare_product> read_fare_products(std::string_view file_content) {
  struct fare_product_record {
    utl::csv_col<utl::cstr, UTL_NAME("fare_product_id")> fare_product_id_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_product_name")> fare_product_name_;
    utl::csv_col<double, UTL_NAME("amount")> amount_;
    utl::csv_col<utl::cstr, UTL_NAME("currency")> currency_;
    utl::csv_col<utl::cstr, UTL_NAME("rider_category_id")> rider_category_id_;
    utl::csv_col<utl::cstr, UTL_NAME("timeframe_id")> timeframe_id_;
    utl::csv_col<utl::cstr, UTL_NAME("fare_media_id")> fare_media_id_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Fare Products")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                            progress_tracker->update_fn())} |
         utl::csv<fare_product_record>() |
         utl::transform([](fare_product_record const& r) {
           auto product = fare_product{};
           product.fare_product_id_ = r.fare_product_id_->view();
           product.amount_ = *r.amount_;
           product.currency_ = r.currency_->view();
           
           // Handle optional fields
           if (!r.fare_product_name_->trim().empty()) {
             product.fare_product_name_ = r.fare_product_name_->view();
           }
           
           if (!r.rider_category_id_->trim().empty()) {
             product.rider_category_id_ = r.rider_category_id_->view();
           }
           
           if (!r.timeframe_id_->trim().empty()) {
             product.timeframe_id_ = r.timeframe_id_->view();
           }
           
           if (!r.fare_media_id_->trim().empty()) {
             product.fare_media_id_ = r.fare_media_id_->view();
           }
           
           return product;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares