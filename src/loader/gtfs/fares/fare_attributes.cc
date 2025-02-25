#include "nigiri/loader/gtfs/fares/fare_attributes.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/files.h"

namespace nigiri::loader::gtfs::fares {

std::vector<fare_attribute> read_fare_attributes(
    std::string_view file_content) {
  struct fare_attr_record {
    utl::csv_col<utl::cstr, UTL_NAME("fare_id")> fare_id_;
    utl::csv_col<double, UTL_NAME("price")> price_;
    utl::csv_col<utl::cstr, UTL_NAME("currency_type")> currency_type_;
    utl::csv_col<unsigned, UTL_NAME("payment_method")> payment_method_;
    utl::csv_col<unsigned, UTL_NAME("transfers")> transfers_;
    utl::csv_col<utl::cstr, UTL_NAME("agency_id")> agency_id_;
    utl::csv_col<utl::cstr, UTL_NAME("transfer_duration")> transfer_duration_;
  };

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Fare Attributes")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(file_content,
                                              progress_tracker->update_fn())} |
         utl::csv<fare_attr_record>() |
         utl::transform([](fare_attr_record const& r) {
           auto attr = fare_attribute{};
           attr.fare_id_ = r.fare_id_->view();
           attr.price_ = *r.price_;
           attr.currency_type_ = r.currency_type_->view();
           attr.payment_method_ = *r.payment_method_ == 1U
                                      ? payment_method::kBeforeBoarding
                                      : payment_method::kOnBoard;
           attr.transfer_duration_ =
               r.transfer_duration_->trim().empty()
                   ? std::nullopt
                   : std::optional{utl::parse<unsigned>(*r.transfer_duration_)};
           attr.agency_id_ = r.agency_id_->trim().empty()
                                 ? std::nullopt
                                 : std::optional{r.agency_id_->view()};
           switch (*r.transfers_) {
             case 0U: attr.transfers_ = transfers_type::kUnlimited; break;
             case 1U: attr.transfers_ = transfers_type::kNoTransfers; break;
             case 2U: attr.transfers_ = transfers_type::kOneTransfer; break;
             case 3U: attr.transfers_ = transfers_type::kTwoTransfers; break;
             default: attr.transfers_ = transfers_type::kUnlimited; break;
           }
           return attr;
         }) |
         utl::vec();
}

}  // namespace nigiri::loader::gtfs::fares