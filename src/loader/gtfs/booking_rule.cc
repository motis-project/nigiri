#include "nigiri/loader/gtfs/booking_rule.h"

#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/progress_tracker.h>

#include <nigiri/timetable.h>
#include <utl/pipes/vec.h>

namespace nigiri::loader::gtfs {





booking_rule_map_t read_booking_rule(std::string_view file_content) {
  auto const timer = scoped_timer{"gtfs.loader.booking_rules"};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Booking Rules")
      .out_bounds(0.F, 1.F) //TODO Werte??
      .in_high(file_content.size());

  struct csv_booking_rule {
    utl::csv_col<utl::cstr, UTL_NAME("booking_rule_id")> id_;
    utl::csv_col<uint8_t, UTL_NAME("booking_type")> type_; // 0 immediatly, 1 same-day booking, 2 x-days prior
    utl::csv_col<utl::cstr, UTL_NAME("info_url")> info_url_;
    utl::csv_col<utl::cstr, UTL_NAME("booking_url")> booking_url_;
    utl::csv_col<utl::cstr, UTL_NAME("message")> message_;
    utl::csv_col<utl::cstr, UTL_NAME("phone_number")> phone_number_;
    utl::csv_col<uint16_t, UTL_NAME("prior_notice_duration_min")> prior_notice_duration_min_;
    utl::csv_col<uint16_t, UTL_NAME("prior_notice_duration_max")> prior_notice_duration_max_;
    utl::csv_col<uint16_t, UTL_NAME("prior_notice_last_day")> prior_notice_last_day_;
  };

  utl::line_range{utl::make_buf_reader(file_content,
                                       progress_tracker->update_fn())}  //
  | utl::csv<csv_booking_rule>()  //
  | utl::transform([&](csv_booking_rule const& b) {
    return std::pair{
      b.id_->to_str(),
      std::make_unique<booking_rule>(booking_rule{
        .id_ = b.id_->to_str(),
        .type_ = b.type_.val(),
        .info_url_ = b.info_url_->to_str(),
        .message_ = b.message_->to_str(),
        .phone_number_ = b.phone_number_->to_str(),
        .prior_notice_duration_min_ = b.prior_notice_duration_min_.val(),
        .prior_notice_duration_max_ = b.prior_notice_duration_max_.val(),
        .prior_notice_last_day_ = b.prior_notice_last_day_.val(),
        .info_url_ = b.info_url_->to_str(),
        .booking_url_ = b.booking_url_->to_str()})};
  }) //
  | utl::to<booking_rule_map_t>();
}

}  // namespace nigiri::loader::gtfs
