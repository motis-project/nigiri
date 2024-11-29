#include "nigiri/loader/gtfs/booking_rule.h"

#include <nigiri/loader/gtfs/parse_time.h>
#include <nigiri/loader/gtfs/services.h>
#include <nigiri/logging.h>
#include <utl/parser/buf_reader.h>
#include <utl/parser/csv_range.h>
#include <utl/parser/line_range.h>
#include <utl/pipes/transform.h>
#include <utl/pipes/vec.h>
#include <utl/progress_tracker.h>

#include <nigiri/timetable.h>

namespace nigiri::loader::gtfs {
booking_rule_map_t read_booking_rules(traffic_days_t const& services,
                                      timetable& tt,
                                      std::string_view file_content) {
  auto const kEmptyPair = std::pair<std::string, booking_rule_idx_t>{};

  struct csv_booking_rule {
    utl::csv_col<utl::cstr, UTL_NAME("booking_rule_id")> id_;
    utl::csv_col<std::uint16_t, UTL_NAME("booking_type")> type_;
    utl::csv_col<std::uint16_t, UTL_NAME("prior_notice_duration_min")>
        prior_notice_duration_min_;
    utl::csv_col<std::uint16_t, UTL_NAME("prior_notice_duration_max")>
        prior_notice_duration_max_;
    utl::csv_col<std::uint16_t, UTL_NAME("prior_notice_last_day")>
        prior_notice_last_day_;
    utl::csv_col<utl::cstr, UTL_NAME("prior_notice_last_time")>
        prior_notice_last_time_;
    utl::csv_col<utl::cstr, UTL_NAME("prior_notice_start_day")>
        prior_notice_start_day_;
    utl::csv_col<utl::cstr, UTL_NAME("prior_notice_start_time")>
        prior_notice_start_time_;
    utl::csv_col<utl::cstr, UTL_NAME("prior_notice_service_id")>
        prior_notice_service_id_;
  };
  auto const timer = scoped_timer{"read booking rules"};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Parse Booking Rules")
      .out_bounds(0.F, 1.F)
      .in_high(file_content.size());

  return utl::line_range{utl::make_buf_reader(
             file_content, progress_tracker->update_fn())}  //
         | utl::csv<csv_booking_rule>()  //
         |  //
         utl::transform([&](csv_booking_rule const& b) {
           // Checking GTFS-flex-specification requirements
           if (b.id_->empty()) {
             log(log_lvl::error, "loader.gtfs.booking_rule",
                 "booking_rule_id is empty");
             return kEmptyPair;
           }

           if (b.type_.val() != kRealTimeBooking &&
               b.type_.val() != kSameDayBooking &&
               b.type_.val() != kPriorDaysBooking) {
             log(log_lvl::error, "loader.gtfs.booking_rule",
                 "booking_type \"{}\" is unknown", b.type_.val());
             return kEmptyPair;
           }

           switch (b.type_.val()) {
             case kRealTimeBooking: break;
             case kSameDayBooking: {
               if (b.prior_notice_duration_min_.val() == 0) {
                 log(log_lvl::error, "loader.gtfs.booking_rule",
                     "prior_notice_duration_min cannot be 0");
                 return kEmptyPair;
               }
               break;
             }
             case kPriorDaysBooking: {
               if (b.prior_notice_last_day_.val() == 0) {
                 log(log_lvl::error, "loader.gtfs.booking_rule",
                     "prior_notice_duration_min cannot be 0");
                 return kEmptyPair;
               }
               if (b.prior_notice_last_time_->empty()) {
                 log(log_lvl::error, "loader.gtfs.booking_rule",
                     "prior_notice_last_time_ cannot be empty");
                 return kEmptyPair;
               }
               if (!b.prior_notice_start_day_->empty() &&
                   b.prior_notice_start_time_->empty()) {
                 log(log_lvl::error, "loader.gtfs.booking_rule",
                     "prior_notice_start_time_ cannot be empty if "
                     "prior_notice_start_day_ is not empty");
                 return kEmptyPair;
               }
               break;
             }
             default:
               log(log_lvl::error, "loader.gtfs.booking_rule",
                   "booking_type \"{}\": must be either 1, 2 or 3",
                   b.type_.val());
               return kEmptyPair;
           }

           auto traffic_days_it = services.end();
           auto error = false;
           if (!b.prior_notice_service_id_->empty()) {
             traffic_days_it =
                 services.find(b.prior_notice_service_id_->view());
             if (traffic_days_it == end(services)) {
               log(log_lvl::error, "loader.gtfs.booking_rule",
                   "booking_rule \"{}\": prior_notice_service_id \"{}\" not "
                   "found",
                   b.id_->view(), b.prior_notice_service_id_->view());
               error = true;
             }
           }

           return std::pair{
               b.id_->to_str(),
               tt.register_booking_rule(
                   b.id_->to_str(),
                   {.type_ = (uint8_t)b.type_.val(),
                    .prior_notice_duration_min_ =
                        b.prior_notice_duration_min_.val(),
                    .prior_notice_duration_max_ =
                        b.prior_notice_duration_max_.val(),
                    .prior_notice_last_day_ = b.prior_notice_last_day_.val(),
                    .prior_notice_last_time_ =
                        hhmm_to_min(*b.prior_notice_last_time_),
                    .prior_notice_start_day_ =
                        b.prior_notice_start_day_->empty()
                            ? static_cast<std::uint16_t>(0)
                            : static_cast<std::uint16_t>(
                                  strtoul(b.prior_notice_start_day_->c_str(),
                                          NULL, 10)),
                    .prior_notice_start_time_ =
                        hhmm_to_min(*b.prior_notice_start_time_),
                    .bitfield_idx_ =
                        b.prior_notice_service_id_->empty() || error
                            ? kInvalidBitfieldIdx
                            : tt.register_bitfield(*traffic_days_it->second)})};
         })  //
         | utl::to<booking_rule_map_t>();
}

}  // namespace nigiri::loader::gtfs