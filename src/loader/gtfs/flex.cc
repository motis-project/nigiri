#include "nigiri/loader/gtfs/flex.h"

#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/pipes/transform.h"

#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

using namespace std::string_view_literals;

namespace nigiri::loader::gtfs {

constexpr auto const kBookingRulesFile = "booking_rules.txt"sv;

void parse_booking_rules(timetable& tt,
                         std::string_view file_content,
                         traffic_days_t const& traffic_days) {
  using utl::csv_col;

  struct booking_rule_record {
    csv_col<utl::cstr, UTL_NAME("booking_rule_id")> booking_rule_id_;
    csv_col<std::uint8_t, UTL_NAME("booking_type")> booking_type_;
    csv_col<std::optional<std::int32_t>, UTL_NAME("prior_notice_duration_min")>
        prior_notice_duration_min_;
    csv_col<std::optional<std::int32_t>, UTL_NAME("prior_notice_duration_max")>
        prior_notice_duration_max_;
    csv_col<std::optional<std::uint16_t>, UTL_NAME("prior_notice_last_day")>
        prior_notice_last_day_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("prior_notice_last_time")>
        prior_notice_last_time_;
    csv_col<std::optional<std::uint16_t>, UTL_NAME("prior_notice_start_day")>
        prior_notice_start_day_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("prior_notice_start_time")>
        prior_notice_start_time_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("prior_notice_service_id")>
        prior_notice_service_id_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("message")> message_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("pickup_message")>
        pickup_message_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("drop_off_message")>
        drop_off_message_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("phone_number")> phone_number_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("info_url")> info_url_;
    csv_col<std::optional<utl::cstr>, UTL_NAME("booking_url")> booking_url_;
  };

  auto const to_str = [&](auto const& col) {
    return col
        ->and_then([&](utl::cstr const& x) {
          return std::optional{tt.strings_.store(x.view())};
        })
        .value_or(string_idx_t::invalid());
  };

  utl::for_each_row<booking_rule_record>(
      file_content, [&](booking_rule_record const& r) {
        auto const get_type = [&]() -> booking_rule::booking_type {
          switch (*r.booking_type_) {
            case 0: return booking_rule::real_time{};
              
            case 1: {
              auto prior_notice = booking_rule::prior_notice{};
              if (r.prior_notice_duration_min_->has_value()) {
                prior_notice.prior_notice_duration_min_ =
                    i32_minutes{**r.prior_notice_duration_min_};
              }
              if (r.prior_notice_duration_max_->has_value()) {
                prior_notice.prior_notice_duration_max_ =
                    i32_minutes{**r.prior_notice_duration_max_};
              }
              return prior_notice;
            }

            case 2: {
              auto prior_day = booking_rule::prior_day{};
              if (r.prior_notice_last_day_->has_value()) {
                prior_day.prior_notice_last_day_ = **r.prior_notice_last_day_;
              }
              if (r.prior_notice_last_time_->has_value()) {
                prior_day.prior_notice_last_time_ =
                    hhmm_to_min(**r.prior_notice_last_time_);
              }
              if (r.prior_notice_start_day_->has_value()) {
                prior_day.prior_notice_start_day_ = **r.prior_notice_start_day_;
              }
              if (r.prior_notice_start_time_->has_value()) {
                prior_day.prior_notice_start_time_ =
                    hhmm_to_min(**r.prior_notice_start_time_);
              }
              if (r.prior_notice_service_id_->has_value()) {
                prior_day.prior_notice_service_id_ =
                    traffic_days.at((*r.prior_notice_service_id_)->view());
              }
            }
          }
          return booking_rule::real_time{};
        };

        tt.booking_rules_.emplace_back(
            booking_rule{.id_ = tt.strings_.store(r.booking_rule_id_->view()),
                         .type_ = get_type(),
                         .message_ = to_str(r.message_),
                         .pickup_message_ = to_str(r.pickup_message_),
                         .drop_off_message_ = to_str(r.drop_off_message_),
                         .phone_number_ = to_str(r.phone_number_),
                         .info_url_ = to_str(r.info_url_),
                         .booking_url_ = to_str(r.booking_url_)});
      });
}

void load_flex(timetable& tt,
               dir const& d,
               traffic_days_t const& traffic_days,
               locations_map const&) {
  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  parse_booking_rules(tt, load(kBookingRulesFile).data(), traffic_days);
}

}  // namespace nigiri::loader::gtfs