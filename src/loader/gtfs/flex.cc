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

using booking_rule_map_t = hash_map<std::string, flex_booking_rule_idx_t>;
using location_group_map_t = hash_map<std::string, flex_location_group_idx_t>;

string_idx_t to_str(timetable& tt, auto const& col) {
  return col
      ->and_then([&](utl::cstr const& x) {
        return std::optional{tt.strings_.store(x.view())};
      })
      .value_or(string_idx_t::invalid());
}

location_group_map_t parse_location_groups(timetable& tt,
                                           std::string_view file_content) {
  struct location_group_record {
    utl::csv_col<utl::cstr, UTL_NAME("location_group_id")> location_group_id_;
    utl::csv_col<std::optional<utl::cstr>, UTL_NAME("location_group_name")>
        location_group_name_;
  };

  auto map = location_group_map_t{};
  utl::for_each_row<location_group_record>(
      file_content, [&](location_group_record const& r) {
        tt.location_group_names_.emplace_back(
            to_str(tt, r.location_group_name_));
      });
  return map;
}

booking_rule_map_t parse_booking_rules(
    timetable& tt,
    std::string_view file_content,
    traffic_days_t const& traffic_days,
    hash_map<bitfield, bitfield_idx_t>& bitfield_indices) {
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

  auto map = booking_rule_map_t{};
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
                auto const& bitfield =
                    *traffic_days.at((*r.prior_notice_service_id_)->view());
                prior_day.prior_notice_bitfield_ = utl::get_or_create(
                    bitfield_indices, bitfield,
                    [&]() { return tt.register_bitfield(bitfield); });
              }
              return prior_day;
            }
          }
          return booking_rule::real_time{};
        };

        auto const idx = flex_booking_rule_idx_t{tt.booking_rules_.size()};
        tt.booking_rules_.emplace_back(
            booking_rule{.id_ = tt.strings_.store(r.booking_rule_id_->view()),
                         .type_ = get_type(),
                         .message_ = to_str(tt, r.message_),
                         .pickup_message_ = to_str(tt, r.pickup_message_),
                         .drop_off_message_ = to_str(tt, r.drop_off_message_),
                         .phone_number_ = to_str(tt, r.phone_number_),
                         .info_url_ = to_str(tt, r.info_url_),
                         .booking_url_ = to_str(tt, r.booking_url_)});
        map.emplace(r.booking_rule_id_->to_str(), idx);
      });
  return map;
}

void load_flex(timetable& tt,
               dir const& d,
               traffic_days_t const& traffic_days,
               hash_map<bitfield, bitfield_idx_t>& bitfield_indices,
               locations_map const&) {
  auto const load = [&](std::string_view file_name) -> file {
    return d.exists(file_name) ? d.get_file(file_name) : file{};
  };

  auto const booking_rules = parse_booking_rules(
      tt, load(kBookingRulesFile).data(), traffic_days, bitfield_indices);
}

}  // namespace nigiri::loader::gtfs