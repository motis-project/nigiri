#include <nigiri/loader/gtfs/booking_rule.h>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/booking_rule.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/parse_date.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

TEST(gtfs, booking_rule) {
  timetable tt;

  auto calendar =
      read_calendar(example_files().get_file(kBookingRuleCalendarFile).data());
  auto calendar_dates = read_calendar_date(
      example_files().get_file(kBookingRuleCalendarDatesFile).data());
  auto services =
      merge_traffic_days(tt.internal_interval_days(), calendar, calendar_dates);

  auto const booking_rules = read_booking_rules(
      services, tt, example_files().get_file(kBookingRulesFile).data());

  // Real-Time-Booking
  auto const assert_realtime_booking = [&](std::string const& id) {
    ASSERT_NO_THROW({
      auto const booking_idx = booking_rules.find(id);
      ASSERT_NE(booking_idx, end(booking_rules));
      auto const booking_rule = tt.booking_rules_.at(booking_idx->second);

      EXPECT_EQ(booking_rule.type_, Booking_type::kRealTimeBooking);
      EXPECT_EQ(booking_rule.prior_notice_duration_min_, 0);
      EXPECT_EQ(booking_rule.prior_notice_duration_max_, 0);
      EXPECT_EQ(booking_rule.prior_notice_last_day_, 0);
      EXPECT_EQ(booking_rule.prior_notice_last_time_, kInterpolate);
      EXPECT_EQ(booking_rule.prior_notice_start_day_, 0);
      EXPECT_EQ(booking_rule.prior_notice_start_time_, kInterpolate);
      EXPECT_EQ(booking_rule.bitfield_idx_, kInvalidBitfieldIdx);
    });
  };

  assert_realtime_booking("1");
  assert_realtime_booking("2");

  // Same-Day-Booking
  ASSERT_NO_THROW({
    auto const booking_idx = booking_rules.at("3");
    auto const booking_rule = tt.booking_rules_.at(booking_idx);

    EXPECT_EQ(booking_rule.type_, Booking_type::kSameDayBooking);
    EXPECT_EQ(booking_rule.prior_notice_duration_min_, 5);
    EXPECT_EQ(booking_rule.prior_notice_duration_max_, 0);
    EXPECT_EQ(booking_rule.prior_notice_last_day_, 0);
    EXPECT_EQ(booking_rule.prior_notice_last_time_, kInterpolate);
    EXPECT_EQ(booking_rule.prior_notice_start_day_, 0);
    EXPECT_EQ(booking_rule.prior_notice_start_time_, kInterpolate);
    EXPECT_EQ(booking_rule.bitfield_idx_, kInvalidBitfieldIdx);
  });

  ASSERT_NO_THROW({
    auto const booking_idx = booking_rules.at("4");
    auto const booking_rule = tt.booking_rules_.at(booking_idx);

    EXPECT_EQ(booking_rule.type_, Booking_type::kSameDayBooking);
    EXPECT_EQ(booking_rule.prior_notice_duration_min_, 15);
    EXPECT_EQ(booking_rule.prior_notice_duration_max_, 1440);
    EXPECT_EQ(booking_rule.prior_notice_last_day_, 0);
    EXPECT_EQ(booking_rule.prior_notice_last_time_, kInterpolate);
    EXPECT_EQ(booking_rule.prior_notice_start_day_, 0);
    EXPECT_EQ(booking_rule.prior_notice_start_time_, kInterpolate);
    EXPECT_EQ(booking_rule.bitfield_idx_, kInvalidBitfieldIdx);
  });

  ASSERT_NO_THROW({
    auto const booking_idx = booking_rules.at("5");
    auto const booking_rule = tt.booking_rules_.at(booking_idx);

    EXPECT_EQ(booking_rule.type_, Booking_type::kSameDayBooking);
    EXPECT_EQ(booking_rule.prior_notice_duration_min_, 30);
    EXPECT_EQ(booking_rule.prior_notice_duration_max_, 10080);
    EXPECT_EQ(booking_rule.prior_notice_last_day_, 0);
    EXPECT_EQ(booking_rule.prior_notice_last_time_, kInterpolate);
    EXPECT_EQ(booking_rule.prior_notice_start_day_, 0);
    EXPECT_EQ(booking_rule.prior_notice_start_time_, kInterpolate);
    EXPECT_EQ(booking_rule.bitfield_idx_, kInvalidBitfieldIdx);
  });

  // Prior-Days-Booking
  ASSERT_NO_THROW({
    auto const booking_idx = booking_rules.at("7");
    auto const booking_rule = tt.booking_rules_.at(booking_idx);

    EXPECT_EQ(booking_rule.type_, Booking_type::kPriorDaysBooking);
    EXPECT_EQ(booking_rule.prior_notice_duration_min_, 0);
    EXPECT_EQ(booking_rule.prior_notice_duration_max_, 0);
    EXPECT_EQ(booking_rule.prior_notice_last_day_, 1);
    EXPECT_EQ(booking_rule.prior_notice_last_time_, hhmm_to_min("12:00:00"));
    EXPECT_EQ(booking_rule.prior_notice_start_day_, 0);
    EXPECT_EQ(booking_rule.prior_notice_start_time_, kInterpolate);
    EXPECT_EQ(booking_rule.bitfield_idx_, kInvalidBitfieldIdx);
  });

  ASSERT_NO_THROW({
    auto const booking_idx = booking_rules.at("8");
    auto const booking_rule = tt.booking_rules_.at(booking_idx);

    EXPECT_EQ(booking_rule.type_, Booking_type::kPriorDaysBooking);
    EXPECT_EQ(booking_rule.prior_notice_duration_min_, 0);
    EXPECT_EQ(booking_rule.prior_notice_duration_max_, 0);
    EXPECT_EQ(booking_rule.prior_notice_last_day_, 3);
    EXPECT_EQ(booking_rule.prior_notice_last_time_, hhmm_to_min("18:00:00"));
    EXPECT_EQ(booking_rule.prior_notice_start_day_, 7);
    EXPECT_EQ(booking_rule.prior_notice_start_time_, hhmm_to_min("18:00:00"));
    EXPECT_EQ(booking_rule.bitfield_idx_, kInvalidBitfieldIdx);
  });

  ASSERT_NO_THROW({
    auto const booking_idx = booking_rules.at("9");
    auto const booking_rule = tt.booking_rules_.at(booking_idx);

    EXPECT_EQ(booking_rule.type_, Booking_type::kPriorDaysBooking);
    EXPECT_EQ(booking_rule.prior_notice_duration_min_, 0);
    EXPECT_EQ(booking_rule.prior_notice_duration_max_, 0);
    EXPECT_EQ(booking_rule.prior_notice_last_day_, 7);
    EXPECT_EQ(booking_rule.prior_notice_last_time_, hhmm_to_min("00:00:00"));
    EXPECT_EQ(booking_rule.prior_notice_start_day_, 30);
    EXPECT_EQ(booking_rule.prior_notice_start_time_, hhmm_to_min("08:00:00"));
    EXPECT_NE(booking_rule.bitfield_idx_, kInvalidBitfieldIdx);
  });
}
