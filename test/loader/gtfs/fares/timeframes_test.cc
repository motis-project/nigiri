#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/fares/timeframes.h"

namespace nigiri::loader::gtfs::fares {

TEST(fares, timeframes_parse_basic_fields) {
  auto const content = R"(timeframe_id,timeframe_name,timeframe_start_time,timeframe_end_time,timeframe_duration,timeframe_disable_after_purchase
time1,Morning Peak,07:00,09:30,9000,0
time2,Evening Peak,16:30,19:00,9000,1
time3,All Day,06:00,22:00,57600,0)";

  auto const result = read_timeframes(content);
  ASSERT_EQ(3, result.size());

  EXPECT_EQ("time1", result[0].timeframe_id_);
  EXPECT_EQ("Morning Peak", result[0].timeframe_name_.value());
  EXPECT_EQ(duration_t{7 * 60}, result[0].timeframe_start_time_.value());
  EXPECT_EQ(duration_t{9 * 60 + 30}, result[0].timeframe_end_time_.value());
  EXPECT_EQ(9000U, result[0].timeframe_duration_.value());
  EXPECT_FALSE(result[0].timeframe_disable_after_purchase_.value());
  
  EXPECT_EQ("time2", result[1].timeframe_id_);
  EXPECT_EQ("Evening Peak", result[1].timeframe_name_.value());
  EXPECT_EQ(duration_t{16 * 60 + 30}, result[1].timeframe_start_time_.value());
  EXPECT_EQ(duration_t{19 * 60}, result[1].timeframe_end_time_.value());
  EXPECT_EQ(9000U, result[1].timeframe_duration_.value());
  EXPECT_TRUE(result[1].timeframe_disable_after_purchase_.value());
  
  EXPECT_EQ("time3", result[2].timeframe_id_);
  EXPECT_EQ("All Day", result[2].timeframe_name_.value());
  EXPECT_EQ(duration_t{6 * 60}, result[2].timeframe_start_time_.value());
  EXPECT_EQ(duration_t{22 * 60}, result[2].timeframe_end_time_.value());
  EXPECT_EQ(57600U, result[2].timeframe_duration_.value());
  EXPECT_FALSE(result[2].timeframe_disable_after_purchase_.value());
}

TEST(fares, timeframes_parse_with_empty_optional_fields) {
  auto const content = R"(timeframe_id,timeframe_name,timeframe_start_time,timeframe_end_time,timeframe_duration,timeframe_disable_after_purchase
time1,,07:00,,,
time2,Evening Peak,,19:00,9000,
time3,,,,,1)";

  auto const result = read_timeframes(content);
  ASSERT_EQ(3, result.size());

  EXPECT_EQ("time1", result[0].timeframe_id_);
  EXPECT_FALSE(result[0].timeframe_name_.has_value());
  EXPECT_EQ(duration_t{7 * 60}, result[0].timeframe_start_time_.value());
  EXPECT_FALSE(result[0].timeframe_end_time_.has_value());
  EXPECT_FALSE(result[0].timeframe_duration_.has_value());
  EXPECT_FALSE(result[0].timeframe_disable_after_purchase_.has_value());
  
  EXPECT_EQ("time2", result[1].timeframe_id_);
  EXPECT_EQ("Evening Peak", result[1].timeframe_name_.value());
  EXPECT_FALSE(result[1].timeframe_start_time_.has_value());
  EXPECT_EQ(duration_t{19 * 60}, result[1].timeframe_end_time_.value());
  EXPECT_EQ(9000U, result[1].timeframe_duration_.value());
  EXPECT_FALSE(result[1].timeframe_disable_after_purchase_.has_value());
  
  EXPECT_EQ("time3", result[2].timeframe_id_);
  EXPECT_FALSE(result[2].timeframe_name_.has_value());
  EXPECT_FALSE(result[2].timeframe_start_time_.has_value());
  EXPECT_FALSE(result[2].timeframe_end_time_.has_value());
  EXPECT_FALSE(result[2].timeframe_duration_.has_value());
  EXPECT_TRUE(result[2].timeframe_disable_after_purchase_.value());
}

TEST(fares, timeframes_parse_with_non_standard_time_formats) {
  auto const content = R"(timeframe_id,timeframe_start_time,timeframe_end_time
time1,7:00,9:30
time2,16:30,19:00
time3,6:00,22:00)";

  auto const result = read_timeframes(content);
  ASSERT_EQ(3, result.size());

  EXPECT_EQ(duration_t{7 * 60}, result[0].timeframe_start_time_.value());
  EXPECT_EQ(duration_t{9 * 60 + 30}, result[0].timeframe_end_time_.value());
  
  EXPECT_EQ(duration_t{16 * 60 + 30}, result[1].timeframe_start_time_.value());
  EXPECT_EQ(duration_t{19 * 60}, result[1].timeframe_end_time_.value());
  
  EXPECT_EQ(duration_t{6 * 60}, result[2].timeframe_start_time_.value());
  EXPECT_EQ(duration_t{22 * 60}, result[2].timeframe_end_time_.value());
}

}  // namespace nigiri::loader::gtfs::fares