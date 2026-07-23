#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/feed_info.h"

using namespace date;

constexpr auto const kFeedInfo =
    R"(feed_publisher_name,feed_publisher_url,feed_lang,feed_start_date,feed_end_date,feed_version
City of Moose Jaw,https://moosejaw.ca,EN,20230101,20240331,1)";

TEST(gtfs, read_feed_info) {
  auto const feed_info = nigiri::loader::gtfs::read_feed_info(kFeedInfo);
  ASSERT_TRUE(feed_info.feed_end_date_.has_value());
  EXPECT_EQ(date::sys_days{2024_y / March / 31}, *feed_info.feed_end_date_);
  EXPECT_EQ("EN", feed_info.feed_lang_);
}

TEST(gtfs, read_feed_info_default_lang_fallback) {
  constexpr auto const kFeedInfoNoFeedLang =
      R"(feed_publisher_name,feed_publisher_url,default_lang,feed_version
City of Moose Jaw,https://moosejaw.ca,fr,1)";
  auto const feed_info =
      nigiri::loader::gtfs::read_feed_info(kFeedInfoNoFeedLang);
  EXPECT_EQ("fr", feed_info.feed_lang_);
}

TEST(gtfs, read_feed_info_lang_default) {
  constexpr auto const kFeedInfoNoLang =
      R"(feed_publisher_name,feed_publisher_url,feed_version
City of Moose Jaw,https://moosejaw.ca,1)";
  auto const feed_info = nigiri::loader::gtfs::read_feed_info(kFeedInfoNoLang);
  EXPECT_EQ("en", feed_info.feed_lang_);
}