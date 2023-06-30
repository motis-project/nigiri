#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/files.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;
using namespace date;

TEST(gtfs, calendar) {
  auto const calendar =
      read_calendar(example_files().get_file(kCalenderFile).data());

  EXPECT_EQ(2, calendar.size());

  auto const we_it = calendar.find("WE");
  ASSERT_NE(we_it, end(calendar));

  auto const wd_it = calendar.find("WD");
  ASSERT_NE(wd_it, end(calendar));

  EXPECT_TRUE(we_it->second.week_days_.test(0));
  EXPECT_FALSE(we_it->second.week_days_.test(1));
  EXPECT_FALSE(we_it->second.week_days_.test(2));
  EXPECT_FALSE(we_it->second.week_days_.test(3));
  EXPECT_FALSE(we_it->second.week_days_.test(4));
  EXPECT_FALSE(we_it->second.week_days_.test(5));
  EXPECT_TRUE(we_it->second.week_days_.test(6));

  EXPECT_FALSE(wd_it->second.week_days_.test(0));
  EXPECT_TRUE(wd_it->second.week_days_.test(1));
  EXPECT_TRUE(wd_it->second.week_days_.test(2));
  EXPECT_TRUE(wd_it->second.week_days_.test(3));
  EXPECT_TRUE(wd_it->second.week_days_.test(4));
  EXPECT_TRUE(wd_it->second.week_days_.test(5));
  EXPECT_FALSE(wd_it->second.week_days_.test(6));

  EXPECT_EQ(July / 1 / 2006, we_it->second.interval_.from_);
  EXPECT_EQ(August / 1 / 2006, we_it->second.interval_.to_);
  EXPECT_EQ(July / 1 / 2006, wd_it->second.interval_.from_);
  EXPECT_EQ(August / 1 / 2006, wd_it->second.interval_.to_);
}
