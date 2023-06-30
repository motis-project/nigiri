#include "gtest/gtest.h"

#include "date/date.h"

#include "nigiri/loader/hrd/stamm/basic_info.h"
#include "nigiri/types.h"

using namespace nigiri::loader::hrd;

constexpr auto const file_content =
    "14.12.2014\n"
    "12.12.2015\n"
    "JF064 EVA_ABN~RIS Server~RIS OEV IMM~~J15~064_001 000000 END\n";

TEST(hrd, eckdaten_simple_interval) {
  using namespace date;

  auto const [a, b] = parse_interval(file_content);

  EXPECT_EQ(date::sys_days{2014_y / December / 14}, std::chrono::sys_days{a});
  EXPECT_EQ(date::sys_days{2015_y / December / 12}, std::chrono::sys_days{b});
}

TEST(hrd, schedule_name) {
  auto name = parse_schedule_name(file_content);
  EXPECT_EQ("JF064 EVA_ABN~RIS Server~RIS OEV IMM~~J15~064_001 000000 END",
            name);
}
