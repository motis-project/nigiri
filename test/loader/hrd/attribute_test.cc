#include "gtest/gtest.h"

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/stamm/attribute.h"
#include "nigiri/byte_sizes.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST(hrd, parse_attributes_line) {
  constexpr auto const file_content = ",  0 260 10 Bus mit Fahrradanh\xE4nger#";

  for (auto const& c : configs) {
    timetable tt{};
    auto const attributes = parse_attributes(c, tt, file_content);
    auto const it = attributes.find(", ");
    ASSERT_NE(end(attributes), it);
    EXPECT_EQ((attribute{.code_ = ", ", .text_ = "Bus mit Fahrradanhänger"}),
              tt.attributes_.at(it->second));
  }
}

TEST(hrd, parse_attributes_and_ignore_line) {
  constexpr auto const file_content =
      "ZZ 0 060 10 zus\xE4tzlicher Zug#\n# ,  ,  ,";

  for (auto const& c : configs) {
    timetable tt{};
    auto attributes = parse_attributes(c, tt, file_content);
    auto const it = attributes.find("ZZ");
    ASSERT_NE(end(attributes), it);
    EXPECT_EQ((attribute{.code_ = "ZZ", .text_ = "zusätzlicher Zug"}),
              tt.attributes_.at(it->second));
  }
}

TEST(hrd, ignore_attributes_output_rules) {
  constexpr auto const file_content = "# ,  ,  ,";
  for (auto const& c : configs) {
    timetable tt{};
    EXPECT_TRUE(parse_attributes(c, tt, file_content).empty());
  }
}
