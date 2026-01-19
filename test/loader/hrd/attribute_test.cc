#include "gtest/gtest.h"

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/loader/hrd/stamm/attribute.h"
#include "nigiri/byte_sizes.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST(hrd, parse_attributes_line) {
  constexpr auto const file_content = ",  0 260 10 Bus mit Fahrradanh\xE4nger#";

  for (auto const& c : configs) {
    auto tt = timetable{};
    auto const attributes = parse_attributes(c, tt, file_content);
    auto const it = attributes.find(", ");
    ASSERT_NE(end(attributes), it);

    auto const attr = tt.attributes_.at(it->second);
    EXPECT_EQ(", ", attr.code_);
    EXPECT_EQ("Bus mit Fahrradanhänger",
              tt.get_default_translation(attr.text_));
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

    auto const attr = tt.attributes_.at(it->second);
    EXPECT_EQ("ZZ", attr.code_);
    EXPECT_EQ("zusätzlicher Zug", tt.get_default_translation(attr.text_));
  }
}

TEST(hrd, ignore_attributes_output_rules) {
  constexpr auto const file_content = "# ,  ,  ,";
  for (auto const& c : configs) {
    timetable tt{};
    EXPECT_TRUE(parse_attributes(c, tt, file_content).empty());
  }
}
