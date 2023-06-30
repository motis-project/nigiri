#include "gtest/gtest.h"

#include "nigiri/loader/hrd/stamm/category.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST(hrd, parse_categories) {
  constexpr auto const file_content = R"(Bsv  3 C 0  Bus       0 N Bus
Bus  5 C 0  Bus       0 N Bus
RB   3 C 0  RB        0 N Regionalbahn
s    4 C 0  S         0 N S-Bahn)";

  for (auto const& c : configs) {
    auto categories = parse_categories(c, file_content);
    ASSERT_EQ(4U, categories.size());

    auto const it = categories.find("s  ");
    ASSERT_NE(end(categories), it);
    EXPECT_EQ((category{.name_ = "s",
                        .long_name_ = "S",
                        .output_rule_ = 0,
                        .clasz_ = clasz::kMetro}),
              it->second);
  }
}
