#include "doctest/doctest.h"

#include "nigiri/loader/hrd/category.h"
#include "nigiri/byte_sizes.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST_CASE("loader_hrd_categories, parse_line") {
  constexpr auto const file_content = R"(Bsv  3 C 0  Bus       0 N Bus
Bus  5 C 0  Bus       0 N Bus
RB   3 C 0  RB        0 N Regionalbahn
s    4 C 0  S         0 N S-Bahn)";

  for (auto const& c : configs) {
    auto categories = parse_categories(c, file_content);
    CHECK(categories.size() == 4U);

    auto const it = categories.find("s  ");
    CHECK(it != end(categories));
    CHECK((it->second ==
           category{.name_ = "s  ", .long_name_ = "S", .output_rule_ = 0}));
  }
}
