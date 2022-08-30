#include "doctest/doctest.h"

#include "nigiri/loader/hrd/attribute.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/byte_sizes.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST_CASE("loader_hrd_attributes, parse_line") {
  constexpr auto const file_content = ",  0 260 10 Bus mit Fahrradanhänger";

  for (auto const& c : configs) {
    timetable tt;
    auto const attributes = parse_attributes(c, tt, file_content);
    auto const it = attributes.find(", ");
    CHECK(it != end(attributes));
    CHECK_EQ(tt.attributes_.at(it->second),
             attribute{.code_ = ", ", .text_ = "Bus mit Fahrradanhänger"});
  }
}

TEST_CASE("loader_hrd_attributes, parse_and_ignore_line") {
  constexpr auto const file_content = "ZZ 0 060 10 zusätzlicher Zug\n# ,  ,  ,";

  for (auto const& c : configs) {
    timetable tt;
    auto attributes = parse_attributes(c, tt, file_content);
    auto const it = attributes.find("ZZ");
    CHECK(it != end(attributes));
    CHECK(tt.attributes_.at(it->second) ==
          attribute{.code_ = "ZZ", .text_ = "zusätzlicher Zug"});
  }
}

TEST_CASE("loader_hrd_attributes, ignore_output_rules") {
  constexpr auto const file_content = "# ,  ,  ,";
  for (auto const& c : configs) {
    timetable tt;
    CHECK(parse_attributes(c, tt, file_content).empty());
  }
}
