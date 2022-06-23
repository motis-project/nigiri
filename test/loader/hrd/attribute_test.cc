#include "doctest/doctest.h"

#include "nigiri/byte_sizes.h"
#include "nigiri/loader/hrd/attribute.h"
#include "nigiri/loader/hrd/parse_config.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST_CASE("loader_hrd_attributes, parse_line") {
  constexpr auto const file_content = ",  0 260 10 Bus mit Fahrradanhänger#";

  info_db db{"./test_db", 512_kB};

  for (auto const& c : configs) {
    auto attributes = parse_attributes(c, db, file_content);
    CHECK(attributes.size() == 1);

    auto const it = attributes.find(", ");
    CHECK(it != end(attributes));
    CHECK(db.get<attribute>(it->second) ==
          attribute{.code_ = ", ", .text_ = "Bus mit Fahrradanhänger"});
  }
}

TEST_CASE("loader_hrd_attributes, parse_and_ignore_line") {
  constexpr auto const file_content =
      "ZZ 0 060 10 zusätzlicher Zug#\n# ,  ,  ,";

  info_db db{"./test_db", 512_kB};

  for (auto const& c : configs) {
    auto attributes = parse_attributes(c, db, file_content);
    CHECK(attributes.size() == 1);

    auto const it = attributes.find("ZZ");
    CHECK(it != end(attributes));
    CHECK(db.get<attribute>(it->second) ==
          attribute{.code_ = "ZZ", .text_ = "zusätzlicher Zug"});
  }
}

TEST_CASE("loader_hrd_attributes, ignore_output_rules") {
  constexpr auto const file_content = "# ,  ,  ,";
  info_db db{"./test_db", 512_kB};
  for (auto const& c : configs) {
    CHECK(parse_attributes(c, db, file_content).empty());
  }
}
