#include "doctest/doctest.h"

#include "nigiri/loader/hrd/attribute.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/byte_sizes.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST_CASE("loader_hrd_attributes, parse_line") {
  constexpr auto const file_content = ",  0 260 10 Bus mit Fahrradanhänger#";

  for (auto const& c : configs) {
    info_db db{"./test_db", 512_kB, info_db::init_type::CLEAR};
    CHECK(db.size<attribute>() == 0U);

    auto const attributes = parse_attributes(c, db, file_content);
    CHECK(attributes.size() == 1U);
    CHECK(db.size<attribute>() == 1U);

    auto const it = attributes.find(", ");
    CHECK(it != end(attributes));
    CHECK(db.get<attribute>(it->second) ==
          attribute{.code_ = ", ", .text_ = "Bus mit Fahrradanhänger"});
  }
}

TEST_CASE("loader_hrd_attributes, parse_and_ignore_line") {
  constexpr auto const file_content =
      "ZZ 0 060 10 zusätzlicher Zug#\n# ,  ,  ,";

  for (auto const& c : configs) {
    info_db db{"./test_db", 512_kB, info_db::init_type::CLEAR};
    CHECK(db.size<attribute>() == 0U);

    auto attributes = parse_attributes(c, db, file_content);
    CHECK(attributes.size() == 1U);
    CHECK(db.size<attribute>() == 1U);

    auto const it = attributes.find("ZZ");
    CHECK(it != end(attributes));
    CHECK(db.get<attribute>(it->second) ==
          attribute{.code_ = "ZZ", .text_ = "zusätzlicher Zug"});
  }
}

TEST_CASE("loader_hrd_attributes, ignore_output_rules") {
  constexpr auto const file_content = "# ,  ,  ,";
  info_db db{"./test_db", 512_kB, info_db::init_type::CLEAR};
  CHECK(db.size<attribute>() == 0U);
  for (auto const& c : configs) {
    CHECK(parse_attributes(c, db, file_content).empty());
  }
  CHECK(db.size<attribute>() == 0);
}
