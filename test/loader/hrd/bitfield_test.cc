#include "doctest/doctest.h"

#include "nigiri/loader/hrd/bitfield.h"
#include "nigiri/byte_sizes.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST_CASE("loader_hrd_bitfields.hex_str_to_bitset") {
  CHECK_THROWS(hex_str_to_bitset("0", 1));
  CHECK_THROWS(hex_str_to_bitset("1", 1));
  CHECK_THROWS(hex_str_to_bitset("3", 1));
  CHECK_THROWS(hex_str_to_bitset("7", 1));
  CHECK_THROWS(hex_str_to_bitset("F", 1));

  // 0x0653 = 0000 0110 0101 0011
  CHECK(bitfield("0010100") == hex_str_to_bitset("0653", 1));

  // 0xC218 = 1100 0010 0001 1000
  CHECK(bitfield("000010000") == hex_str_to_bitset("C218", 1));
}

TEST_CASE("loader_hrd_bitfields.parse_file") {
  constexpr auto const file_content =
      R"(000001 C0200C
000002 C0100C)";

  nigiri::info_db db{"./test_db", 512_kB, info_db::init_type::CLEAR};
  CHECK(db.size<bitfield>() == 0U);
  auto const map =
      parse_bitfields(nigiri::loader::hrd::hrd_5_00_8, db, file_content);
  CHECK(db.size<bitfield>() == 2U);
  {
    auto const it = map.find(1);
    REQUIRE(it != end(map));
    CHECK_EQ(it->second.first, bitfield{"100000000"});
    CHECK_EQ(db.get<bitfield>(it->second.second), bitfield{"100000000"});
  }
  {
    auto const it = map.find(2);
    REQUIRE(it != end(map));
    CHECK_EQ(it->second.first, bitfield{"1000000000"});
    CHECK_EQ(db.get<bitfield>(it->second.second), bitfield{"1000000000"});
  }
}
