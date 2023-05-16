#include "gtest/gtest.h"

#include "nigiri/loader/hrd/stamm/bitfield.h"
#include "nigiri/byte_sizes.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

TEST(hrd, bitfields_hex_str_to_bitset) {
  EXPECT_ANY_THROW(hex_str_to_bitset("0"));
  EXPECT_ANY_THROW(hex_str_to_bitset("1"));
  EXPECT_ANY_THROW(hex_str_to_bitset("3"));
  EXPECT_ANY_THROW(hex_str_to_bitset("7"));
  EXPECT_EQ(bitfield(""), hex_str_to_bitset("F"));

  // 0x0653 = 0000 0110 0101 0011
  EXPECT_EQ(bitfield("0010100"), hex_str_to_bitset("0653"));

  // 0xC218 = 1100 0010 0001 1000
  EXPECT_EQ(bitfield("000010000"), hex_str_to_bitset("C218"));
}

TEST(hrd, parse_bitfields) {
  constexpr auto const file_content =
      R"(000001 C0200C
000002 C0100C)";

  auto const map =
      parse_bitfields(nigiri::loader::hrd::hrd_5_00_8, file_content);
  {
    auto const it = map.find(1);
    ASSERT_NE(it, end(map));
    EXPECT_EQ(it->second, bitfield{"100000000"});
  }
  {
    auto const it = map.find(2);
    ASSERT_NE(it, end(map));
    EXPECT_EQ(it->second, bitfield{"1000000000"});
  }
}
