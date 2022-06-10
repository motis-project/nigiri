#include "nigiri/loader/hrd/bitfield.h"

#include <bitset>

#include "utl/parser/arg_parser.h"

namespace nigiri::loader::hrd {

// source: https://stackoverflow.com/a/42201530/10794188
constexpr int const ascii_hex_to_int[] = {
    // clang-format off
  // ASCII
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1, -1, -1, -1, -1,
  -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,

  // non-ASCII
  -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
  -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
  -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
  -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
  -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
  -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
  -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
  -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
    // clang-format on
};

std::string hex_to_string(char c) {
  auto const i = static_cast<unsigned long long>(ascii_hex_to_int[c]);
  utl::verify(i != -1, "invalid bitfield char {}", static_cast<int>(c));
  return std::bitset<4>{i}.to_string();
}

template <typename T>
std::string hex_to_string(T const& char_collection) {
  std::string bit_str;
  for (auto const& c : char_collection) {
    bit_str.append(hex_to_string(c));
  }
  return bit_str;
}

bitfield hex_str_to_bitset(utl::cstr hex, int line_number) {
  auto const bit_str = hex_to_string(hex);
  auto const period_begin = bit_str.find("11");
  auto const period_end = bit_str.rfind("11");
  if (period_begin == std::string::npos || period_end == std::string::npos ||
      period_begin == period_end || period_end - period_begin <= 2) {
    throw utl::fail("invalid bitfield at line={}", line_number);
  }
  std::string bitstring(std::next(begin(bit_str), period_begin + 2),
                        std::next(begin(bit_str), period_end));
  std::reverse(begin(bitstring), end(bitstring));
  return bitfield{bitstring};
}

bitfield_map_t parse_bitfields(config const& c,
                               info_db& db,
                               std::string_view file_content) {
  bitfield_map_t bitfields;
  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, int line_number) {
        if (line.len == 0 || line.str[0] == '%') {
          return;
        } else if (line.len < 9) {
          throw utl::fail("bitfield line too short line={} length={}",
                          line_number, line.len);
        }

        auto const index = parse_verify<int>(line.substr(c.bf_.index_));
        auto b = hex_str_to_bitset(line.substr(c.bf_.value_), line_number);
        bitfields[index] = std::pair{b, db.add(b)};
      });

  // traffic day bitfield 0 = operates every day
  for (auto i = 0; i != bitfields[0].first.size(); ++i) {
    bitfields[0].first.set(i, true);
  }

  db.flush();

  return bitfields;
}

}  // namespace nigiri::loader::hrd
