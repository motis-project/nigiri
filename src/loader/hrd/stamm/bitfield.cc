#include "nigiri/loader/hrd/stamm/bitfield.h"

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

bitfield hex_str_to_bitset(utl::cstr char_collection) {
  bitfield b;

  auto consumed = 0U;
  bool started = false;
  bool prev_was_set = false;
  auto consume_bit = [&, i = 0U](std::bitset<4> const& in,
                                 size_t const bit) mutable {
    auto const current_bit_set = in.test(bit);
    if (!started) {
      if (prev_was_set && current_bit_set) {
        started = true;
      }
      prev_was_set = current_bit_set;
    } else {
      if (current_bit_set) {
        b.set(i, current_bit_set);
      }
      ++i;
    }
    ++consumed;
  };

  for (auto const& c : char_collection) {
    auto const idx = static_cast<unsigned char>(c);
    auto const i = ascii_hex_to_int[idx];
    utl::verify(i >= 0, "invalid bitfield char {}", idx);
    auto const tmp = std::bitset<4>{static_cast<std::uint64_t>(i)};

    consume_bit(tmp, 3U);
    consume_bit(tmp, 2U);
    consume_bit(tmp, 1U);
    consume_bit(tmp, 0U);
  }

  utl::verify(started, "no start (11) found for traffic day bitfield {}",
              char_collection.view());

  bool prev_set = false;
  for (auto i = 0U; i != kMaxDays; ++i) {
    auto const j = kMaxDays - i - 1U;
    auto const curr_set = b.test(j);
    b.set(j, false);
    if (prev_set && curr_set) {
      return b;
    }
    prev_set = curr_set;
  }

  throw utl::fail("no end (11) found for traffic day bitfield {}",
                  char_collection.view());
}

bitfield_map_t parse_bitfields(config const& c, std::string_view file_content) {
  scoped_timer timer{"parse bitfields"};

  bitfield_map_t bitfields;
  utl::for_each_line_numbered(
      file_content, [&](utl::cstr line, unsigned const line_number) {
        if (line.len == 0 || line.str[0] == '%') {
          return;
        } else if (line.len < 9) {
          throw utl::fail("bitfield line too short line={} length={}",
                          line_number, line.len);
        }

        auto const index = parse_verify<unsigned>(line.substr(c.bf_.index_));
        auto b = hex_str_to_bitset(line.substr(c.bf_.value_));
        bitfields[index] = b;
      });

  return bitfields;
}

}  // namespace nigiri::loader::hrd
