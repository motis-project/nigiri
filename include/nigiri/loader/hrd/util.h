#pragma once

#include <string>
#include <vector>

namespace nigiri::loader::hrd {

constexpr int hhmm_to_min(int const hhmm) {
  if (hhmm < 0) {
    return hhmm;
  } else {
    return (hhmm / 100) * 60 + (hhmm % 100);
  }
}

template <typename T>
std::string iso_8859_1_to_utf8(T const& s) {
  std::vector<unsigned char> utf8(s.length() * 2, '\0');
  auto const input_size = s.length();
  auto in = reinterpret_cast<unsigned char const*>(&s[0]);
  auto const out_begin = &utf8[0];
  auto out = out_begin;
  for (auto i = std::size_t{0U}; i < input_size; ++i) {
    if (*in < 128) {
      *out++ = *in++;
    } else {
      *out++ = 0xc2 + (*in > 0xbfU);
      *out++ = (*in++ & 0x3fU) + 0x80U;
    }
  }
  return std::string{reinterpret_cast<char const*>(out_begin),
                     static_cast<std::size_t>(std::distance(out_begin, out))};
}

}  // namespace nigiri::loader::hrd
