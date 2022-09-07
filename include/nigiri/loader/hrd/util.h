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
  std::string utf8(s.length() * 2, '\0');
  auto const input_size = s.length();
  auto in = reinterpret_cast<unsigned char const*>(&s[0]);
  auto const out_begin = &utf8[0];
  auto out = out_begin;
  for (auto i = std::size_t{0U}; i < input_size; ++i) {
    if (*in < 128) {
      *out++ = static_cast<char>(*in++);
    } else {
      *out++ = static_cast<char>(0xc2 + (*in > 0xbfU));
      *out++ = static_cast<char>((*in++ & 0x3fU) + 0x80U);
    }
  }
  utf8.resize(std::distance(out_begin, out));
  return utf8;
}

}  // namespace nigiri::loader::hrd
