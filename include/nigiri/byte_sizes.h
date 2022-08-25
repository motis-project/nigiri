#pragma once

#include <cinttypes>

namespace nigiri {

constexpr std::size_t operator""_kB(unsigned long long v) { return 1024U * v; }
constexpr std::size_t operator""_MB(unsigned long long v) {
  return 1024U * 1024U * v;
}

}  // namespace nigiri
