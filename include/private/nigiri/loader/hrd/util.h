#pragma once

namespace nigiri::loader::hrd {

constexpr int hhmm_to_min(int const hhmm) {
  if (hhmm < 0) {
    return hhmm;
  } else {
    return (hhmm / 100) * 60 + (hhmm % 100);
  }
}

constexpr duration_t operator""_hhmm(unsigned long long hhmm) {
  return duration_t{hhmm_to_min(hhmm)};
}

}  // namespace nigiri::loader::hrd
