#pragma once

namespace nigiri::loader::hrd {

constexpr int hhmm_to_min(int const hhmm) {
  if (hhmm < 0) {
    return hhmm;
  } else {
    return (hhmm / 100) * 60 + (hhmm % 100);
  }
}

}  // namespace nigiri::loader::hrd
