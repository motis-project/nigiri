#pragma once

namespace nigiri::loader {

void transpose_bitfields(timetable& tt) {
  // resize
  tt.day_bits_.resize(kMaxDays);
  for (auto& bv : tt.day_bits_) {
    bv.resize(tt.bitfields_.size());
  }
  // fill
  for (auto bf_idx = 0U; bf_idx != tt.bitfields_.size(); ++bf_idx) {
    auto const& bf = tt.bitfields_[bitfield_idx_t{bf_idx}];
    for (auto bit_idx = 0U; bit_idx != bf.size(); ++bit_idx) {
      tt.day_bits_[bit_idx].set(bf_idx, bf[bit_idx]);
    }
  }
}

}  // namespace nigiri::loader