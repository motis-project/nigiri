#pragma once

namespace nigiri::loader {

void transpose_bitfields(timetable& tt) {
  tt.day_bits_.resize(kMaxDays * tt.bitfields_.size());
  for (auto bf_idx = bitfield_idx_t{0U}; bf_idx != tt.bitfields_.size();
       ++bf_idx) {
    auto const& bf = tt.bitfields_[bf_idx];
    for (auto day_idx = 0U; day_idx != bf.size(); ++day_idx) {
      tt.day_bits_.set(day_idx * tt.bitfields_.size() + bf_idx.v_, bf[day_idx]);
    }
  }
}

}  // namespace nigiri::loader