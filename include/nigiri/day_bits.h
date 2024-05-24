#pragma once

#include "nigiri/types.h"

namespace nigiri {

struct day_bits {
  explicit day_bits(std::uint32_t const n_days, std::uint32_t const n_bitfields)
      : n_days_{n_days}, n_bitfields_{n_bitfields} {
    bits_.resize(n_days * n_bitfields);
  }

  void set(bitfield_idx_t const bf_idx, bitfield const& bf) {
    assert(bf.size() == n_days_ && "bitfield length mismatch");
    assert(bf_idx < n_bitsfields_ && "bitfield idx out of bounds");
    for (auto day_idx = 0U; day_idx != bf.size(); ++day_idx) {
      bits_.set(day_idx * n_bitfields_ + bf_idx.v_, bf[day_idx]);
    }
  }

  bool test(day_idx_t const day_idx, bitfield_idx_t const bf_idx) {
    assert(day_idx.v_ < n_days && "day idx out of bounds");
    assert(bf_idx < n_bitsfields_ && "bitfield idx out of bounds");
    return bits_.test(day_idx.v_ * n_bitfields_ + bf_idx.v_);
  }

  std::uint32_t n_days_;
  std::uint32_t n_bitfields_;
  bitvec bits_{};
};

}  // namespace nigiri