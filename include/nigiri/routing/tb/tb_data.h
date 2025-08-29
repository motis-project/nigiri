#pragma once

#include "nigiri/types.h"

namespace nigiri::routing::tb {

using segment_idx_t = cista::strong<std::uint32_t, struct _segment_idx>;

using tb_bitfield_idx_t = cista::strong<std::uint32_t, struct _tb_bitfield_idx>;

struct transfer {
  segment_idx_t to_segment_;
  transport_idx_t to_transport_;
  tb_bitfield_idx_t traffic_days_;

  // Shift amount between trip t and trip u:
  // This includes:
  //   - (positive) how many times did t cross midnight until the transfer
  //   - (negative) how many times did u cross midnight until the transfer
  //   - (+1) does the transfer itself cross midnight
  std::int8_t day_offset_;
};

struct tb_data {
  interval<segment_idx_t> get_segment_range(transport_idx_t const t) const {
    return interval{transport_first_segment_[t],
                    transport_first_segment_[t + 1]};
  }

  profile_idx_t prf_idx_;
  vector_map<transport_idx_t, segment_idx_t> transport_first_segment_;
  vecvec<segment_idx_t, transfer> segment_transfers_;
  vector_map<segment_idx_t, transport_idx_t> segment_transports_;
  vector_map<tb_bitfield_idx_t, bitfield> bitfields_;
};

}  // namespace nigiri::routing::tb