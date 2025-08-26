#pragma once

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

// Graph node.
using segment_idx_t = cista::strong<std::uint32_t, struct _segment_idx>;

using tb_bitfield_idx_t = cista::strong<std::uint32_t, struct _tb_bitfield_idx>;

struct transfer {
  bitfield_idx_t::value_t traffic_days_ : 31;
  bitfield_idx_t::value_t crosses_midnight_ : 1;
};

struct tb_data {
  interval<segment_idx_t> get_segment_range(transport_idx_t const t) {
    return interval{transport_first_segment_[t],
                    transport_first_segment_[t + 1]};
  }

  profile_idx_t prf_idx_;
  vector_map<transport_idx_t, segment_idx_t> transport_first_segment_;
  vecvec<segment_idx_t, segment_idx_t> segment_transfers_;
  vecvec<segment_idx_t, transfer> segment_traffic_days_;
  vector_map<tb_bitfield_idx_t, bitfield> bitfields_;
};

tb_data preprocess(timetable const&, profile_idx_t);

}  // namespace nigiri::routing::tb