#pragma once

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

// Graph node.
using segment_idx_t = cista::strong<std::uint32_t, struct _segment_idx>;

struct transfer {
  bitfield_idx_t::value_t traffic_days_ : 31;
  bitfield_idx_t::value_t crosses_midnight_ : 1;
};

struct tb_data {
  vecvec<segment_idx_t, segment_idx_t> segment_transfers_;
  vecvec<segment_idx_t, transfer> segment_traffic_days_;
};

tb_data preprocess(timetable&);

}  // namespace nigiri::routing::tb