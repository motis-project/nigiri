#pragma once

#include "nigiri/types.h"

namespace nigiri::loader::netex {

using service_journey_idx_t =
    cista::strong<std::uint32_t, struct _service_journey_idx>;

struct utc_trip {
  date::days first_dep_offset_;
  duration_t tz_offset_;
  basic_string<duration_t> utc_times_;
  bitfield utc_traffic_days_;
  gtfs::stop_seq_t stop_seq_;
  basic_string<service_journey_idx_t> trips_;
  basic_string<translation_idx_t> trip_direction_;
  basic_string<attribute_combination_idx_t> attributes_;
  route_id_idx_t route_id_;
};

}  // namespace nigiri::loader::netex