#pragma once

#include <cstdint>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/limits.h"

namespace nigiri::routing::gpu {

inline constexpr unsigned kMaxRecLegs = 2U * kMaxTransfers + 4U;

struct gpu_journey_leg {
  std::uint8_t is_footpath_;  // 1 = footpath, 0 = transport (run_enter_exit)
  std::uint32_t from_l_;  // location_idx value
  std::uint32_t to_l_;
  delta_t dep_;  // delta relative to base
  delta_t arr_;
  std::uint32_t transport_;  // transport_idx value (transport legs)
  std::uint16_t day_;  // day_idx value (transport legs)
  std::uint16_t enter_stop_;  // route stop_idx (transport legs)
  std::uint16_t exit_stop_;
  std::uint16_t fp_duration_;  // minutes (footpath legs)
};

struct gpu_journey {
  std::uint8_t valid_;
  std::uint8_t n_legs_;
  std::uint8_t transfers_;
  std::uint32_t dest_l_;  // location_idx value
  delta_t dest_time_;  // delta relative to base
  std::uint32_t start_l_;  // location_idx value of the resolved start
  gpu_journey_leg legs_[kMaxRecLegs];
};

}  // namespace nigiri::routing::gpu
