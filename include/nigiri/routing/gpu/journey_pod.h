#pragma once

#include <cstdint>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/limits.h"
#include "nigiri/types.h"

namespace nigiri::routing::gpu {

inline constexpr unsigned kMaxRecLegs = 2U * kMaxTransfers + 4U;

struct gpu_journey_leg {
  bool is_footpath_;
  location_idx_t from_l_;
  location_idx_t to_l_;
  delta_t dep_;
  delta_t arr_;
  transport_idx_t transport_;
  rt_transport_idx_t rt_transport_;
  day_idx_t day_;
  stop_idx_t enter_stop_;
  stop_idx_t exit_stop_;
  std::uint16_t fp_duration_;
};

enum class reconstruction_result : std::uint8_t {
  kNotReconstructed,
  kOk,
  kReconstructionFailed,
};

struct gpu_journey {
  reconstruction_result state_;
  std::uint8_t n_legs_;
  std::uint8_t transfers_;
  location_idx_t dest_l_;
  delta_t dest_time_;
  // generalized-cost extras of the label (mcraptor cost config; the host
  // adds the elapsed part) - unused by the single-criterion raptor
  std::uint16_t criteria_cost_;
  // tight-start shift (mcraptor, see gpu_mcraptor::set_tight_start):
  // journey's latest feasible departure minus the step start, i.e. first
  // boarding departure minus the minimum-walk feasible round-0 label of
  // the boarding stop. 0 when unavailable. Unused by the single-criterion
  // raptor.
  delta_t start_shift_;
  location_idx_t start_l_;
  gpu_journey_leg legs_[kMaxRecLegs];
};

}  // namespace nigiri::routing::gpu
