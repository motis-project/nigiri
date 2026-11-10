#pragma once

#include <cstdint>

#include "cista/cuda_check.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/types.h"

// McRAPTOR's extra criteria and breadcrumbs, shared by the CPU engine
// (raptor/mcraptor.h) and the GPU one (gpu/mcraptor_impl.cuh).
namespace nigiri::routing {

// ---- non_transit ------------------------------------------------------------
// Each walk (offset, footpath) counts its minutes floored to whole units of
// kNonTransitBin; units are summed and saturate at kNonTransitMax. Per walk,
// so the sum is order-independent (forward and backward searches agree).
inline constexpr bool kNonTransitFullResolution = false;
inline constexpr std::uint32_t kNonTransitBin =
    kNonTransitFullResolution ? 1U : 5U;
inline constexpr std::uint32_t kNonTransitMax = 0x7FU;

CISTA_CUDA_COMPAT constexpr std::uint32_t non_transit_add(
    std::uint32_t const units, std::uint32_t const minutes) {
  auto const sum = units + minutes / kNonTransitBin;
  return sum < kNonTransitMax ? sum : kNonTransitMax;
}

// ---- mode switches ----------------------------------------------------------
// Vehicle class groups; a change of group between trips is a switch.
inline constexpr std::uint32_t kNoModeGroup = 0U;  // no trip ridden yet
inline constexpr std::uint32_t kMaxModeSwitches = 0xFU;

CISTA_CUDA_COMPAT constexpr std::uint32_t mode_group(clasz const c) {
  switch (c) {
    case clasz::kHighSpeed: return 1U;
    case clasz::kLongDistance: return 2U;
    case clasz::kCoach: return 3U;
    case clasz::kNight: return 4U;
    case clasz::kRegional: return 5U;
    case clasz::kSuburban: [[fallthrough]];
    case clasz::kSubway: [[fallthrough]];
    case clasz::kTram: [[fallthrough]];
    case clasz::kBus: return 6U;
    default: return 7U;
  }
}

CISTA_CUDA_COMPAT constexpr std::uint32_t switches_after_ride(
    std::uint32_t const prev_group,
    std::uint32_t const switches,
    std::uint32_t const group) {
  auto const switched = prev_group != kNoModeGroup && prev_group != group;
  auto const s = switches + (switched ? 1U : 0U);
  return s < kMaxModeSwitches ? s : kMaxModeSwitches;
}

// Labels in different groups only dominate with a switch to spare (the next
// switch may differ); a completed journey has no next switch.
CISTA_CUDA_COMPAT constexpr bool switches_dominate(std::uint32_t const a_group,
                                                   std::uint32_t const a_sw,
                                                   std::uint32_t const b_group,
                                                   std::uint32_t const b_sw) {
  auto const penalty =
      (a_group == b_group || a_group == kNoModeGroup) ? 0U : 1U;
  return a_sw + penalty <= b_sw;
}

// ---- packed form (the GPU label's 16 bit extras slot) ---------------------
//   bit  0      mode_filter: uses an avoided vehicle class
//   bits 1..7   non_transit units
//   bits 8..10  mode group of the last trip
//   bits 11..14 mode switches
// Unused criteria stay 0, so one dominance rule serves every configuration.
namespace mc_extras {

inline constexpr std::uint32_t kNtShift = 1U;
inline constexpr std::uint32_t kGroupShift = 8U;
inline constexpr std::uint32_t kGroupMask = 0x7U;
inline constexpr std::uint32_t kSwShift = 11U;

CISTA_CUDA_COMPAT constexpr std::uint32_t mode_filter(std::uint32_t const x) {
  return x & 1U;
}
CISTA_CUDA_COMPAT constexpr std::uint32_t non_transit(std::uint32_t const x) {
  return (x >> kNtShift) & kNonTransitMax;
}
CISTA_CUDA_COMPAT constexpr std::uint32_t group(std::uint32_t const x) {
  return (x >> kGroupShift) & kGroupMask;
}
CISTA_CUDA_COMPAT constexpr std::uint32_t switches(std::uint32_t const x) {
  return (x >> kSwShift) & kMaxModeSwitches;
}

CISTA_CUDA_COMPAT constexpr std::uint32_t pack(std::uint32_t const mode_filter,
                                               std::uint32_t const non_transit,
                                               std::uint32_t const group,
                                               std::uint32_t const switches) {
  return mode_filter | (non_transit << kNtShift) | (group << kGroupShift) |
         (switches << kSwShift);
}

CISTA_CUDA_COMPAT constexpr bool dominates(std::uint32_t const a,
                                           std::uint32_t const b) {
  return mode_filter(a) <= mode_filter(b) && non_transit(a) <= non_transit(b) &&
         switches_dominate(group(a), switches(a), group(b), switches(b));
}

CISTA_CUDA_COMPAT constexpr bool completed_dominates(std::uint32_t const a,
                                                     std::uint32_t const b) {
  return mode_filter(a) <= mode_filter(b) && non_transit(a) <= non_transit(b) &&
         switches(a) <= switches(b);
}

// Without the group, (completed) dominance implies a <= b as integers.
CISTA_CUDA_COMPAT constexpr std::uint32_t ordered(std::uint32_t const x) {
  return x & ~(kGroupMask << kGroupShift);
}

}  // namespace mc_extras

// ---- breadcrumbs ------------------------------------------------------------
// The transit leg behind an arrival: the trip (transport_idx_t on traffic day
// day_, or rt_transport_idx_t if day_ is invalid), its board and alight stops,
// the arrival at the label's stop and the boarded label's breadcrumb (each
// engine has its own "start" sentinel).
struct mc_breadcrumb {
  std::uint32_t transport_;
  std::uint32_t parent_;
  day_idx_t day_;
  stop_idx_t board_;
  stop_idx_t alight_;
  delta_t arr_;
};
static_assert(sizeof(mc_breadcrumb) == 16U);

}  // namespace nigiri::routing
