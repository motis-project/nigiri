#pragma once

#include <cstdint>

#include "cista/cuda_check.h"

namespace nigiri::routing::gpu {

// A breadcrumb is the 48-bit payload stored in the low bits of a packed
// round-time word (see device_times). It records how a location was reached in
// a given round so the journey can be reconstructed on the GPU as a pure
// pointer-chase -- no transfer of the round-time matrix and no re-derivation by
// walking the trip.
//
// The arrival in a round_times entry is always produced by a train. We store
// everything needed to emit its run_enter_exit leg and recurse:
//   [47:37] alight stop_idx (11 bits) -- where the train was left
//   [36:26] board  stop_idx (11 bits) -- where the train was boarded
//   [25:0]  transport_idx   (26 bits)
// The traffic day is NOT stored (recovered from the arrival time minus the
// event's over-midnight offset; transport_traffic_days_ is the first-departure
// day). The footpath/transfer hop to the round_times location is derived from
// alight: l' = route_location_seq_[r][alight]; if l' == location -> transfer,
// else footpath l' -> location.
//
// Two special cases avoid needing a kind tag:
//   * start seed: transport_idx == kStartSentinel (reconstruction stops here)
//   * intermodal egress: detected by location == kIntermodalTarget; then the
//     low 26 bits hold the egress source location_idx (handled on the host).
//
// tmp_ stores the same transport payload (transport|board|alight); round_times
// kTransport entries copy it verbatim.

// The 48-bit breadcrumb payload carried in the low bits of a packed round-time
// word (the high 16 bits are the time key; see device_times). Same underlying
// type as the packed word -- the alias only documents that a value is a bare
// breadcrumb, not a full time||breadcrumb word.
using breadcrumb_t = std::uint64_t;

inline constexpr std::uint64_t kBcMask = 0x0000'FFFF'FFFF'FFFFULL;  // 48 bits
inline constexpr std::uint64_t kBcTransportMask = 0x03FF'FFFFULL;  // 26 bits
inline constexpr std::uint64_t kBcStopMask = 0x7FFULL;  // 11 bits
inline constexpr unsigned kBcBoardShift = 26U;
inline constexpr unsigned kBcAlightShift = 37U;
inline constexpr std::uint32_t kStartSentinel =
    static_cast<std::uint32_t>(kBcTransportMask);  // 26-bit all-ones

CISTA_CUDA_COMPAT inline breadcrumb_t make_transport_payload(
    std::uint32_t const transport_idx,
    std::uint32_t const board_stop,
    std::uint32_t const alight_stop) {
  return (static_cast<breadcrumb_t>(transport_idx) & kBcTransportMask) |
         ((static_cast<breadcrumb_t>(board_stop) & kBcStopMask)
          << kBcBoardShift) |
         ((static_cast<breadcrumb_t>(alight_stop) & kBcStopMask)
          << kBcAlightShift);
}

CISTA_CUDA_COMPAT inline breadcrumb_t make_start_bc() {
  return static_cast<breadcrumb_t>(kStartSentinel);
}

CISTA_CUDA_COMPAT inline breadcrumb_t make_egress_bc(
    std::uint32_t const source_location) {
  return static_cast<breadcrumb_t>(source_location) & kBcTransportMask;
}

CISTA_CUDA_COMPAT inline std::uint32_t bc_transport(breadcrumb_t const bc) {
  return static_cast<std::uint32_t>(bc & kBcTransportMask);
}

CISTA_CUDA_COMPAT inline std::uint32_t bc_board(breadcrumb_t const bc) {
  return static_cast<std::uint32_t>((bc >> kBcBoardShift) & kBcStopMask);
}

CISTA_CUDA_COMPAT inline std::uint32_t bc_alight(breadcrumb_t const bc) {
  return static_cast<std::uint32_t>((bc >> kBcAlightShift) & kBcStopMask);
}

CISTA_CUDA_COMPAT inline bool bc_is_start(breadcrumb_t const bc) {
  return bc_transport(bc) == kStartSentinel;
}

}  // namespace nigiri::routing::gpu
