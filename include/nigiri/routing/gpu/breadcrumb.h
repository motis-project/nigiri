#pragma once

#include <cstdint>

#include "cista/cuda_check.h"

namespace nigiri::routing::gpu {

// The arrival in a round_times entry is always produced by a train. We store
// everything needed to emit its run_enter_exit leg and recurse:
//   [47:37] alight stop_idx (11 bits)
//   [36:26] board  stop_idx (11 bits)
//   [25:0]  transport  (26 bits)
//             -> static transport_idx counting up from 0,
//             -> rt_transport_idx counting down from just below kStartSentinel
//                (device timetable upload verifies the two ranges plus the
//                 sentinel fit into 26 bits, so they never overlap)
//
// The traffic day is NOT stored: for a static transport it is recovered from
// the arrival time minus the event's over-midnight offset
// (transport_traffic_days_ is the first-departure day); an rt transport needs
// no recovery, its event times are stored absolute.
//
// The footpath/transfer hop to the round_times location is derived from
// alight: l' = the ride's stop sequence at alight (route_location_seq_[r] for
// static, rt_transport_location_seq_[rt_t] for rt); if l' == location it is a
// transfer, else a footpath l' -> location.
//
// Two special cases avoid needing a kind tag:
//   * start seed: transport == kStartSentinel (reconstruction stops here)
//   * intermodal egress: detected by location == kIntermodalTarget; the
//     breadcrumb is the egress ride's regular transport payload (the journey
//     terminal is its alighting stop; the last-mile leg is added on the
//     host).
//
// The 48-bit breadcrumb payload carried in the low bits of a packed round-time
// word (the high 16 bits are the time key; see device_times).
using breadcrumb_t = std::uint64_t;

inline constexpr std::uint64_t kBcMask = 0x0000'FFFF'FFFF'FFFFULL;  // 48 bits
inline constexpr std::uint64_t kBcTransportMask = 0x03FF'FFFFULL;  // 26 bits
inline constexpr std::uint64_t kBcStopMask = 0x7FFULL;  // 11 bits
inline constexpr unsigned kBcBoardShift = 26U;
inline constexpr unsigned kBcAlightShift = 37U;
inline constexpr std::uint32_t kStartSentinel =
    static_cast<std::uint32_t>(kBcTransportMask);

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

CISTA_CUDA_COMPAT inline std::uint32_t encode_rt_bc_transport(
    std::uint32_t const rt_transport_idx) {
  return kStartSentinel - 1U - rt_transport_idx;
}

CISTA_CUDA_COMPAT inline std::uint32_t decode_rt_bc_transport(
    std::uint32_t const field) {
  return kStartSentinel - 1U - field;
}

CISTA_CUDA_COMPAT inline bool is_rt_bc_transport(
    std::uint32_t const field, std::uint32_t const n_rt_transports) {
  return field != kStartSentinel && field >= kStartSentinel - n_rt_transports;
}

CISTA_CUDA_COMPAT inline bool bc_transport_space_fits(
    std::uint64_t const n_transports, std::uint64_t const n_rt_transports) {
  return n_transports + n_rt_transports + 1U <= kStartSentinel;
}

}  // namespace nigiri::routing::gpu
