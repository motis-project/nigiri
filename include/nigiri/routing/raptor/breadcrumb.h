#pragma once

#include <cassert>
#include <cstdint>

namespace nigiri::routing {

// Packed reconstruction breadcrumb payload (adapted 1:1 from the gouda GPU
// raptor's breadcrumb.h; the realtime encoding is dropped - mcraptor does
// not support realtime). Everything needed to emit a transit leg and recurse
// is packed into 48 bits:
//   [47:37] alight stop_idx (11 bits)
//   [36:26] board  stop_idx (11 bits)
//   [25:0]  transport_idx   (26 bits, counting up from 0)
//
// NOT stored (recovered at reconstruction time, off the hot path):
//   * traffic day  - from the arrival time minus the event's over-midnight
//                    offset (a single footpath/transfer crosses midnight at
//                    most once, so two candidate days suffice)
//   * the footpath/transfer to the arrival location - derived by comparing
//     the ride's alight-stop location (route_location_seq_[r][alight]) to the
//     bag location: equal -> same-station transfer, else a footpath.
//
// Unlike the single-criterion GPU raptor - whose round_times_ matrix has one
// cell per (round, stop) so the previous leg is just (k-1, board) - mcraptor
// keeps a pareto *set* per stop, so the predecessor is ambiguous and the
// breadcrumb still carries an explicit parent arena index (stored next to the
// payload in the breadcrumb struct, not in this word).
using breadcrumb_t = std::uint64_t;

inline constexpr std::uint64_t kBcMask = 0x0000'FFFF'FFFF'FFFFULL;  // 48 bits
inline constexpr std::uint64_t kBcTransportMask = 0x03FF'FFFFULL;  // 26 bits
inline constexpr std::uint64_t kBcStopMask = 0x7FFULL;  // 11 bits
inline constexpr unsigned kBcBoardShift = 26U;
inline constexpr unsigned kBcAlightShift = 37U;
inline constexpr std::uint32_t kStartSentinel =
    static_cast<std::uint32_t>(kBcTransportMask);

inline breadcrumb_t make_transport_payload(std::uint32_t const transport_idx,
                                           std::uint32_t const board_stop,
                                           std::uint32_t const alight_stop) {
  assert(transport_idx < kStartSentinel);
  assert(board_stop <= kBcStopMask && alight_stop <= kBcStopMask);
  return (static_cast<breadcrumb_t>(transport_idx) & kBcTransportMask) |
         ((static_cast<breadcrumb_t>(board_stop) & kBcStopMask)
          << kBcBoardShift) |
         ((static_cast<breadcrumb_t>(alight_stop) & kBcStopMask)
          << kBcAlightShift);
}

inline breadcrumb_t make_start_bc() {
  return static_cast<breadcrumb_t>(kStartSentinel);
}

inline std::uint32_t bc_transport(breadcrumb_t const bc) {
  return static_cast<std::uint32_t>(bc & kBcTransportMask);
}

inline std::uint32_t bc_board(breadcrumb_t const bc) {
  return static_cast<std::uint32_t>((bc >> kBcBoardShift) & kBcStopMask);
}

inline std::uint32_t bc_alight(breadcrumb_t const bc) {
  return static_cast<std::uint32_t>((bc >> kBcAlightShift) & kBcStopMask);
}

inline bool bc_is_start(breadcrumb_t const bc) {
  return bc_transport(bc) == kStartSentinel;
}

// device-upload check analog: the transport range must fit in 26 bits
inline bool bc_transport_space_fits(std::uint64_t const n_transports) {
  return n_transports + 1U <= kStartSentinel;
}

}  // namespace nigiri::routing
