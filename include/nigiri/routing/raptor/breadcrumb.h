#pragma once

#include <cassert>
#include <cstdint>

namespace nigiri::routing {

// Packed reconstruction breadcrumb payload (adapted 1:1 from the gouda GPU
// raptor's breadcrumb.h). Everything needed to emit a transit leg and
// recurse is packed into 48 bits:
//   [47:37] alight stop_idx (11 bits)
//   [36:26] board  stop_idx (11 bits)
//   [25:0]  transport        (26 bits)
//             -> static transport_idx counting up from 0,
//             -> rt_transport_idx counting down from just below
//                kStartSentinel (bc_transport_space_fits() verifies the two
//                ranges plus the sentinel fit into 26 bits, so they never
//                overlap)
//
// NOT stored (recovered at reconstruction time, off the hot path):
//   * traffic day  - for a static transport, from the arrival time minus the
//                    event's over-midnight offset (a single footpath/transfer
//                    crosses midnight at most once, so two candidate days
//                    suffice). An rt transport needs no recovery: its event
//                    times are stored absolute.
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

// rt transports occupy the top of the 26-bit transport space, counting DOWN
// from just below the start sentinel, so the two index spaces share the field
// without a kind tag (identical encoding to gpu/breadcrumb.h).
inline std::uint32_t encode_rt_bc_transport(
    std::uint32_t const rt_transport_idx) {
  return kStartSentinel - 1U - rt_transport_idx;
}

inline std::uint32_t decode_rt_bc_transport(std::uint32_t const field) {
  return kStartSentinel - 1U - field;
}

inline bool is_rt_bc_transport(std::uint32_t const field,
                               std::uint32_t const n_rt_transports) {
  return field != kStartSentinel && field >= kStartSentinel - n_rt_transports;
}

// the static and rt transport ranges plus the sentinel must fit in 26 bits
inline bool bc_transport_space_fits(std::uint64_t const n_transports,
                                    std::uint64_t const n_rt_transports = 0U) {
  return n_transports + n_rt_transports + 1U <= kStartSentinel;
}

}  // namespace nigiri::routing
