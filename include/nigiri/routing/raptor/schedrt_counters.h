#pragma once

#include <atomic>
#include <cstdint>

// measurement-only: the end-of-phase cell scan roughly doubles query cost,
// keep off for any timing run
#ifndef NIGIRI_SCHEDRT_COUNTERS
#define NIGIRI_SCHEDRT_COUNTERS 0
#endif

namespace nigiri::routing {

// experiment instrumentation for the combined scheduled+rt search:
// how often do the two worlds actually diverge?
struct schedrt_divergence_counters {
  // boarding sites: label pair equal (fused dual walk) vs split
  std::atomic<std::uint64_t> board_fused_{0U};
  std::atomic<std::uint64_t> board_split_{0U};
  // fused walks whose two worlds picked different transports
  std::atomic<std::uint64_t> et_dual_diverged_{0U};
  // end-of-phase state: touched (round, location) cells with equal /
  // diverged slot pairs
  std::atomic<std::uint64_t> cells_equal_{0U};
  std::atomic<std::uint64_t> cells_diverged_{0U};
};

schedrt_divergence_counters& get_schedrt_divergence_counters();

}  // namespace nigiri::routing
