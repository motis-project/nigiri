#pragma once

#include <filesystem>
#include <string>

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

// max. average per-event time difference for two transports of the same
// class to be considered duplicates of each other
using merge_threshold_t = std::array<duration_t, kNumClasses>;

constexpr merge_threshold_t uniform_merge_threshold(duration_t const d) {
  auto t = merge_threshold_t{};
  t.fill(d);
  return t;
}

void merge_duplicates(timetable&,
                      merge_threshold_t const&,
                      bool intra_src,
                      bool inter_src,
                      std::filesystem::path const& stats_dir,
                      vector_map<source_idx_t, std::string> const& src_tags);

}  // namespace nigiri::loader
