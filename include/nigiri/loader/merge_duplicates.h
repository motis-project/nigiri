#pragma once

#include <filesystem>
#include <string>
#include <vector>

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

// merges duplicate transports. if stats_dir is set, merge_stats.json and
// merge_stats.html are written there describing what was found per agency and
// per feed
void merge_duplicates(timetable&,
                      merge_threshold_t const&,
                      bool intra_src,
                      bool inter_src,
                      std::filesystem::path const& stats_dir,
                      std::vector<std::string> const& src_tags);

}  // namespace nigiri::loader
