#pragma once

#include "nigiri/loader/merge_duplicates.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

struct finalize_options {
  bool adjust_footpaths_{true};
  bool merge_dupes_intra_src_{true};
  bool merge_dupes_inter_src_{true};
  std::uint16_t max_footpath_length_{20};
  merge_threshold_t merge_threshold_{uniform_merge_threshold(duration_t{1})};
  std::filesystem::path merge_stats_dir_{};
  vector_map<source_idx_t, std::string> src_tags_{};
};

void build_footpaths(timetable& tt, finalize_options);

// Drop the default profile's footpaths that a hub already hands out at the
// same weight or better.
void prune_hub_covered_footpaths(timetable&);

}  // namespace nigiri::loader
