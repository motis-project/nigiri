#pragma once

#include <cstdint>
#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include "nigiri/loader/merge_duplicates.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/td_footpath.h"
#include "nigiri/timetable.h"

namespace nigiri {
struct shapes_storage;
}

namespace nigiri::loader {

struct finalize_options {
  bool merge_dupes_intra_src_{true};
  bool merge_dupes_inter_src_{true};
  std::uint16_t max_footpath_length_{20};
  merge_threshold_t merge_threshold_{uniform_merge_threshold(duration_t{1})};
  std::filesystem::path merge_stats_dir_{};
  vector_map<source_idx_t, std::string> src_tags_{};
  // Walks instead of the beelines (motis: routed on OSM): fills the walks
  // (default profile, walking minutes) and the footpaths of the other
  // profiles, the transfer rules apply on them.
  std::function<void(timetable&,
                     mutable_fws_multimap<location_idx_t, footpath>&)>
      walks_{};
};

void build_footpaths(timetable& tt,
                     finalize_options,
                     shapes_storage* = nullptr);

// A profile's walks ignore transfer rules: a virtual location walks like its
// stop (a walk between two stops connects all their locations, the stop's
// transfer time connects its own locations). Rebuilds the profile's lower bound
// graphs.
void expand_profile_to_virts(timetable&, profile_idx_t);

}  // namespace nigiri::loader
