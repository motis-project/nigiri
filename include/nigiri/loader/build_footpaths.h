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
  bool apply_transfer_rules_{true};
  // Walks instead of the beelines (motis: routed on OSM): fills
  // preprocessing_footpaths_out_ (default profile) and the footpaths of the
  // other profiles, the transfer rules apply on them.
  std::function<void(timetable&)> walks_{};
};

void build_footpaths(timetable& tt,
                     finalize_options,
                     shapes_storage* = nullptr);

void write_default_profile(timetable&);

// Profiles ignore transfer rules: a virtual location walks like its stop (a
// walk between two stops connects all their locations, the stop's transfer
// time connects its own locations). Rebuilds the profile's lower bound graphs.
void expand_profile_to_virts(timetable&, profile_idx_t);

// The same for time-dependent walks between stops (by their source): they
// replace all static walks of their source, so they connect its locations.
void expand_td_footpaths_to_virts(
    timetable const&, std::map<location_idx_t, std::vector<td_footpath>>&);

inline duration_t max_with_transfer_times(timetable const& tt,
                                          location_idx_t const from,
                                          location_idx_t const to,
                                          duration_t const walk) {
  auto const transfer_time = [&](location_idx_t const l) {
    auto const t = tt.locations_.transfer_time_[l];
    return t == kNoTransferAllowed ? duration_t{0} : duration_t{t};
  };
  return std::max({walk, transfer_time(from), transfer_time(to)});
}

}  // namespace nigiri::loader
