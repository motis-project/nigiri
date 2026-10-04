#pragma once

#include <algorithm>
#include <optional>

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

std::optional<duration_t> adjust_to_walk_speed(timetable const&,
                                               location_idx_t a,
                                               location_idx_t b,
                                               duration_t);

void write_default_profile(timetable&, bool adjust_footpaths);

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
