#pragma once

#include <cstdint>
#include <functional>

#include "nigiri/loader/import_context.h"
#include "nigiri/loader/merge_duplicates.h"
#include "nigiri/loader/station_footpaths.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/td_footpath.h"
#include "nigiri/timetable.h"

namespace nigiri {
struct shapes_storage;
struct station_paths;
}  // namespace nigiri

namespace nigiri::loader {

struct finalize_options {
  bool merge_dupes_intra_src_{true};
  bool merge_dupes_inter_src_{true};
  std::uint16_t max_footpath_length_{20};
  merge_threshold_t merge_threshold_{uniform_merge_threshold(duration_t{1})};
  std::filesystem::path merge_stats_dir_{};
  vector_map<source_idx_t, std::string> src_tags_{};
  // Writes the walks (default profile) and the footpaths of the other
  // profiles; footpaths with restriction records (e.g. routed alternatives
  // without steps) go to the third parameter, per profile.
  std::function<void(timetable&,
                     mutable_fws_multimap<location_idx_t, footpath>&,
                     station_footpaths_t&)>
      compute_walks_{};
  station_paths* station_paths_{nullptr};  // geometry of station graph paths
};

// The walk, at least the transfer times of both stops.
duration_t max_with_transfer_times(timetable const&,
                                   location_idx_t from,
                                   location_idx_t to,
                                   duration_t walk);

void build_footpaths(timetable& tt,
                     import_context&,
                     finalize_options const&,
                     shapes_storage* = nullptr);

}  // namespace nigiri::loader
