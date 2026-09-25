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
  // false: no hubs at all, every transfer is an explicit footpath (see
  // materialize_hubs)
  bool hubs_{true};
};

void build_footpaths(timetable& tt, finalize_options);

// Replaces the walks of the default profile of a finished timetable with
// `walks` - e.g. those a street router found: the beelines the loader wrote
// exist only because nigiri cannot route. Everything transfers.txt states stays
// authoritative (a rule fixes the transfer time, shorter or longer than the
// walk; a ban removes the walk), the rule-derived hubs are kept, and the walk
// hubs, which hand a stop's walks to its virtual locations, the footpath
// layer and the lower bound graphs are rebuilt. `walks` holds stops only: a
// virtual location walks through its stop.
void rebuild_default_profile(
    timetable&, vector_map<location_idx_t, std::vector<footpath>> const& walks);

// Drop the default profile's footpaths that a hub already hands out at the
// same weight or better.
void prune_hub_covered_footpaths(timetable&);

// Every transfer a hub stands for, as an explicit footpath - in all profiles;
// the timetable has no hubs afterwards. Routing gives the same journeys on
// it, which makes it the reference to check the hubs against, and it serves
// the consumers that read footpaths only (e.g. trip-based routing). The
// price is the size: a stop with n virtual locations has n^2 transfers.
// It has to come last: rebuild_default_profile keeps the rule-derived hubs
// and replaces the rest of the layer, so it cannot run on a timetable whose
// hubs were materialized.
void materialize_hubs(timetable&);

}  // namespace nigiri::loader
