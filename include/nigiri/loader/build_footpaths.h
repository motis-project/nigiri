#pragma once

#include "nigiri/timetable.h"

namespace nigiri::loader {

struct finalize_options {
  bool adjust_footpaths_{true};
  bool merge_dupes_intra_src_{true};
  bool merge_dupes_inter_src_{true};
  std::uint16_t max_footpath_length_{20};
  // false = no transitive closure: input footpaths, parent/child links and
  // transfer rule edges are used as-is (HAFAS semantics; automatically
  // selected by loader::load() for a single HRDF dataset)
  bool transitive_footpaths_{true};
};

void build_footpaths(timetable& tt, finalize_options);

// (re)build the implicit-transfer hub edge lists for one profile from that
// profile's final footpaths. called by build_footpaths for the default
// profile; street-routing integrations (motis) must call it again for every
// profile whose footpaths they rebuild.
void build_hubs(timetable& tt, profile_idx_t prf);

}  // namespace nigiri::loader
