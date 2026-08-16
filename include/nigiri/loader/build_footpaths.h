#pragma once

#include "nigiri/timetable.h"

namespace nigiri::loader {

struct finalize_options {
  bool adjust_footpaths_{true};
  bool merge_dupes_intra_src_{true};
  bool merge_dupes_inter_src_{true};
  std::uint16_t max_footpath_length_{20};

  // Derive walking transfers (beelines between equivalent stops, copied onto
  // generated children) while loading. Off for callers that compute the whole
  // footpath layer themselves - the transfers a rule states are written
  // either way, they are not walking transfers.
  bool beeline_footpaths_{true};

  // Do not copy the walking transfers onto generated children - the routing
  // resolves them through the parent instead.
  bool share_child_footpaths_{false};
};

void build_footpaths(timetable& tt, finalize_options);

}  // namespace nigiri::loader
