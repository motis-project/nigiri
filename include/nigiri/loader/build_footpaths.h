#pragma once

#include "nigiri/timetable.h"

namespace nigiri::loader {

struct finalize_options {
  bool adjust_footpaths_{true};
  bool merge_dupes_intra_src_{true};
  bool merge_dupes_inter_src_{true};
  std::uint16_t max_footpath_length_{20U};
};

void build_footpaths(timetable& tt, finalize_options);

}  // namespace nigiri::loader
