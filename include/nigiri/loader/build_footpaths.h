#pragma once

#include <cstdint>
#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include "nigiri/loader/import_context.h"
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
  std::function<void(timetable&,
                     mutable_fws_multimap<location_idx_t, footpath>&)>
      compute_walks_{};
};

void build_footpaths(timetable& tt,
                     import_context&,
                     finalize_options const&,
                     shapes_storage* = nullptr);

}  // namespace nigiri::loader
