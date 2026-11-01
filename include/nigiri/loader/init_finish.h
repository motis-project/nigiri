#pragma once

#include <cinttypes>
#include <limits>

#include "nigiri/loader/build_footpaths.h"

namespace nigiri {
struct timetable;
struct shapes_storage;
}  // namespace nigiri

namespace nigiri::loader {

void register_special_stations(timetable&, import_context&);
void finalize(timetable&,
              import_context&,
              finalize_options const&,
              shapes_storage* = nullptr);
void finalize(timetable&,
              import_context&,
              bool merge_dupes_intra_src = false,
              bool merge_dupes_inter_src = false,
              std::uint16_t max_footpath_length =
                  std::numeric_limits<std::uint16_t>::max());

}  // namespace nigiri::loader