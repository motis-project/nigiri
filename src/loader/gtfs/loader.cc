#include "nigiri/loader/gtfs/loader.h"

#include "nigiri/loader/gtfs/load_timetable.h"

namespace nigiri::loader::gtfs {

bool gtfs_loader::applicable(dir const& d) const {
  return nigiri::loader::gtfs::applicable(d);
}

void gtfs_loader::load(
    loader_config const& c,
    source_idx_t const src,
    dir const& d,
    timetable& tt,
    hash_map<bitfield, bitfield_idx_t>& global_bitfield_indices,
    assistance_times* assistance,
    shapes_storage* shapes_data) const {
  return nigiri::loader::gtfs::load_timetable(
      c, src, d, tt, global_bitfield_indices, assistance, shapes_data);
}

std::string_view gtfs_loader::name() const { return "gtfs"; }

}  // namespace nigiri::loader::gtfs