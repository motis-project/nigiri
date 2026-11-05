#include "nigiri/loader/gtfs/loader.h"

#include "nigiri/loader/gtfs/load_timetable.h"

namespace nigiri::loader::gtfs {

bool gtfs_loader::applicable(dir const& d) const {
  return nigiri::loader::gtfs::applicable(d);
}

void gtfs_loader::load(loader_config const& c,
                       source_idx_t const src,
                       dir const& d,
                       timetable& tt,
                       import_context& ctx,
                       assistance_times* assistance,
                       shapes_storage* shapes_data) const {
  return nigiri::loader::gtfs::load_timetable(c, src, d, tt, ctx, assistance,
                                              shapes_data);
}

std::string_view gtfs_loader::name() const { return "gtfs"; }

}  // namespace nigiri::loader::gtfs