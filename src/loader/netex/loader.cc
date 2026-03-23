#include "nigiri/loader/netex/loader.h"

#include "nigiri/loader/netex/load_timetable.h"

namespace nigiri::loader::netex {

bool netex_loader::applicable(dir const& d) const {
  return nigiri::loader::netex::applicable(d);
}

void netex_loader::load(
    loader_config const& c,
    source_idx_t const src,
    dir const& d,
    timetable& tt,
    hash_map<bitfield, bitfield_idx_t>& global_bitfield_indices,
    assistance_times* assistance,
    shapes_storage* shapes_data) const {
  return nigiri::loader::netex::load_timetable(
      c, src, d, tt, global_bitfield_indices, assistance, shapes_data);
}

std::string_view netex_loader::name() const { return "netex"; }

}  // namespace nigiri::loader::netex
