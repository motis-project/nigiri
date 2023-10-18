#include "nigiri/loader/netex/loader.h"

#include "nigiri/loader/netex/load_timetable.h"

namespace nigiri::loader::netex {

void netex_loader::load(loader_config const&,
                        source_idx_t const src,
                        dir const& d,
                        timetable& tt) const {
  return nigiri::loader::netex::load_timetable(src, d, tt);
}

}  // namespace nigiri::loader::netex