#include "nigiri/transfer_rules.h"

#include "nigiri/timetable.h"

namespace nigiri {

bool covers(timetable const& tt,
            location_idx_t const stop,
            location_idx_t const base) {
  return stop == base || tt.locations_.parents_[base] == stop;
}

}  // namespace nigiri
