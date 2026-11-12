#pragma once

#include "nigiri/loader/transfer_rules.h"
#include "nigiri/loader/virt/signatures.h"
#include "nigiri/timetable.h"
#include "nigiri/transfer_rules.h"
#include "nigiri/types.h"

namespace nigiri::loader {

struct transfer_durations {
  duration_t fallback_;
  window_t hull_;
};

bool is_admissible(window_t const&);

duration_t get_transfer_time(timetable const&, location_idx_t);

hash_map<transfer_pair, transfer_durations> get_transfer_durations(
    timetable const&,
    mutable_fws_multimap<location_idx_t, footpath> const& fallback_fps,
    signatures const&);

}  // namespace nigiri::loader
