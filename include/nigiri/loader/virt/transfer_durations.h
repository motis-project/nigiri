#pragma once

#include "nigiri/loader/transfer_rules.h"
#include "nigiri/transfer_rules.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

struct signatures;

struct transfer_durations {
  duration_t fallback_;
  window_t hull_;
};

bool is_admissible(window_t const&);

duration_t get_transfer_time(timetable const&, location_idx_t);

hash_map<transfer_pair, transfer_durations> get_transfer_durations(
    timetable const&,
    mutable_fws_multimap<location_idx_t, footpath> const& walks,
    signatures const&);

}  // namespace nigiri::loader
