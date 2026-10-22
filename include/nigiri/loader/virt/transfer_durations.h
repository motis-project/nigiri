#pragma once

#include "nigiri/loader/transfer_rules.h"
#include "nigiri/common/interval.h"
#include "nigiri/routing/limits.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

struct signatures;

using window_t = interval<duration_t>;

constexpr auto const kNever = duration_t{routing::kMaxTravelTime};

struct transfer_durations {
  duration_t fallback_;
  window_t hull_;
};

duration_t get_walk(timetable const&, location_idx_t from, location_idx_t to);

hash_map<transfer_pair, transfer_durations> get_transfer_durations(
    timetable const&, signatures const&);

}  // namespace nigiri::loader
