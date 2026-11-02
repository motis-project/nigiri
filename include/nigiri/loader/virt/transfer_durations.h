#pragma once

#include <optional>

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

std::optional<duration_t> pick_duration(window_t const&, duration_t fallback);

bool is_admissible(window_t const&);

duration_t get_transfer_time(timetable const&, location_idx_t);

hash_map<transfer_pair, transfer_durations> get_transfer_durations(
    timetable const&,
    mutable_fws_multimap<location_idx_t, footpath> const& walks,
    signatures const&);

}  // namespace nigiri::loader
