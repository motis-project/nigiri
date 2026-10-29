#pragma once

#include <optional>

#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

struct rule_index {
  explicit rule_index(mutable_fws_multimap<location_idx_t, footpath>&&);

  std::optional<duration_t> get(location_idx_t from, location_idx_t to) const;

  mutable_fws_multimap<location_idx_t, footpath> footpaths_;
};

rule_index get_rule_index(timetable const&);

void write_default_profile(
    timetable&,
    mutable_fws_multimap<location_idx_t, footpath> const& walks,
    mutable_fws_multimap<location_idx_t, footpath> const& durations = {});

}  // namespace nigiri::loader
