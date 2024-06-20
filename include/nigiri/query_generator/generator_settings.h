#pragma once

#include "nigiri/query_generator/transport_mode.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/location_match_mode.h"
#include "nigiri/timetable.h"

namespace nigiri::query_generation {

struct generator_settings {
  duration_t interval_size_{60U};
  routing::location_match_mode start_match_mode_{
      routing::location_match_mode::kIntermodal};
  routing::location_match_mode dest_match_mode_{
      routing::location_match_mode::kIntermodal};
  transport_mode start_mode_{kWalk};
  transport_mode dest_mode_{kWalk};
  bool use_start_footpaths_{false};
  std::uint8_t max_transfers_{routing::kMaxTransfers};
  unsigned min_connection_count_{0U};
  bool extend_interval_earlier_{false};
  bool extend_interval_later_{false};
  profile_idx_t prf_idx_{0};
  routing::clasz_mask_t allowed_claszes_{routing::all_clasz_allowed()};
};

}  // namespace nigiri::query_generation