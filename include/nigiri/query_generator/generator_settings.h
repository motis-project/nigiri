#pragma once

#include "geo/box.h"

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/location_match_mode.h"
#include "nigiri/timetable.h"

#include "transport_mode.h"

namespace nigiri::query_generation {

struct generator_settings {

  friend std::ostream& operator<<(std::ostream& out,
                                  generator_settings const& gs) {
    using namespace nigiri::routing;
    auto const match_mode_str = [](auto const& mm) {
      return mm == location_match_mode::kIntermodal ? "intermodal" : "station";
    };

    out << "interval_size: " << gs.interval_size_.count();
    if (gs.bbox_.has_value()) {
      out << "\nbbox: min(" << gs.bbox_.value().min_ << "), max("
          << gs.bbox_.value().max_ << ")";
    }
    out << "\nstart_mode: " << match_mode_str(gs.start_match_mode_)
        << "\ndest_mode: " << match_mode_str(gs.dest_match_mode_)
        << "\nintermodal_start_mode: " << gs.start_mode_
        << "\nintermodal_dest_mode: " << gs.dest_mode_
        << "\nuse_start_footpaths: "
        << (gs.use_start_footpaths_ ? "true" : "false")
        << "\nmax_transfers: " << std::uint32_t{gs.max_transfers_}
        << "\nmin_connection_count: " << gs.min_connection_count_
        << "\nextend_interval_earlier: "
        << (gs.extend_interval_earlier_ ? "true" : "false")
        << "\nextend_interval_later: "
        << (gs.extend_interval_later_ ? "true" : "false")
        << "\nprf_idx: " << std::uint32_t{gs.prf_idx_}
        << "\nallowed_claszes: " << gs.allowed_claszes_;
    return out;
  }

  duration_t interval_size_{60U};
  std::optional<geo::box> bbox_;
  routing::location_match_mode start_match_mode_{
      routing::location_match_mode::kIntermodal};
  routing::location_match_mode dest_match_mode_{
      routing::location_match_mode::kIntermodal};
  transport_mode start_mode_{kWalk};
  transport_mode dest_mode_{kWalk};
  bool use_start_footpaths_{true};
  std::uint8_t max_transfers_{routing::kMaxTransfers};
  unsigned min_connection_count_{0U};
  bool extend_interval_earlier_{false};
  bool extend_interval_later_{false};
  profile_idx_t prf_idx_{0};
  routing::clasz_mask_t allowed_claszes_{routing::all_clasz_allowed()};
};

}  // namespace nigiri::query_generation