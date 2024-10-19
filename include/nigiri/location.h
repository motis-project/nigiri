#pragma once

#include <iosfwd>
#include <span>

#include "fmt/ostream.h"

#include "geo/latlng.h"

#include "nigiri/common/it_range.h"
#include "nigiri/types.h"
#include "nigiri/routing/gpu_timetable.h"

namespace nigiri {
struct timetable;

struct location {
  friend std::ostream& operator<<(std::ostream&, location const&);
  location(timetable const&, location_idx_t);
  location(gpu_timetable const&, gpu_location_idx_t);
  location(std::string_view id,
           std::string_view name,
           geo::latlng pos,
           source_idx_t,
           location_type,
           location_idx_t parent,
           timezone_idx_t,
           duration_t transfer_time,
           it_range<vector<location_idx_t>::const_iterator> equivalences);
  location_idx_t l_{location_idx_t::invalid()};
  std::string_view id_;
  std::string_view name_;
  geo::latlng pos_;
  source_idx_t src_;
  location_type type_;
  location_idx_t parent_;
  timezone_idx_t timezone_idx_;
  duration_t transfer_time_;
  it_range<vector<location_idx_t>::const_iterator> equivalences_;
};

}  // namespace nigiri

template <>
struct fmt::formatter<nigiri::location> : ostream_formatter {};