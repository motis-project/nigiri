#pragma once

#include "geo/latlng.h"

#include "oh/parser.h"

#include "nigiri/timetable.h"

namespace nigiri::loader {

using assistance_idx_t = cista::strong<std::uint32_t, struct assistance_idx_>;

struct assistance_times {
  vecvec<assistance_idx_t, char> names_;
  vector_map<assistance_idx_t, geo::latlng> pos_;
  vector_map<assistance_idx_t, oh::ruleset_t> rules_;
  vector_map<location_idx_t, assistance_idx_t> location_assistance_;
};

assistance_times read_assistance(std::string_view);

}  // namespace nigiri::loader