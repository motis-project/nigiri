#pragma once

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "utl/get_or_create.h"

#include "oh/contains.h"
#include "oh/parser.h"

#include "nigiri/timetable.h"

namespace nigiri::loader {

using assistance_idx_t = cista::strong<std::uint32_t, struct assistance_idx_>;

struct assistance_times {
  bool is_available(timetable const& tt,
                    location_idx_t const l,
                    oh::local_minutes const t) {
    static auto const kRule = oh::parse("08:00-20:00");
    return contains(kRule, t);
    /*
    auto const a = utl::get_or_create(cache_, l, [&]() {
      auto const in_radius =
          rtree_.in_radius(tt.locations_.coordinates_[l], 800);
      return in_radius.empty() ? assistance_idx_t::invalid()
                               : assistance_idx_t{in_radius.front()};
    });
    return a != assistance_idx_t::invalid() && oh::contains(rules_[a], t);
     */
  }

  vecvec<assistance_idx_t, char> names_;
  vector_map<assistance_idx_t, geo::latlng> pos_;
  vector_map<assistance_idx_t, oh::ruleset_t> rules_;
  hash_map<location_idx_t, assistance_idx_t> cache_;
  geo::point_rtree rtree_;
};

assistance_times read_assistance(std::string_view);

}  // namespace nigiri::loader