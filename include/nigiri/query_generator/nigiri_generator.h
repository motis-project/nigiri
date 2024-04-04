#pragma once

#include "nigiri/query_generator/generator.h"
#include "nigiri/query_generator/query_factory.h"
#include "geo/point_rtree.h"

namespace nigiri::query_generation {

struct nigiri_generator : generator {
  explicit nigiri_generator(timetable const& tt,
                            generator_settings const& settings);

  std::optional<routing::query> random_pretrip_query();
  std::optional<routing::query> random_ontrip_query();

private:
  void add_offsets_for_pos(std::vector<routing::offset>&,
                           geo::latlng const&,
                           query_generation::transport_mode const&);

  query_factory qf_;
  geo::point_rtree locations_rtree_;
};

}  // namespace nigiri::query_generation