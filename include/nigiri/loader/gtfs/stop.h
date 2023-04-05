#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct stop {
  void compute_close_stations(geo::point_rtree const& stop_rtree);
  std::set<stop*> get_metas(std::vector<stop*> const& stops);

  std::string id_;
  std::string name_;
  geo::latlng coord_;
  std::string timezone_;
  std::set<stop*> same_name_, parents_, children_;
  std::vector<unsigned> close_;
};

using stop_map = hash_map<std::string, std::unique_ptr<stop>>;

stop_map read_stops(std::string_view file_content);

}  // namespace nigiri::loader::gtfs
