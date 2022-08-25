#pragma once

#include <iosfwd>

#include "geo/latlng.h"

#include "nigiri/common/it_range.h"
#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable;

struct location {
  friend std::ostream& operator<<(std::ostream&, location const&);
  location(timetable const&, location_idx_t);
  location(string const& id,
           string const& name,
           geo::latlng pos,
           source_idx_t,
           location_type,
           osm_node_id_t,
           location_idx_t,
           timezone_idx_t,
           it_range<vector<location_idx_t>::iterator> equivalences,
           it_range<vector<footpath>::iterator> footpaths_in,
           it_range<vector<footpath>::iterator> footpaths_out);
  string const& id_;
  string const& name_;
  geo::latlng pos_;
  source_idx_t src_;
  location_type type_;
  osm_node_id_t osm_id_;
  location_idx_t parent_;
  timezone_idx_t timezone_idx_;
  it_range<vector<location_idx_t>::iterator> equivalences_;
  it_range<vector<footpath>::iterator> footpaths_out_, footpaths_in_;
};

}  // namespace nigiri