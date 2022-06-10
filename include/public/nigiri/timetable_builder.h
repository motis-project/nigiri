#pragma once

#include "nigiri/section_db.h"
#include "nigiri/timetable.h"

namespace nigiri {

struct builder_location {
  location_id id_;
  std::string name_;
  geo::latlng pos_;
  location_type type_;
  osm_node_id_t osm_id_;
  std::vector<builder_location*> parents_, children_;
};

}  // namespace nigiri
