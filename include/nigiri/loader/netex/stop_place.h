#pragma once

#include "nigiri/timetable.h"
#include <pugixml.hpp>

namespace nigiri::loader::netex {

struct quay {
  std::string_view id;
  std::string_view name;
  geo::latlng coords;
  // information about level
};

struct stop_place {
  std::string_view id;
  std::string_view name;
  geo::latlng coords;
  hash_map<std::string_view, quay> quays;
};

void read_stop_places(const pugi::xml_document& doc,
                      hash_map<std::string_view, stop_place>& stop_map);
}  // namespace nigiri::loader::netex