#pragma once

#include "nigiri/types.h"
#include <pugixml.hpp>
#include <string_view>

namespace nigiri::loader::netex {
struct scheduled_stop_point {
  std::string_view id;
  std::string_view name;
  // std::string_view public_code;
  std::string_view stop_type;
};

void read_scheduled_stop_points(
    const pugi::xml_document& doc,
    hash_map<std::string_view, scheduled_stop_point>& stops_map);
}  // namespace nigiri::loader::netex