#pragma once

#include <string_view>
#include <vector>
#include "line.h"
#include "scheduled_stop_point.h"

namespace nigiri::loader::netex {

struct stop_point_in_journey {
  std::string_view id;
  scheduled_stop_point ssp;
  bool for_alighting;
  bool for_boarding;
  std::string_view front_display_text;  // DestinationDisplayView->FrontText
};

struct service_journey_pattern {
  std::string_view id;
  std::string_view line_name;
  std::vector<stop_point_in_journey> points_in_sequence;
  std::string_view start_point_id;
  std::string_view end_point_id;
};

void parse_journey_patterns(
    const pugi::xml_document& doc,
    hash_map<std::string_view, service_journey_pattern>& journeys_map);
}  // namespace nigiri::loader::netex