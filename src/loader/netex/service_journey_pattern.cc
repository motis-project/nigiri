#include "nigiri/loader/netex/service_journey_pattern.h"
#include "nigiri/loader/netex/line.h"

namespace nigiri::loader::netex {

void parse_journey_patterns(
    const pugi::xml_document& doc,
    hash_map<std::string_view, service_journey_pattern>& journeys_map,
    hash_map<std::string_view, line>& lines_map,
    hash_map<std::string_view, scheduled_stop_point>& ssp_map) {
  for (const auto& sjp : doc.select_nodes("//ServiceJourneyPattern")) {
    auto id = sjp.node().attribute("id").value();
    auto line_name =
        lines_map[sjp.node().select_node("//LineRef").node().value()].name;

    auto start_point_id = sjp.node()
                              .select_node("//StartPointInPatternRef")
                              .node()
                              .attribute("ref")
                              .value();
    auto end_point_id = sjp.node()
                            .select_node("//EndPointInPatternRef")
                            .node()
                            .attribute("ref")
                            .value();

    std::vector<stop_point_in_journey> points_in_sequence;

    for (const auto& stop_point_xml :
         sjp.node().select_nodes("StopPointInJourneyPattern")) {
      auto stop_point_id = stop_point_xml.node().attribute("id").value();
      auto ssp = ssp_map[stop_point_xml.node()
                             .child("ScheduledStopPointRef")
                             .attribute("ref")
                             .value()];
      // todo check whether found or not
      auto for_alighting = static_cast<bool>(
          stop_point_xml.node().child("ForAlighting").text().get());
      auto for_boarding = static_cast<bool>(
          stop_point_xml.node().child("ForBoarding").text().get());

      auto front_display_text = stop_point_xml.node()
                                    .child("DestinationDisplayView")
                                    .child("FrontText")
                                    .text()
                                    .get();

      stop_point_in_journey stop_point{stop_point_id, ssp, for_alighting,
                                       for_boarding, front_display_text};
      points_in_sequence.emplace_back(stop_point);
    }

    journeys_map.insert(std::make_pair(
        id, service_journey_pattern{id, line_name, points_in_sequence,
                                    start_point_id, end_point_id}));
  }
}
}  // namespace nigiri::loader::netex
