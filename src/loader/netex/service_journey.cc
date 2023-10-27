#include "nigiri/loader/netex/service_journey.h"

namespace nigiri::loader::netex {

void parse_service_journeys(timetable& tt, const pugi::xml_document& doc) {
  for (auto const& sj : doc.select_nodes("//ServiceJourney")) {
    auto const sj_node = sj.node();
    auto const id = sj_node.attribute("id").value();
    auto const transport_mode = sj_node.child("TransportMode").text().get();
    auto const departure_time = sj_node.child("DepartureTime").text().get();
    auto const journey_duration = sj_node.child("JourneyDuration").text().get();
    std::cout << id << transport_mode << departure_time << journey_duration
              << tt.n_locations()
              << std::endl;  // just to have compiler stop crying
    // tt.register_route(stop_seq, clasz_sections);
  }
}
}  // namespace nigiri::loader::netex