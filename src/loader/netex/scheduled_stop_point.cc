#include "nigiri/loader/netex/scheduled_stop_point.h"

namespace nigiri::loader::netex {

void read_scheduled_stop_points(
    const pugi::xml_document& doc,
    hash_map<std::string_view, scheduled_stop_point>& stops_map) {
  for (const auto& ssp : doc.select_nodes("//ScheduledStopPoint")) {
    auto id = ssp.node().attribute("id").value();
    auto short_name = ssp.node().child("PublicCode").text().get();
    auto stop_type = ssp.node().child("StopType").text().get();

    stops_map.insert(
        std::make_pair(id, scheduled_stop_point{id, short_name, stop_type}));
  }
}
}  // namespace nigiri::loader::netex