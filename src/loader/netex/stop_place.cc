#include "nigiri/loader/netex/stop_place.h"
#include "utl/parser/arg_parser.h"

using namespace nigiri;

namespace nigiri::loader::netex {

void read_stop_places(const pugi::xml_document& doc,
                      hash_map<std::string_view, stop_place>& stop_map) {
  // Step 1: Get to site frame
  for (auto const s : doc.select_nodes("//StopPlace")) {

    auto id = s.node().attribute("id").value();
    auto name = s.node().child("Name").text().get();
    auto longitude = s.node().select_node("//Longitude").node().text().get();
    auto latitude = s.node().select_node("//Latitude").node().text().get();
    geo::latlng coords{utl::parse<double>(latitude),
                       utl::parse<double>(longitude)};

    hash_map<std::string_view, quay> quay_map;
    for (auto const q : s.node().select_nodes("//Quay")) {
      auto q_id = q.node().attribute("id").value();
      auto q_name = q.node().child("Name").text().get();
      auto q_longitude =
          q.node().select_node("//Longitude").node().text().get();
      auto q_latitude = q.node().select_node("//Latitude").node().text().get();
      geo::latlng q_coords{utl::parse<double>(q_latitude),
                           utl::parse<double>(q_longitude)};

      quay_map.insert(std::make_pair<std::string_view, quay>(
          q_id, {q_id, q_name, q_coords}));
    }
    // TODO for later: Check whether level is important or not
    stop_map.insert(std::make_pair<std::string_view, stop_place>(
        id, {id, name, coords, quay_map}));
  }
  // Step 2: Read in the stop places & quays and add them if not existent yet
}
}  // namespace nigiri::loader::netex