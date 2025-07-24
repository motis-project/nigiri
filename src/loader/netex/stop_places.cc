#include "nigiri/loader/netex/stop_places.h"

#include "utl/get_or_create.h"
#include "utl/parser/arg_parser.h"
#include "utl/to_vec.h"

#include <algorithm>

namespace nigiri::loader::netex {

geo::latlng parse_location(pugi::xml_node const& loc_node) {
  if (!loc_node) {
    return {};
  }
  return {std::clamp(utl::parse<double>(loc_node.child_value("Latitude")),
                     -90.0, 90.0),
          std::clamp(utl::parse<double>(loc_node.child_value("Longitude")),
                     -180.0, 180.0)};
}

quay parse_quay(pugi::xml_node const& quay_node) {
  auto const id = quay_node.attribute("id").value();
  auto const public_code = quay_node.child("PublicCode");
  auto const name = quay_node.child_value("Name");
  auto const loc =
      parse_location(quay_node.child("Centroid").child("Location"));
  return {.id_ = id,
          .name_ = public_code ? public_code.value() : name,
          .centroid_ = loc};
}

void load_stop_places(netex_data& data,
                      loader_config const& config,
                      source_idx_t src,
                      timetable& tt,
                      pugi::xml_document const& doc) {
  for (auto const& xpn : doc.select_nodes("//SiteFrame/stopPlaces/StopPlace")) {
    auto const sp = xpn.node();
    auto const id = sp.attribute("id").value();
    auto const name = sp.child_value("Name");
    auto const centroid = sp.child("Centroid");
    auto const loc = parse_location(centroid.child("Location"));
    auto const description = sp.child_value("Description");

    utl::get_or_create(data.stop_places_, id, [&]() {
      return stop_place{.id_ = id,
                        .name_ = name,
                        .description_ = description,
                        .centroid_ = loc,
                        .quays_ = utl::to_vec(sp.select_nodes("quays/Quay"),
                                              [&](auto const& xpqn) {
                                                return parse_quay(xpqn.node());
                                              })};
    });
  }
}

}  // namespace nigiri::loader::netex
