//
// Created by mirko on 10/11/23.
//

#include "nigiri/loader/netex/route_operator.h"
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/tz_map.h"

namespace nigiri::loader::netex {
/*
 * Registers providers in the timetable and creates a map from provider_id to
 * name
 */

void read_resource_frame(
    timetable& tt,
    const pugi::xml_document& doc,
    hash_map<std::string_view, provider_idx_t>& operatorMap) {

  for (auto const& responsible_operator : doc.select_nodes("//Operator")) {
    auto const short_name =
        responsible_operator.node().child("ShortName").text().get();
    auto const name = responsible_operator.node().child("Name").text().get();
    auto const op = route_operator{name, short_name};
    auto const op_id = responsible_operator.node().attribute("id").value();

    timezone_idx_t dummy;  // todo how to retrieve the timezone?
    auto const provider_idx = tt.register_provider({name, short_name, dummy});
    operatorMap.try_emplace(op_id, provider_idx);
  }
}

}  // namespace nigiri::loader::netex
