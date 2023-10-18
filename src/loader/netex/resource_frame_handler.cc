//
// Created by mirko on 10/11/23.
//

#include "resource_frame_handler.h"
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/tz_map.h"

namespace nigiri::loader::netex {
/*
 * Registers the provider in the timetable, and should gradually build up a map
 * from agency names to the index (todo not implemented yet)
 */
void processResourceFrame(timetable& t,
                          const pugi::xml_node& frame,
                          gtfs::tz_map& timezones,
                          gtfs::agency_map_t& agencyMap) {
  auto const operatorAgency = frame.child("organisations").child("Operator");

  auto const provider_id = operatorAgency.attribute("id").value();
  std::cout << "Provider id: " << provider_id << "\n";

  auto const provider_name = operatorAgency.child("Name").text().get();
  auto tz_name = frame.parent()
                     .parent()
                     .child("FrameDefaults")
                     .child("DefaultLocale")
                     .child("TimeZone")
                     .text()
                     .get();

  std::cout << "Timezone name:  " << tz_name << "\n";
  auto const tz = gtfs::get_tz_idx(t, timezones, tz_name);
  t.register_provider({provider_id, provider_name, tz});

  // Step 2: Fill in the agency map
  std::cout << "Dummy: Size of agency map: " << agencyMap.size() << "\n";
}
}  // namespace nigiri::loader::netex
