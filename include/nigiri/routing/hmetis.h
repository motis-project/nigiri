#pragma once

#include <ostream>

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

void write_hmetis_file(std::ostream& out, timetable const&);
void hmetis_out_to_geojson(std::string_view in,
                           std::ostream& out,
                           timetable const&);

}  // namespace nigiri::routing