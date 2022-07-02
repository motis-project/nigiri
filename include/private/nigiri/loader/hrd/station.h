#pragma once

#include <string_view>

#include <nigiri/section_db.h>

namespace nigiri::loader::hrd {

hash_map<int, location> parse_stations(
    std::string_view station_file_content,
    std::string_view station_coord_file_content,
    std::string_view timezones_file_content);

}  // namespace nigiri::loader::hrd
