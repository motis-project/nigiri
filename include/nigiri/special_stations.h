#pragma once

#include "nigiri/types.h"

namespace nigiri {

enum class special_station : location_idx_t::value_t {
  kStart,
  kDest,
  kVia0,
  kVia1,
  kVia2,
  kVia3,
  kVia4,
  kVia5,
  kVia6,
  kSpecialStationsSize
};

constexpr location_idx_t get_special_station(special_station const x) {
  return location_idx_t{
      static_cast<std::underlying_type_t<special_station>>(x)};
}

constexpr auto const special_stations_names =
    cista::array<std::string_view,
                 static_cast<std::underlying_type_t<special_station>>(
                     special_station::kSpecialStationsSize)>{
        "START", "DEST", "VIA0", "VIA1", "VIA2",
        "VIA3",  "VIA4", "VIA5", "VIA6"};

}  // namespace nigiri
