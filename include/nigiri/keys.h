#pragma once

#include "fmt/core.h"
#include "geo/latlng.h"

#include "types.h"

namespace nigiri {

inline string to_location_key(geo::latlng const coord) {
  return string{fmt::format("{}:{}", std::to_string(coord.lat_),
                            std::to_string(coord.lng_))};
}

}  // namespace nigiri
