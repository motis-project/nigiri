#pragma once

namespace nigiri::routing {

enum class location_match_mode {
  kExact,  // only use exactly the specified location
  kOnlyChildren,  // use also children (tracks at this location)
  kEquivalent  // use equivalent locations (includes children)
};

}  // namespace nigiri::routing
