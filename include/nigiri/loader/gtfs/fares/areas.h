#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

enum class area_type : std::uint8_t {
  kStop = 0,
  kZone = 1,
  kRoute = 2
};

struct area {
  std::string_view area_id_;
  std::optional<std::string_view> area_name_;
  std::optional<area_type> area_type_{std::nullopt};
  std::optional<std::string_view> geometry_id_;
};

std::vector<area> read_areas(std::string_view);

}  // namespace nigiri::loader::gtfs::fares