#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

struct stop_area {
  std::string_view stop_id_;
  std::string_view area_id_;
};

std::vector<stop_area> read_stop_areas(std::string_view);

}  // namespace nigiri::loader::gtfs::fares