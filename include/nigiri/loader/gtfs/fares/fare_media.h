#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

enum class fare_media_type : std::uint8_t { 
  kPhysical = 0,
  kVirtual = 1
};

enum class fare_media_restriction : std::uint8_t {
  kNone = 0,
  kReserveFirstUse = 1,
  kReserveBeforeUse = 2
};

struct fare_medium {
  std::string_view fare_media_id_;
  std::optional<std::string_view> fare_media_name_;
  fare_media_type media_type_{fare_media_type::kPhysical};
  std::optional<fare_media_restriction> restrictions_{std::nullopt};
};

std::vector<fare_medium> read_fare_media(std::string_view);

}  // namespace nigiri::loader::gtfs::fares