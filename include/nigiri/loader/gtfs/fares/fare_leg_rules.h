#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

enum class contains_area_type : std::uint8_t {
  kAny = 0,
  kAll = 1
};

struct fare_leg_rule {
  std::string_view fare_leg_rule_id_;
  std::string_view fare_product_id_;
  std::optional<std::string_view> leg_group_id_;
  std::optional<std::string_view> network_id_;
  std::optional<std::string_view> from_area_id_;
  std::optional<std::string_view> to_area_id_;
  std::optional<std::string_view> route_id_;
  std::optional<std::string_view> contains_area_id_;
  std::optional<contains_area_type> contains_area_type_{std::nullopt};
  std::optional<std::string_view> contains_route_id_;
};

std::vector<fare_leg_rule> read_fare_leg_rules(std::string_view);

}  // namespace nigiri::loader::gtfs::fares