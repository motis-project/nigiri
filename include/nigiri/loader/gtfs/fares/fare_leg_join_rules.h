#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

struct fare_leg_join_rule {
  std::string_view fare_leg_rule_id_;
  std::optional<std::string_view> from_leg_price_group_id_;
  std::optional<std::string_view> to_leg_price_group_id_;
  std::optional<unsigned> fare_leg_rule_sequence_;
};

std::vector<fare_leg_join_rule> read_fare_leg_join_rules(std::string_view);

}  // namespace nigiri::loader::gtfs::fares