#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

enum class transfer_type : std::uint8_t {
  kDefaultCost = 0,
  kNotPermitted = 1,
  kPermittedWithCost = 2,
  kPermittedWithFareProduct = 3
};

struct fare_transfer_rule {
  std::string_view from_leg_group_id_;
  std::string_view to_leg_group_id_;
  transfer_type transfer_type_{transfer_type::kDefaultCost};
  std::optional<std::string_view> transfer_fare_product_id_;
  std::optional<double> transfer_amount_;
  std::optional<unsigned> transfer_count_;
};

std::vector<fare_transfer_rule> read_fare_transfer_rules(std::string_view);

}  // namespace nigiri::loader::gtfs::fares