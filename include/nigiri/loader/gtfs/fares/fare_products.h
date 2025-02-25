#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

struct fare_product {
  std::string_view fare_product_id_;
  std::optional<std::string_view> fare_product_name_;
  double amount_{0.0};
  std::string_view currency_;
  std::optional<std::string_view> rider_category_id_;
  std::optional<std::string_view> timeframe_id_;
  std::optional<std::string_view> fare_media_id_;
};

std::vector<fare_product> read_fare_products(std::string_view);

}  // namespace nigiri::loader::gtfs::fares