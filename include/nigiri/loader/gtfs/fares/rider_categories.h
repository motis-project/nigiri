#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

struct rider_category {
  std::string_view rider_category_id_;
  std::optional<std::string_view> rider_category_name_;
  std::optional<std::string_view> eligible_for_fare_product_id_;
  std::optional<unsigned> min_age_;
  std::optional<unsigned> max_age_;
};

std::vector<rider_category> read_rider_categories(std::string_view);

}  // namespace nigiri::loader::gtfs::fares