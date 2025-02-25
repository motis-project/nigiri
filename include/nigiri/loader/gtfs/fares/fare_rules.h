#pragma once

#include <string_view>
#include <vector>
#include <optional>

namespace nigiri::loader::gtfs::fares {

struct fare_rule {
  std::string_view fare_id_;
  std::optional<std::string_view> route_id_;
  std::optional<std::string_view> origin_id_;
  std::optional<std::string_view> destination_id_;
  std::optional<std::string_view> contains_id_;
};

std::vector<fare_rule> read_fare_rules(std::string_view file_content);

}  // namespace nigiri::loader::gtfs::fares