#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

struct route_network {
  std::string_view route_id_;
  std::string_view network_id_;
};

std::vector<route_network> read_route_networks(std::string_view);

}  // namespace nigiri::loader::gtfs::fares