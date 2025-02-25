#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nigiri/loader/dir.h"

namespace nigiri::loader::gtfs::fares {

struct network {
  std::string_view network_id_;
  std::optional<std::string_view> network_name_;
};

std::vector<network> read_networks(std::string_view);

}  // namespace nigiri::loader::gtfs::fares