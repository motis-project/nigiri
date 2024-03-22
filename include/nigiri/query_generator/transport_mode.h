#pragma once

#include <cstdint>

namespace nigiri::query_generator {

enum class mode_id : std::uint16_t { kStation, kNumModeIds };

struct transport_mode {
  mode_id mode_id_;
  std::uint16_t speed_;
  std::uint32_t range_;
};

constexpr static transport_mode station_mode{mode_id::kStation, 0U, 0U};

}  // namespace nigiri::query_generator