#pragma once

#include <cstdint>
#include "nigiri/types.h"

namespace nigiri::query_generation {

struct transport_mode {
  constexpr std::uint32_t range() const {
    return speed_ * max_duration_ * 60U;
  }  // [m]

  std::int32_t mode_id_;
  std::uint16_t speed_;  // [m/s]
  std::uint16_t max_duration_;  // [minutes]
};

constexpr static transport_mode kWalk{
    .mode_id_ = 1, .speed_ = 1U, .max_duration_ = 15};
constexpr static transport_mode kBicycle{
    .mode_id_ = 2, .speed_ = 3U, .max_duration_ = 15};
constexpr static transport_mode kCar{
    .mode_id_ = 3, .speed_ = 10, .max_duration_ = 15};

}  // namespace nigiri::query_generation