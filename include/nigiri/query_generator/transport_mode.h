#pragma once

#include <cstdint>
#include "nigiri/types.h"

namespace nigiri::query_generation {

struct transport_mode {
  constexpr std::uint32_t range() const {
    return speed_ * max_duration_;
  }  // [m]

  friend std::ostream& operator<<(std::ostream& out, transport_mode const& tm) {
    out << "(id: " << tm.mode_id_ << ", speed: " << tm.speed_
        << ", max_duration: " << tm.max_duration_ << ")";
    return out;
  }

  friend bool operator==(transport_mode const& a,
                         transport_mode const& b) = default;

  transport_mode_id_t mode_id_;
  std::uint16_t speed_;  // [m/minute]
  std::uint16_t max_duration_;  // [minutes]
};

constexpr auto const kWalk =
    transport_mode{.mode_id_ = 1, .speed_ = 50U, .max_duration_ = 15U};
constexpr auto const kBicycle =
    transport_mode{.mode_id_ = 2, .speed_ = 200U, .max_duration_ = 15U};
constexpr auto const kCar =
    transport_mode{.mode_id_ = 3, .speed_ = 800U, .max_duration_ = 15U};

constexpr std::optional<transport_mode> to_transport_mode(
    std::string_view const& str) {
  if (str == "walk") {
    return kWalk;
  } else if (str == "bicycle") {
    return kBicycle;
  } else if (str == "car") {
    return kCar;
  }
  return std::nullopt;
}

constexpr std::optional<std::string> to_string(transport_mode const& tm) {
  using namespace nigiri::query_generation;
  if (tm == kWalk) {
    return "walk";
  } else if (tm == kBicycle) {
    return "bicycle";
  } else if (tm == kCar) {
    return "car";
  }
  return std::nullopt;
}

}  // namespace nigiri::query_generation