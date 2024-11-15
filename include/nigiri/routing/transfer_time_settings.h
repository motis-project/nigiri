#pragma once

#include "nigiri/types.h"

namespace nigiri::routing {

struct transfer_time_settings {
  bool operator==(transfer_time_settings const& o) const {
    return default_ == o.default_ ||
           std::tie(min_transfer_time_, disposable_time_, factor_) ==
               std::tie(o.min_transfer_time_, disposable_time_, o.factor_);
  }

  bool default_{true};
  duration_t min_transfer_time_{0};
  duration_t disposable_time_{0};
  float factor_{1.0F};
};

template <typename T>
inline T adjusted_transfer_time(transfer_time_settings const& settings,
                                T const duration) {
  if (settings.default_) {
    return duration;
  } else {
    return static_cast<T>(settings.disposable_time_.count()) +
           std::max(
               static_cast<T>(settings.min_transfer_time_.count()),
               static_cast<T>(static_cast<float>(duration) * settings.factor_));
  }
}

template <typename Rep>
inline std::chrono::duration<Rep, std::ratio<60>> adjusted_transfer_time(
    transfer_time_settings const& settings,
    std::chrono::duration<Rep, std::ratio<60>> const duration) {
  if (settings.default_) {
    return duration;
  } else {
    return std::chrono::duration<Rep, std::ratio<60>>{
        static_cast<Rep>(settings.disposable_time_.count()) +
        std::max(static_cast<Rep>(settings.min_transfer_time_.count()),
                 static_cast<Rep>(static_cast<float>(duration.count()) *
                                  settings.factor_))};
  }
}

}  // namespace nigiri::routing
