#pragma once

#include <optional>

#include "nigiri/footpath_restriction.h"
#include "nigiri/types.h"

namespace nigiri::routing {

// Per-query accessibility requirements for footpaths with a restriction
// record. Thresholds use the units of footpath_restriction. strict: edges
// without a value fail an active threshold, otherwise they pass (DELFI
// "tolerant"). Known values always have to meet the threshold. penalty_[i]:
// minutes added to a footpath that contains path kind 1 << i (kStairs ..
// kStep).
struct footpath_restriction_settings {
  static constexpr auto const kNoSlopeLimit = std::uint8_t{255U};
  static constexpr auto const kNoWidthLimit = std::uint8_t{0U};
  static constexpr auto const kNPenalties = 5U;

  friend constexpr bool operator==(footpath_restriction_settings const&,
                                   footpath_restriction_settings const&) =
      default;

  constexpr bool is_default() const {
    if (forbidden_ != 0U || max_slope_ != kNoSlopeLimit ||
        min_width_ != kNoWidthLimit) {
      return false;
    }
    for (auto const p : penalty_) {
      if (p != 0U) {
        return false;
      }
    }
    return true;
  }

  // Minutes added to a footpath with this record.
  constexpr int get_penalty(footpath_restriction const& r) const {
    auto penalty = 0;
    for (auto i = 0U; i != kNPenalties; ++i) {
      if ((r.flags_ & (1U << i)) != 0U) {
        penalty += penalty_[i];
      }
    }
    return penalty;
  }

  std::uint8_t forbidden_{0U};
  std::uint8_t max_slope_{kNoSlopeLimit};
  std::uint8_t min_width_{kNoWidthLimit};
  bool strict_{false};
  std::uint8_t penalty_[kNPenalties]{};
};

inline constexpr bool is_usable(footpath_restriction const& r,
                                footpath_restriction_settings const& s) {
  if ((r.flags_ & s.forbidden_) != 0U) {
    return false;
  }
  if (s.max_slope_ != footpath_restriction_settings::kNoSlopeLimit &&
      (r.slope() > s.max_slope_ || (s.strict_ && r.has_unknown_slope()))) {
    return false;
  }
  if (s.min_width_ != footpath_restriction_settings::kNoWidthLimit &&
      (r.width() < s.min_width_ || (s.strict_ && r.has_unknown_width()))) {
    return false;
  }
  return true;
}

struct transfer_time_settings {
  bool operator==(transfer_time_settings const& o) const {
    return default_ == o.default_ && restrictions_ == o.restrictions_ &&
           (default_ ||
            std::tie(min_transfer_time_, additional_time_, factor_) ==
                std::tie(o.min_transfer_time_, o.additional_time_, o.factor_));
  }

  bool default_{true};
  duration_t min_transfer_time_{0};
  duration_t additional_time_{0};
  float factor_{1.0F};
  footpath_restriction_settings restrictions_{};
};

template <typename T>
inline constexpr T adjusted_transfer_time(
    transfer_time_settings const& settings, T const duration) {
  if (settings.default_) {
    return duration;
  } else {
    return static_cast<T>(settings.additional_time_.count()) +
           std::max(
               static_cast<T>(settings.min_transfer_time_.count()),
               static_cast<T>(static_cast<float>(duration) * settings.factor_));
  }
}

// Duration in minutes of a footpath with restriction record, or -1 if the
// query's restriction settings forbid it. The factor only scales the walking
// part; the fixed part (elevator, escalator rides) and the penalties are added
// unscaled.
inline constexpr int adjusted_transfer_time(transfer_time_settings const& s,
                                            int const duration,
                                            footpath_restriction const& r) {
  if (!is_usable(r, s.restrictions_)) {
    return -1;
  }
  auto const penalty = s.restrictions_.get_penalty(r);
  if (s.default_) {
    return duration + penalty;
  }
  auto const fixed = static_cast<int>((r.fixed_seconds() + 59U) / 60U);
  auto const fixed_part = fixed < duration ? fixed : duration;
  auto const scaled =
      static_cast<int>(static_cast<float>(duration - fixed_part) * s.factor_) +
      fixed_part;
  auto const min = static_cast<int>(s.min_transfer_time_.count());
  return static_cast<int>(s.additional_time_.count()) +
         (scaled > min ? scaled : min) + penalty;
}

template <typename Rep>
inline constexpr std::chrono::duration<Rep, std::ratio<60>>
adjusted_transfer_time(
    transfer_time_settings const& settings,
    std::chrono::duration<Rep, std::ratio<60>> const duration) {
  if (settings.default_) {
    return duration;
  } else {
    return std::chrono::duration<Rep, std::ratio<60>>{
        static_cast<Rep>(settings.additional_time_.count()) +
        std::max(static_cast<Rep>(settings.min_transfer_time_.count()),
                 static_cast<Rep>(static_cast<float>(duration.count()) *
                                  settings.factor_))};
  }
}

}  // namespace nigiri::routing
