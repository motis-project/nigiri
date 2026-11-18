#pragma once

#include <compare>
#include <cstdint>

namespace nigiri {

// Accessibility properties of a footpath that follows a station graph
// (GTFS pathways, ISA Umsteigebauwerke, OSM). All values aggregate over the
// edges of the route the footpath follows: flags are OR-ed, the slope is the
// steepest edge, the width is the narrowest edge, fixed times are summed.
//
// slope_ / width_: bits 0..6 = value over the edges with a known value,
// bit 7 (kUnknown) = at least one edge without a value.
struct footpath_restriction {
  // Path kinds (user switches "avoid ...").
  static constexpr auto const kStairs = std::uint8_t{1U << 0U};
  static constexpr auto const kEscalator = std::uint8_t{1U << 1U};
  static constexpr auto const kElevator = std::uint8_t{1U << 2U};
  static constexpr auto const kRamp = std::uint8_t{1U << 3U};
  static constexpr auto const kStep = std::uint8_t{1U << 4U};

  // Explicit bans per user group (ISA "Gesperrt fuer ...").
  static constexpr auto const kBanWheelchair = std::uint8_t{1U << 5U};
  static constexpr auto const kBanWheelchairAssisted = std::uint8_t{1U << 6U};
  static constexpr auto const kBanWalker = std::uint8_t{1U << 7U};

  static constexpr auto const kUnknown = std::uint8_t{1U << 7U};
  static constexpr auto const kValueMask = std::uint8_t{0x7FU};

  // Slope in 0.2 % steps (0..25.4 %), rounded up.
  static constexpr auto const kSlopeStepsPerPercent = 5.0;
  // Width in 5 cm steps (0..6.35 m), rounded down; kNoWidth = no edge with a
  // known width.
  static constexpr auto const kWidthStepsPerMeter = 20.0;
  static constexpr auto const kNoWidth = kValueMask;

  // Unit of the fixed (not scaled by the walking speed factor) part.
  static constexpr auto const kFixedSecondsUnit = 10U;

  static constexpr std::uint8_t to_slope(double const percent) {
    auto const steps = percent * kSlopeStepsPerPercent - 1E-6;
    if (steps <= 0.0) {
      return 0U;
    }
    auto const truncated = static_cast<unsigned>(steps);
    auto const up =
        static_cast<double>(truncated) < steps ? truncated + 1U : truncated;
    return up >= kValueMask ? kValueMask : static_cast<std::uint8_t>(up);
  }

  static constexpr std::uint8_t to_width(double const meters) {
    auto const steps = meters * kWidthStepsPerMeter + 1E-6;
    return steps <= 0.0 ? std::uint8_t{0U}
           : steps >= static_cast<double>(kNoWidth)
               ? kNoWidth
               : static_cast<std::uint8_t>(steps);
  }

  static constexpr std::uint8_t to_fixed(unsigned const seconds) {
    auto const units = (seconds + kFixedSecondsUnit - 1U) / kFixedSecondsUnit;
    return units >= 255U ? std::uint8_t{255U}
                         : static_cast<std::uint8_t>(units);
  }

  constexpr std::uint8_t slope() const { return slope_ & kValueMask; }
  constexpr std::uint8_t width() const { return width_ & kValueMask; }
  constexpr bool has_unknown_slope() const { return (slope_ & kUnknown) != 0U; }
  constexpr bool has_unknown_width() const { return (width_ & kUnknown) != 0U; }
  constexpr unsigned fixed_seconds() const {
    return static_cast<unsigned>(fixed_) * kFixedSecondsUnit;
  }

  // Restriction of the concatenation of two routes.
  friend constexpr footpath_restriction operator+(
      footpath_restriction const& a, footpath_restriction const& b) {
    auto const slope = a.slope() > b.slope() ? a.slope() : b.slope();
    auto const width = a.width() < b.width() ? a.width() : b.width();
    auto const fixed = static_cast<unsigned>(a.fixed_) + b.fixed_;
    return {.flags_ = static_cast<std::uint8_t>(a.flags_ | b.flags_),
            .slope_ = static_cast<std::uint8_t>(
                slope | ((a.slope_ | b.slope_) & kUnknown)),
            .width_ = static_cast<std::uint8_t>(
                width | ((a.width_ | b.width_) & kUnknown)),
            .fixed_ = static_cast<std::uint8_t>(fixed >= 255U ? 255U : fixed)};
  }

  // True if a is at least as good as b for every user: no additional path
  // kinds, not steeper, not narrower, no additional unknowns.
  friend constexpr bool is_less_restricted(footpath_restriction const& a,
                                           footpath_restriction const& b) {
    return (a.flags_ & ~b.flags_) == 0U && a.slope() <= b.slope() &&
           a.width() >= b.width() && (a.slope_ & ~b.slope_ & kUnknown) == 0U &&
           (a.width_ & ~b.width_ & kUnknown) == 0U;
  }

  friend constexpr bool operator==(footpath_restriction const&,
                                   footpath_restriction const&) = default;
  friend constexpr auto operator<=>(footpath_restriction const&,
                                    footpath_restriction const&) = default;

  // Default: identity of operator+ (empty route).
  std::uint8_t flags_{0U};
  std::uint8_t slope_{0U};
  std::uint8_t width_{kNoWidth};
  std::uint8_t fixed_{0U};
};

static_assert(sizeof(footpath_restriction) == 4U);

}  // namespace nigiri
