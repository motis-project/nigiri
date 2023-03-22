#pragma once

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct routing_time {
  constexpr routing_time() = default;
  constexpr routing_time(day_idx_t const d, minutes_after_midnight_t const mam)
      : offset_{to_idx(d) * 1440 + mam.count()} {}
  constexpr explicit routing_time(std::int32_t const offset)
      : offset_{offset} {}
  routing_time(timetable const& tt, unixtime_t const t)
      : offset_{(t - tt.begin()).count()} {}
  constexpr unixtime_t to_unixtime(timetable const& tt) const {
    return tt.date_range_.from_ + offset_ * 1_minutes;
  }
  constexpr static routing_time max() {
    return routing_time{std::numeric_limits<std::int32_t>::max()};
  }
  constexpr static routing_time min() {
    return routing_time{std::numeric_limits<std::int32_t>::min()};
  }
  friend auto operator<=>(routing_time const&, routing_time const&) = default;
  friend bool operator==(routing_time const&, routing_time const&) = default;
  friend bool operator!=(routing_time const&, routing_time const&) = default;
  friend bool operator<=(routing_time const&, routing_time const&) = default;
  friend bool operator<(routing_time const&, routing_time const&) = default;
  friend bool operator>=(routing_time const&, routing_time const&) = default;
  friend bool operator>(routing_time const&, routing_time const&) = default;
  constexpr inline std::int32_t t() const { return offset_; }
  constexpr inline day_idx_t day() const { return day_idx_t{offset_ / 1440}; }
  constexpr inline minutes_after_midnight_t mam() const {
    return minutes_after_midnight_t{offset_ % 1440};
  }
  constexpr inline std::pair<day_idx_t, minutes_after_midnight_t> day_idx_mam()
      const {
    return {day(), mam()};
  }
  constexpr routing_time operator+(duration_t const& rt) const {
    return routing_time{static_cast<std::int32_t>(offset_ + rt.count())};
  }
  constexpr routing_time operator-(duration_t const& rt) const {
    return routing_time{static_cast<std::int32_t>(offset_ - rt.count())};
  }
  friend constexpr duration_t operator-(routing_time const a,
                                        routing_time const b) {
    return duration_t{b.offset_ - a.offset_};
  }
  friend std::ostream& operator<<(std::ostream& out, routing_time const t) {
    if (t == min()) {
      return out << "MIN";
    } else if (t == max()) {
      return out << "MAX";
    } else {
      return out << i32_minutes{t.t()};
    }
  }
  std::int32_t offset_;  // minutes since timetable begin
};

template <direction SearchDir>
inline constexpr auto const kInvalidTime = SearchDir == direction::kForward
                                               ? routing_time::max()
                                               : routing_time::min();

}  // namespace nigiri::routing
