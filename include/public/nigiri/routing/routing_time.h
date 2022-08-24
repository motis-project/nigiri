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
  constexpr routing_time(timetable const& tt, unixtime_t const t)
      : offset_{(t - tt.begin_).count()} {}
  constexpr unixtime_t to_unixtime(timetable const& tt) const {
    return tt.begin_ + offset_ * 1_minutes;
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
  constexpr inline std::int32_t mam() const { return offset_ % 1440; }
  constexpr inline std::pair<day_idx_t, std::int32_t> day_idx_mam() const {
    return {day(), mam()};
  }
  constexpr routing_time operator+(duration_t const& rt) const {
    return routing_time{static_cast<std::int32_t>(offset_ + rt.count())};
  }
  constexpr routing_time operator-(duration_t const& rt) const {
    return routing_time{static_cast<std::int32_t>(offset_ - rt.count())};
  }
  friend std::ostream& operator<<(std::ostream& out, routing_time const t) {
    if (t == min()) {
      return out << "MIN";
    } else if (t == max()) {
      return out << "MAX";
    } else {
      return out << duration_t{t.t()};
    }
  }
  std::int32_t offset_;  // minutes since timetable begin
};

}  // namespace nigiri::routing