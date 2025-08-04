#include "nigiri/timetable_metrics.h"

#include <chrono>

#include "utl/enumerate.h"

#include "fmt/format.h"

#include "date/date.h"

namespace nigiri {

struct transport_day_metric {
  std::uint16_t first_ = std::numeric_limits<std::uint16_t>::max();
  std::uint16_t last_ = std::numeric_limits<std::uint16_t>::min();
  std::size_t count_ = 0U;
};

timetable_metrics get_metrics(timetable const& tt) {
  auto const transport_day_metrics =
      [&](bitfield const& traffic_days) -> transport_day_metric {
    auto tdm = transport_day_metric{};
    traffic_days.for_each_set_bit([&](std::uint16_t const day) {
      tdm.first_ = std::min(tdm.first_, day);
      tdm.last_ = std::max(tdm.last_, day);
      ++tdm.count_;
    });
    return tdm;
  };

  auto m = timetable_metrics{};
  m.feeds_.resize(tt.n_sources());

  for (auto const src : tt.locations_.src_) {
    if (src == source_idx_t::invalid()) {
      continue;
    }
    ++m.feeds_[src].locations_;
  }

  // Count regular trips / transports
  // TODO Approach might count duplicates with merged trips
  for (auto const& [trip_id, trip_idx] : tt.trip_id_to_idx_) {
    auto const src = tt.trip_id_src_[trip_id];
    auto days = bitfield{};
    for (auto const& [t, _] : tt.trip_transport_ranges_[trip_idx]) {
      days |= tt.bitfields_[tt.transport_traffic_days_[t]];
    }
    auto const tdm = transport_day_metrics(days);
    ++m.feeds_[src].trips_;
    m.feeds_[src].first_ = std::min(m.feeds_[src].first_, tdm.first_);
    m.feeds_[src].last_ = std::max(m.feeds_[src].last_, tdm.last_);
    m.feeds_[src].transport_days_ += tdm.count_;
  }

  // Count flex transports
  for (auto ft = flex_transport_idx_t{0U}; ft < tt.flex_transport_trip_.size();
       ++ft) {
    auto const trip_idx = tt.flex_transport_trip_[ft];
    auto const tdm = transport_day_metrics(
        tt.bitfields_[tt.flex_transport_traffic_days_[ft]]);
    for (auto const trip_id : tt.trip_ids_[trip_idx]) {
      auto const src = tt.trip_id_src_[trip_id];
      m.feeds_[src].first_ = std::min(m.feeds_[src].first_, tdm.first_);
      m.feeds_[src].last_ = std::max(m.feeds_[src].last_, tdm.last_);
      m.feeds_[src].transport_days_ += tdm.count_;
    }
  }

  return m;
}

std::string to_str(timetable_metrics const& m, timetable const& tt) {
  auto const from = std::chrono::time_point_cast<date::sys_days::duration>(
      tt.internal_interval().from_);
  std::stringstream ss{};
  ss << '[';
  for (auto const [idx, fm] : utl::enumerate(m.feeds_)) {
    if (idx > 0U) {
      ss << ',';
    }
    ss << fmt::format(
        R"({{"idx":{},"firstDay":"{:%F}","lastDay":"{:%F}","noLocations":{},"noTrips":{},"transportsXDays":{}}})",
        idx, from + date::days{fm.first_}, from + date::days{fm.last_},
        fm.locations_, fm.trips_, fm.transport_days_);
  }
  ss << ']';
  return ss.str();
}

}  // namespace nigiri
