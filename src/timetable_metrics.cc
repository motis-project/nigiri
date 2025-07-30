#include "nigiri/timetable_metrics.h"

#include <chrono>

#include "utl/enumerate.h"

#include "fmt/format.h"

#include "date/date.h"

namespace nigiri {

timetable_metrics get_metrics(timetable const& tt) {
  auto m = timetable_metrics{};
  m.feeds_.resize(tt.n_sources());

  for (auto const src : tt.locations_.src_) {
    if (src == source_idx_t::invalid()) {
      continue;
    }
    ++m.feeds_[src].locations_;
  }

  std::cout << "Sizes: " << tt.transport_traffic_days_.size() << ", "
            << tt.transport_traffic_days_.size() << ", "
            << tt.route_transport_ranges_.size() << '\n';
  for (auto const& ti : tt.route_transport_ranges_) {
    std::cout << "Range: " << ti.from_ << " -> " << ti.to_ << '\n';
  }
  for (auto t = transport_idx_t{0U}; t < tt.transport_traffic_days_.size();
       ++t) {
    auto const& merged_trips_bucket = tt.transport_to_trip_section_[t];
    auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];
    auto first = std::numeric_limits<std::uint16_t>::max();
    auto last = std::numeric_limits<std::uint16_t>::min();
    traffic_days.for_each_set_bit([&](std::uint16_t const day) {
      first = std::min(first, day);
      last = std::max(last, day);
    });
    auto sources = bitfield{};
    std::cout << "t: " << t << ", " << merged_trips_bucket.size() << " // "
              << traffic_days.count() << '\n';
    for (auto const merged_trips : merged_trips_bucket) {
      std::cout << "  mt: " << merged_trips << ", "
                << tt.merged_trips_[merged_trips].size() << '\n';
      for (auto const trip_idx : tt.merged_trips_[merged_trips]) {
        std::cout << "    trip_idx: " << trip_idx << ", "
                  << tt.trip_ids_[trip_idx].size() << '\n';
        for (auto const trip_id : tt.trip_ids_[trip_idx]) {
          auto const src = tt.trip_id_src_[trip_id];
          std::cout << "      src: " << src << ", " << trip_id << "\n";
          ++m.feeds_[src].trips_;
          sources.set(to_idx(src));
        }
      }
    }
    sources.for_each_set_bit([&](std::size_t const src_) {
      auto const src = source_idx_t{src_};
      std::cout << "Test: " << src << '\n';
      if (src == source_idx_t::invalid()) {
        return;
      }
      m.feeds_[src].first_ = std::min(m.feeds_[src].first_, first);
      m.feeds_[src].last_ = std::max(m.feeds_[src].last_, last);
      m.feeds_[src].transport_days_ += traffic_days.count();
    });
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
        R"("{{idx":{},"first_day":"{:%F}","last_day":"{:%F}","#locations":{},"#trips":{},"transports x days":{}}})",
        idx, from + date::days{fm.first_}, from + date::days{fm.last_},
        fm.locations_, fm.trips_, fm.transport_days_);
  }
  ss << ']';
  return ss.str();
}

}  // namespace nigiri
