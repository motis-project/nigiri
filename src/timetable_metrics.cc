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
    // auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];
    auto tdm = transport_day_metric{
        .count_ = traffic_days.count(),
    };
    traffic_days.for_each_set_bit([&](std::uint16_t const day) {
      tdm.first_ = std::min(tdm.first_, day);
      tdm.last_ = std::max(tdm.last_, day);
    });
    return tdm;
  };

  std::cout << "\n\nDEBUG START\n\n";
  for (auto r = trip_idx_t{0}; r < tt.trip_transport_ranges_.size(); ++r) {
    std::cout << "trip-idx: " << r << "  transports: ";
    for (auto const rr : tt.trip_transport_ranges_[r]) {
      auto const bf = tt.bitfields_[tt.transport_traffic_days_[rr.first]];
      auto const tdm = transport_day_metrics(bf);
      std::cout << rr.first << " (" << tdm.count_ << " / " << tdm.first_ << "-" << tdm.last_ << "), ";
    }
    std::cout << "\n";
  }
  std::cout << "\n\nDEBUG END\n\n";

  auto m = timetable_metrics{};
  m.feeds_.resize(tt.n_sources());

  for (auto const src : tt.locations_.src_) {
    if (src == source_idx_t::invalid()) {
      continue;
    }
    ++m.feeds_[src].locations_;
  }

  for (auto t = transport_idx_t{0U}; t < tt.transport_traffic_days_.size();
       ++t) {
    auto const& merged_trips_bucket = tt.transport_to_trip_section_[t];
    //auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];
    //if (traffic_days.any()) {
      //auto const tdm = transport_day_metrics(traffic_days);
      //auto sources = bitfield{};
      //auto const x = sources | sources;
      auto sources = vector<source_idx_t>{};
      auto days = tt.bitfields_[tt.transport_traffic_days_[t]];
      auto trips = 0U;
      for (auto const merged_trips : merged_trips_bucket) {
        for (auto const trip_idx : tt.merged_trips_[merged_trips]) {
          ++trips;
          for (auto const trip_id : tt.trip_ids_[trip_idx]) {
            auto const src = tt.trip_id_src_[trip_id];
            if (!sources.contains(&src)) {
              sources.push_back(src);
            }
            //++m.feeds_[src].trips_;
            //sources.set(to_idx(src));
          }
          for (auto const ttr : tt.trip_transport_ranges_[trip_idx]) {
            days |= tt.bitfields_[tt.transport_traffic_days_[ttr.first]];
          }
        }
      }
      auto const tdm = transport_day_metrics(days);
      //sources.for_each_set_bit([&](std::size_t const src_) {
      //  auto const src = source_idx_t{src_};
      for (auto const src : sources) {
        /*
        if (src == source_idx_t::invalid()) {
          return;
        }
        */
        m.feeds_[src].trips_ += trips;
        m.feeds_[src].first_ = std::min(m.feeds_[src].first_, tdm.first_);
        m.feeds_[src].last_ = std::max(m.feeds_[src].last_, tdm.last_);
        m.feeds_[src].transport_days_ += tdm.count_;
      }
      //});
    /*
    } else {
      for (auto const merged_trips : merged_trips_bucket) {
        for (auto const trip_idx : tt.merged_trips_[merged_trips]) {
          std::cout << "SIZES: " << tt.trip_transport_ranges_[trip_idx].size() << ", " << tt.trip_ids_[trip_idx].size() << "\n";
          for (auto const ttr : tt.trip_transport_ranges_[trip_idx]) {
            std::cout << "ttr: " << ttr.first << ", " << trip_idx << ", " << t << '\n';
            auto const tdm = transport_day_metrics(
                tt.bitfields_[tt.transport_traffic_days_[ttr.first]]);
            for (auto const trip_id : tt.trip_ids_[trip_idx]) {
              auto const src = tt.trip_id_src_[trip_id];
              // TODO Correct count?
              ++m.feeds_[src].trips_;
              m.feeds_[src].first_ = std::min(m.feeds_[src].first_, tdm.first_);
              m.feeds_[src].last_ = std::max(m.feeds_[src].last_, tdm.last_);
              m.feeds_[src].transport_days_ += tdm.count_;
            }
          }
        }
      }
    }
    */
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
