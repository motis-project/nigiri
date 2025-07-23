#include "nigiri/timetable.h"

#include "cista/io.h"

#include "utl/overloaded.h"

#include "fmt/format.h"
#include "fmt/ranges.h"

#include "nigiri/common/day_list.h"
#include "nigiri/rt/frun.h"

namespace nigiri {

std::string reverse(std::string s) {
  std::reverse(s.begin(), s.end());
  return s;
}

void timetable::resolve() {
  for (auto& tz : locations_.timezones_) {
    if (holds_alternative<pair<string, void const*>>(tz)) {
      auto& [name, ptr] = tz.as<pair<string, void const*>>();
      ptr = date::locate_zone(name);
    }
  }
}

std::ostream& operator<<(std::ostream& out, timetable const& tt) {
  for (auto const [id, idx] : tt.trip_id_to_idx_) {
    auto const str = tt.trip_id_strings_[id].view();
    out << str << ":\n";
    for (auto const& t : tt.trip_transport_ranges_.at(idx)) {
      out << "  " << t.first << ": " << t.second << " active="
          << day_list{tt.bitfields_[tt.transport_traffic_days_[t.first]],
                      tt.internal_interval_days().from_}
          << "\n";
    }
  }

  auto const internal = tt.internal_interval_days();
  auto const num_days =
      static_cast<size_t>((internal.to_ - internal.from_ + 1_days) / 1_days);
  for (auto i = 0U; i != tt.transport_traffic_days_.size(); ++i) {
    auto const transport_idx = transport_idx_t{i};
    auto const num_stops =
        tt.route_location_seq_[tt.transport_route_[transport_idx]].size();
    auto const traffic_days =
        tt.bitfields_.at(tt.transport_traffic_days_.at(transport_idx));
    out << "TRANSPORT=" << transport_idx << ", TRAFFIC_DAYS="
        << reverse(traffic_days.to_string().substr(kMaxDays - num_days))
        << "\n";
    for (auto d = internal.from_; d != internal.to_;
         d += std::chrono::days{1}) {
      auto const day_idx = day_idx_t{
          static_cast<day_idx_t::value_t>((d - internal.from_) / 1_days)};
      if (traffic_days.test(to_idx(day_idx))) {
        date::to_stream(out, "%F", d);
        out << " (day_idx=" << day_idx << ")\n";
        out << rt::frun{
            tt,
            nullptr,
            {.t_ = transport{transport_idx, day_idx},
             .stop_range_ = {0U, static_cast<stop_idx_t>(num_stops)}}};
        out << "\n";
      }
    }
    out << "---\n\n";
  }
  return out;
}

cista::wrapped<timetable> timetable::read(std::filesystem::path const& p) {
  return cista::read<timetable>(p);
}

void timetable::write(std::filesystem::path const& p) const {
  return cista::write(p, *this);
}

std::string timetable::json_stats() const {
  auto ss = std::stringstream{};
  ss << "[";
  for (auto idx = source_idx_t{0}; idx < n_sources(); ++idx) {
    if (idx > 0) {
      ss << ',';
    }
    ss << statistics_[idx].json(idx, source_file_names_[source_file_idx_t{to_idx(idx)}].view());
  }
  ss << "]";
  return ss.str();
}

}  // namespace nigiri
