#include "nigiri/timetable.h"

#include "cista/io.h"

#include "utl/overloaded.h"

#include "nigiri/common/day_list.h"
#include "nigiri/rt/frun.h"

namespace nigiri {

std::string reverse(std::string s) {
  std::reverse(s.begin(), s.end());
  return s;
}

void timetable::resolve() {
  for (auto& tz : timezones_) {
    if (holds_alternative<pair<string, void const*>>(tz)) {
      auto& [name, ptr] = tz.as<pair<string, void const*>>();
      ptr = date::locate_zone(name);
    }
  }
}

provider_idx_t timetable::get_provider_idx(std::string_view id,
                                           source_idx_t const src) const {
  auto const id_str_idx = strings_.find(id);
  if (!id_str_idx.has_value()) {
    return provider_idx_t::invalid();
  }
  auto const it = std::lower_bound(
      begin(provider_id_to_idx_), end(provider_id_to_idx_), *id_str_idx,
      [&](provider_idx_t const a, string_idx_t const b) {
        auto const& p = providers_[a];
        return std::tuple{p.src_, p.id_} < std::tuple{src, b};
      });
  if (it == end(provider_id_to_idx_) || providers_[*it].src_ != src ||
      *id_str_idx != providers_[*it].id_) {
    return provider_idx_t::invalid();
  }
  return *it;
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

day_list timetable::days(bitfield const& bf) const {
  return day_list{bf, internal_interval_days().from_};
}

cista::wrapped<timetable> timetable::read(std::filesystem::path const& p) {
  return cista::read<timetable>(p);
}

void timetable::write(std::filesystem::path const& p) const {
  return cista::write(p, *this);
}

}  // namespace nigiri
