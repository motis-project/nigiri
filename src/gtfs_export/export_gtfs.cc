#include "nigiri/export_gtfs.h"

#include <filesystem>
#include <format>
#include <fstream>

#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/common/day_list.h"
#include "nigiri/timetable.h"

namespace nigiri {

std::string csv_escape(std::string_view input) {
  std::string out;
  out.reserve(input.size());

  for (char c : input) {
    switch (c) {
      case '"':
        out.push_back('"');
        out.push_back('"');
        break;
      case '\n':
      case '\r': out.push_back(' '); break;
      default: out.push_back(c);
    }
  }
  return out;
}

static std::string format_time(delta d) {
  auto total_seconds = static_cast<int>(d.count()) * 60;
  auto h = total_seconds / 3600;
  auto m = (total_seconds % 3600) / 60;
  auto s = total_seconds % 60;

  return std::format("{:02}:{:02}:{:02}", h, m, s);
}

void export_gtfs(timetable const& tt, std::filesystem::path const& dir) {
  std::filesystem::create_directories(dir);

  std::vector<size_t> route_offsets(tt.route_ids_.size());

  size_t sum = 0;
  for (source_idx_t s{0}; s < tt.route_ids_.size(); ++s) {
    route_offsets[to_idx(s)] = sum;
    sum += tt.route_ids_[s].ids_.size();
  }

  write_agencies(tt, dir);
  write_stops(tt, dir);
  write_routes(tt, dir, route_offsets);
  write_trips(tt, dir, route_offsets);
  write_stop_times(tt, dir);
  write_calendar(tt, dir);
  /* write_calendar_dates(tt, dir); */
  write_transfers(tt, dir);
}

void write_agencies(timetable const& tt, std::filesystem::path const& dir) {
  std::ofstream out(dir / "agency.txt");

  out << "agency_id,agency_name,agency_url,agency_timezone\n";

  for (provider_idx_t p{0}; p < tt.providers_.size(); ++p) {
    auto const& provider = tt.providers_[p];

    auto const& tz = nigiri::loader::gtfs::get_timezone_name(tt, provider.tz_);
    auto const timezone_name = tz.value_or("Europe/Berlin");

    out << to_idx(p) << ",\""
        << csv_escape(tt.get_default_translation(provider.name_)) << "\","
        << tt.get_default_translation(provider.url_) << "," << timezone_name
        << "\n";
  }
}

void write_stops(timetable const& tt, std::filesystem::path const& output_dir) {
  std::ofstream out(output_dir / "stops.txt");
  out << "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n";

  for (location_idx_t l{stopOffset}; l < tt.n_locations(); ++l) {
    if (tt.locations_.children_[l].empty()) {
      continue;
    }

    auto const id = (to_idx(l) - stopOffset);
    auto const name = tt.get_default_name(l);
    auto const coord = tt.locations_.coordinates_[l];

    out << id << ",\"" << csv_escape(name) << "\"," << coord.lat_ << ","
        << coord.lng_ << ",1,\n";
  }

  for (location_idx_t l{stopOffset}; l < tt.n_locations(); ++l) {
    auto const id = (to_idx(l) - stopOffset);
    auto const name = tt.get_default_name(l);
    auto const coord = tt.locations_.coordinates_[l];

    auto const root = tt.locations_.get_root_idx(l);
    bool const has_parent = (root != l);

    // Skip pure parent stations written in first pass
    if (!tt.locations_.children_[l].empty() && !has_parent) {
      continue;
    }

    auto const parent_str =
        has_parent ? std::to_string(to_idx(root) - stopOffset) : "";
    out << id << ",\"" << csv_escape(name) << "\"," << coord.lat_ << ","
        << coord.lng_ << ",0," << parent_str << "\n";
  }
}

void write_stop_times(timetable const& tt, std::filesystem::path const& dir) {
  std::ofstream out(dir / "stop_times.txt");

  out << "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n";

  for (route_idx_t r{0}; r < tt.n_routes(); ++r) {
    auto const stops = tt.route_location_seq_.at(r);
    auto const transports = tt.route_transport_ranges_.at(r);

    for (transport_idx_t t = transports.from_; t != transports.to_; ++t) {
      // Skip transports with no active traffic days
      if (tt.bitfields_[tt.transport_traffic_days_[t]].none()) {
        continue;
      }

      for (auto s = stop_idx_t(0); s < stops.size(); ++s) {
        auto loc = stop{stops[s]}.location_idx();

        auto sId = (to_idx(loc) - stopOffset);
        auto arr = (s == stop_idx_t(0) ? tt.event_mam(t, s, event_type::kDep)
                                       : tt.event_mam(t, s, event_type::kArr));
        auto dep = (s == stop_idx_t(stops.size() - 1)
                        ? tt.event_mam(t, s, event_type::kArr)
                        : tt.event_mam(t, s, event_type::kDep));
        out << to_idx(t) << "," << format_time(arr) << "," << format_time(dep)
            << "," << sId << "," << s << "\n";
      }
    }
  }
}

void write_trips(timetable const& tt,
                 std::filesystem::path const& dir,
                 std::vector<size_t> const& route_offsets) {
  std::ofstream out(dir / "trips.txt");
  out << "route_id,service_id,trip_id,trip_headsign,trip_short_name,bikes_"
         "allowed,cars_"
         "allowed\n";

  auto to_global_route_id = [&](source_idx_t s, route_id_idx_t r) {
    return route_offsets[to_idx(s)] + to_idx(r);
  };

  for (route_idx_t r{0}; r < tt.n_routes(); ++r) {
    auto const transport_range = tt.route_transport_ranges_[r];

    auto const bikes_allowed = tt.has_bike_transport(r) ? 1 : 2;
    auto const cars_allowed = tt.has_car_transport(r) ? 1 : 2;

    for (transport_idx_t t = transport_range.from_; t != transport_range.to_;
         ++t) {
      auto const merged_idx = tt.transport_to_trip_section_[t].front();
      auto const trip_idx = tt.merged_trips_[merged_idx].front();

      auto const source_id = tt.trip_id_src_[tt.trip_ids_[trip_idx].front()];

      auto const route_id = tt.trip_route_id_[trip_idx];
      auto const global_route_id = to_global_route_id(source_id, route_id);

      if (tt.bitfields_[tt.transport_traffic_days_[t]].none()) {
        continue;
      }

      auto const service_id = to_idx(tt.transport_traffic_days_[t]);
      auto const trip_id = to_idx(t);
      auto const short_name =
          tt.get_default_translation(tt.trip_short_names_[trip_idx]);
      auto const headsign =
          tt.get_default_translation(tt.trip_display_names_[trip_idx]);

      out << global_route_id << "," << service_id << "," << trip_id << ",\""
          << csv_escape(short_name) << "\",\"" << csv_escape(headsign) << "\","
          << bikes_allowed << "," << cars_allowed << "\n";
    }
  }
}

void write_routes(timetable const& tt,
                  std::filesystem::path const& dir,
                  std::vector<size_t> const& route_offsets) {
  std::ofstream out(dir / "routes.txt");

  auto to_global_route_id = [&](source_idx_t s, route_id_idx_t r) {
    return route_offsets[to_idx(s)] + to_idx(r);
  };

  out << "route_id,agency_id,route_short_name,route_long_name,route_type,route_"
         "color,route_text_color\n";

  for (source_idx_t s{0}; s < tt.route_ids_.size(); ++s) {
    auto const& routes = tt.route_ids_[s];

    for (route_id_idx_t r{0}; r < routes.ids_.size(); ++r) {
      auto const global_id = to_global_route_id(s, r);
      auto short_name =
          tt.get_default_translation(routes.route_id_short_names_[r]);

      auto long_name =
          tt.get_default_translation(routes.route_id_long_names_[r]);

      auto agency = routes.route_id_provider_[r];
      auto type = to_idx(routes.route_id_type_[r]);

      auto const& rc = routes.route_id_colors_[r];
      auto const color_str = to_str(rc.color_).value_or("");
      auto const text_str = to_str(rc.text_color_).value_or("");

      out << global_id << "," << agency << ","
          << "\"" << csv_escape(short_name) << "\","
          << "\"" << csv_escape(long_name) << "\"," << type << "," << color_str
          << "," << text_str << "\n";
    }
  }
}

void write_transfers(timetable const& tt, std::filesystem::path const& dir) {
  std::ofstream out(dir / "transfers.txt");

  out << "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n";

  for (location_idx_t l{stopOffset}; l < tt.n_locations(); ++l) {
    for (auto const& fp : tt.locations_.footpaths_out_[0][l]) {
      auto to = fp.target();

      // TODO check if transfers contain wheelchair or some
      out << (to_idx(l) - stopOffset) << "," << (to_idx(to) - stopOffset) << ","
          << "2," << (fp.duration_ * 60) << "\n";
    }
  }
}

void write_calendar(timetable const& tt, std::filesystem::path const& dir) {
  std::ofstream cal(dir / "calendar.txt");
  std::ofstream exc(dir / "calendar_dates.txt");

  cal << "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
         "start_date,end_date\n";
  exc << "service_id,date,exception_type\n";

  auto const& base = tt.internal_interval_days().from_;
  auto to_ymd_str = [&](std::size_t day) {
    auto sys_day = base + date::days{static_cast<int>(day)};
    auto ymd = date::year_month_day{sys_day};
    return std::format("{:04}{:02}{:02}", int(ymd.year()),
                       unsigned(ymd.month()), unsigned(ymd.day()));
  };

  auto get_weekday = [&](std::size_t day) -> int {
    auto sys_day = base + date::days{static_cast<int>(day)};
    return date::weekday{sys_day}.c_encoding();  // 0=Sun..6=Sat
  };

  for (bitfield_idx_t b{0}; b < tt.bitfields_.size(); ++b) {
    auto const& bf = tt.bitfields_[b];
    if (bf.none()) continue;

    std::size_t first = 0, last = 0;
    for (std::size_t d = 0; d < bf.size(); ++d) {
      if (bf.test(d)) {
        first = d;
        break;
      }
    }
    for (std::size_t d = bf.size(); d-- > 0;) {
      if (bf.test(d)) {
        last = d;
        break;
      }
    }

    if (last - first < 7) {
      for (std::size_t d = first; d <= last; ++d) {
        if (bf.test(d)) exc << to_idx(b) << "," << to_ymd_str(d) << ",1\n";
      }
      continue;
    }

    int active_count[7] = {};
    int total_count[7] = {};
    for (std::size_t d = first; d <= last; ++d) {
      int wd = get_weekday(d);
      total_count[wd]++;
      if (bf.test(d)) active_count[wd]++;
    }

    uint8_t best_map = 0;
    for (int wd = 0; wd < 7; ++wd) {
      if (total_count[wd] > 0 && active_count[wd] * 2 > total_count[wd]) {
        best_map |= (1 << wd);
      }
    }

    if (best_map == 0) {
      for (std::size_t d = first; d <= last; ++d) {
        if (bf.test(d)) exc << to_idx(b) << "," << to_ymd_str(d) << ",1\n";
      }
      continue;
    }

    std::size_t am_start = first - get_weekday(first);
    std::size_t am_end = last + (6 - get_weekday(last));
    int l = static_cast<int>((am_end - am_start) / 7) + 1;  // number of weeks

    uint32_t best_e = UINT32_MAX;
    int best_a = 0, best_b = l - 1;

    for (int a = 0; a < l; ++a) {
      for (int bb = l - 1; bb >= a; --bb) {
        uint32_t e = 0;
        for (std::size_t d = first; d <= last; ++d) {
          int week = static_cast<int>((d - am_start) / 7);
          bool in_span = (week >= a && week <= bb);
          bool in_pattern = in_span && ((best_map >> get_weekday(d)) & 1);
          bool active = bf.test(d);
          if (active != in_pattern) e++;
        }
        if (e < best_e) {
          best_e = e;
          best_a = a;
          best_b = bb;
          if (e == 0) goto done;
        }
      }
    }
  done:

    std::size_t new_begin = am_start + best_a * 7;
    std::size_t new_end = am_start + best_b * 7 + 6;

    while (new_begin <= new_end && !bf.test(new_begin)) ++new_begin;
    while (new_end >= new_begin && !bf.test(new_end)) --new_end;

    if (new_end - new_begin < 7) {
      for (std::size_t d = first; d <= last; ++d) {
        if (bf.test(d)) exc << to_idx(b) << "," << to_ymd_str(d) << ",1\n";
      }
      continue;
    }

    cal << to_idx(b) << "," << ((best_map >> 1) & 1) << ","  // Monday
        << ((best_map >> 2) & 1) << ","  // Tuesday
        << ((best_map >> 3) & 1) << ","  // Wednesday
        << ((best_map >> 4) & 1) << ","  // Thursday
        << ((best_map >> 5) & 1) << ","  // Friday
        << ((best_map >> 6) & 1) << ","  // Saturday
        << ((best_map >> 0) & 1) << ","  // Sunday
        << to_ymd_str(new_begin) << "," << to_ymd_str(new_end) << "\n";

    for (std::size_t d = first; d <= last; ++d) {
      bool active = bf.test(d);
      int week = static_cast<int>((d - am_start) / 7);
      bool in_span = (week >= best_a && week <= best_b);
      bool in_pattern = in_span && ((best_map >> get_weekday(d)) & 1);

      if (active && !in_pattern) {
        exc << to_idx(b) << "," << to_ymd_str(d) << ",1\n";
      } else if (!active && in_pattern) {
        exc << to_idx(b) << "," << to_ymd_str(d) << ",2\n";
      }
    }
  }
}

void write_calendar_dates(timetable const& tt,
                          std::filesystem::path const& dir) {
  std::ofstream out(dir / "calendar_dates.txt");
  out << "service_id,date,exception_type\n";

  auto const& base = tt.internal_interval_days().from_;

  for (bitfield_idx_t b{0}; b < tt.bitfields_.size(); ++b) {
    auto const& bf = tt.bitfields_[b];

    if (bf.none()) {
      continue;
    }

    for (std::size_t day = 0; day < bf.size(); ++day) {
      if (bf.test(day)) {
        auto const sys_day = base + date::days{static_cast<int>(day)};
        auto const ymd = date::year_month_day{sys_day};
        auto buf = std::format("{:04}{:02}{:02}", int(ymd.year()),
                               unsigned(ymd.month()), unsigned(ymd.day()));
        out << to_idx(b) << "," << buf << ",1\n";
      }
    }
  }
}

}  // namespace nigiri
