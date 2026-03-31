#include "nigiri/export_gtfs.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <limits>

#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/common/day_list.h"
#include "nigiri/timetable.h"

namespace nigiri {

std::string csv_escape(std::string_view input) {
  auto const needs_quoting =
      input.find_first_of(",\"\n\r") != std::string_view::npos;

  if (!needs_quoting || input.empty()) {
    return std::string{input};
  }

  auto out = std::string{};
  out.reserve(input.size() + 2);
  out.push_back('"');
  for (auto const c : input) {
    if (c == '"') {

      out.push_back('"');
    } else if (c == '\n' || c == '\r') {
      out.push_back(' ');
    } else {
      out.push_back(c);
    }
  }
  out.push_back('"');
  return out;
}

static std::string format_time(delta d) {
  auto const total_seconds = static_cast<int>(d.count()) * 60;
  auto const h = total_seconds / 3600;
  auto const m = (total_seconds % 3600) / 60;
  auto const s = total_seconds % 60;
  return std::format("{:02}:{:02}:{:02}", h, m, s);
}

void export_gtfs(timetable const& tt, std::filesystem::path const& dir) {
  std::filesystem::create_directories(dir);

  auto route_offsets = std::vector<size_t>(tt.route_ids_.size());
  auto sum = std::size_t{0};

  for (auto s = source_idx_t{0}; s < tt.route_ids_.size(); ++s) {
    route_offsets[to_idx(s)] = sum;
    sum += tt.route_ids_[s].ids_.size();
  }

  write_feed_info(dir);
  write_agencies(tt, dir);
  write_stops(tt, dir);
  write_routes(tt, dir, route_offsets);
  write_trips(tt, dir, route_offsets);
  write_stop_times(tt, dir);
  write_calendar(tt, dir);
  write_transfers(tt, dir);
}

void write_feed_info(std::filesystem::path const& dir) {
  auto out = std::ofstream{dir / "feed_info.txt"};
  out << "feed_publisher_name,feed_publisher_url,feed_lang,agency_timezone\n";
  out << "MOTIS - Export,https://transitous.org/,EN,Etc/UTC\n";
}

void write_agencies(timetable const& tt, std::filesystem::path const& dir) {
  auto out = std::ofstream{dir / "agency.txt"};
  out << "agency_id,agency_name,agency_url,agency_timezone\n";

  for (auto p = provider_idx_t{0}; p < tt.providers_.size(); ++p) {
    auto const& provider = tt.providers_[p];
    out << to_idx(p) << ","
        << csv_escape(tt.get_default_translation(provider.name_)) << ","
        << tt.get_default_translation(provider.url_) << ",Etc/UTC\n";
  }
}

void write_stops(timetable const& tt, std::filesystem::path const& output_dir) {
  std::cout << "writing stops.txt ... ";

  auto out = std::ofstream{output_dir / "stops.txt"};
  out << "stop_id,original_stop_id,stop_name,stop_desc,stop_lat,stop_lon,"
         "location_type,"
         "parent_station\n";

  for (auto l = location_idx_t{stopOffset}; l < tt.n_locations(); ++l) {
    if (tt.locations_.children_[l].empty()) {
      continue;
    }
    auto const id = to_idx(l) - stopOffset;
    auto const original_id = tt.locations_.ids_[l].view();
    auto const name = tt.get_default_name(l);
    auto const desc =
        tt.get_default_translation(tt.locations_.descriptions_[l]);
    auto const coord = tt.locations_.coordinates_[l];
    out << id << "," << csv_escape(original_id) << "," << csv_escape(name)
        << "," << csv_escape(desc) << "," << coord.lat_ << "," << coord.lng_
        << ",1,\n";
  }

  for (auto l = location_idx_t{stopOffset}; l < tt.n_locations(); ++l) {
    auto const id = to_idx(l) - stopOffset;
    auto const original_id = tt.locations_.ids_[l].view();
    auto const name = tt.get_default_name(l);
    auto const desc =
        tt.get_default_translation(tt.locations_.descriptions_[l]);
    auto const coord = tt.locations_.coordinates_[l];
    auto const root = tt.locations_.get_root_idx(l);
    auto const has_parent = (root != l);

    if (!tt.locations_.children_[l].empty() && !has_parent) {
      continue;
    }

    auto const parent_str =
        has_parent ? std::to_string(to_idx(root) - stopOffset) : "";
    out << id << "," << csv_escape(original_id) << "," << csv_escape(name)
        << "," << csv_escape(desc) << "," << coord.lat_ << "," << coord.lng_
        << ",0," << parent_str << "\n";
  }

  std::cout << "done.\n";
}

void write_stop_times(timetable const& tt, std::filesystem::path const& dir) {
  std::cout << "writing stop_times.txt ... ";

  auto out = std::ofstream{dir / "stop_times.txt"};
  out << "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n";

  for (auto r = route_idx_t{0}; r < tt.n_routes(); ++r) {
    auto const stops = tt.route_location_seq_.at(r);
    auto const transports = tt.route_transport_ranges_.at(r);

    for (auto t = transports.from_; t != transports.to_; ++t) {
      if (tt.bitfields_[tt.transport_traffic_days_[t]].none()) {
        continue;
      }
      for (auto s = stop_idx_t{0}; s < stops.size(); ++s) {
        auto const loc = stop{stops[s]}.location_idx();
        auto const s_id = to_idx(loc) - stopOffset;
        auto const arr =
            (s == stop_idx_t{0} ? tt.event_mam(t, s, event_type::kDep)
                                : tt.event_mam(t, s, event_type::kArr));
        auto const dep = (s == stop_idx_t(stops.size() - 1)
                              ? tt.event_mam(t, s, event_type::kArr)
                              : tt.event_mam(t, s, event_type::kDep));
        out << to_idx(t) << "," << format_time(arr) << "," << format_time(dep)
            << "," << s_id << "," << s << "\n";
      }
    }
  }
  std::cout << "done.\n";
}

void write_trips(timetable const& tt,
                 std::filesystem::path const& dir,
                 std::vector<size_t> const& route_offsets) {
  std::cout << "writing trips.txt ... ";

  auto out = std::ofstream{dir / "trips.txt"};
  out << "route_id,service_id,trip_id,trip_headsign,trip_short_name,"
         "bikes_allowed,cars_allowed\n";

  auto const to_global_route_id = [&](source_idx_t s, route_id_idx_t r) {
    return route_offsets[to_idx(s)] + to_idx(r);
  };

  for (auto r = route_idx_t{0}; r < tt.n_routes(); ++r) {
    auto const transport_range = tt.route_transport_ranges_[r];
    auto const bikes_allowed = tt.has_bike_transport(r) ? 1 : 2;
    auto const cars_allowed = tt.has_car_transport(r) ? 1 : 2;

    for (auto t = transport_range.from_; t != transport_range.to_; ++t) {
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
      auto headsign =
          tt.get_default_translation(tt.trip_display_names_[trip_idx]);

      auto const& headsigns = tt.transport_section_directions_.at(t);
      if (!headsigns.empty()) {
        headsign = tt.get_default_translation(headsigns.front());
      }

      out << global_route_id << "," << service_id << "," << trip_id << ","
          << csv_escape(headsign) << "," << csv_escape(short_name) << ","
          << bikes_allowed << "," << cars_allowed << "\n";
    }
  }
  std::cout << "done.\n";
}

void write_routes(timetable const& tt,
                  std::filesystem::path const& dir,
                  std::vector<size_t> const& route_offsets) {
  std::cout << "writing routes.txt ... ";

  auto out = std::ofstream{dir / "routes.txt"};

  auto const to_global_route_id = [&](source_idx_t s, route_id_idx_t r) {
    return route_offsets[to_idx(s)] + to_idx(r);
  };

  out << "route_id,agency_id,route_short_name,route_long_name,route_type,"
         "route_color,route_text_color\n";

  for (auto s = source_idx_t{0}; s < tt.route_ids_.size(); ++s) {
    auto const& routes = tt.route_ids_[s];
    auto const endRouteIds = static_cast<route_id_idx_t>(routes.ids_.size());
    for (auto r = route_id_idx_t{0}; r < endRouteIds; ++r) {
      auto const global_id = to_global_route_id(s, r);
      auto const short_name =
          tt.get_default_translation(routes.route_id_short_names_[r]);
      auto const long_name =
          tt.get_default_translation(routes.route_id_long_names_[r]);
      auto const agency = routes.route_id_provider_[r];
      auto const type = to_idx(routes.route_id_type_[r]);
      auto const& rc = routes.route_id_colors_[r];
      auto const color_str = to_str(rc.color_).value_or("");
      auto const text_str = to_str(rc.text_color_).value_or("");

      out << global_id << "," << agency << "," << csv_escape(short_name) << ","
          << csv_escape(long_name) << "," << type << "," << color_str << ","
          << text_str << "\n";
    }
  }
  std::cout << "done.\n";
}

void write_transfers(timetable const& tt, std::filesystem::path const& dir) {
  std::cout << "writing transfers.txt ... ";

  auto out = std::ofstream{dir / "transfers.txt"};
  out << "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n";

  for (auto l = location_idx_t{stopOffset}; l < tt.n_locations(); ++l) {
    for (auto const& fp : tt.locations_.footpaths_out_[0][l]) {
      auto const to = fp.target();
      out << (to_idx(l) - stopOffset) << "," << (to_idx(to) - stopOffset)
          << ",2," << (fp.duration_ * 60) << "\n";
    }
  }
  std::cout << "done.\n";
}

void write_calendar(timetable const& tt, std::filesystem::path const& dir) {
  std::cout << "writing calendar.txt and calendar_dates.txt ... ";

  auto cal = std::ofstream{dir / "calendar.txt"};
  auto exc = std::ofstream{dir / "calendar_dates.txt"};

  cal << "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,"
         "start_date,end_date\n";
  exc << "service_id,date,exception_type\n";

  auto const& base = tt.internal_interval_days().from_;

  auto const to_ymd_str = [&](std::size_t day) {
    auto const sys_day = base + date::days{static_cast<int>(day)};
    auto const ymd = date::year_month_day{sys_day};
    return std::format("{:04}{:02}{:02}", int(ymd.year()),
                       unsigned(ymd.month()), unsigned(ymd.day()));
  };

  auto const get_weekday = [&](std::size_t day) -> int {
    auto const sys_day = base + date::days{static_cast<int>(day)};
    return static_cast<int>(date::weekday{sys_day}.c_encoding());
  };

  for (auto b = bitfield_idx_t{0}; b < tt.bitfields_.size(); ++b) {
    auto const& bf = tt.bitfields_[b];
    if (bf.none()) {
      continue;
    }

    auto first = std::size_t{0};
    auto last = std::size_t{0};
    for (auto d = std::size_t{0}; d < bf.size(); ++d) {
      if (bf.test(d)) {
        first = d;
        break;
      }
    }
    for (auto d = bf.size(); d-- > 0;) {
      if (bf.test(d)) {
        last = d;
        break;
      }
    }

    if (last - first < 7) {
      for (auto d = first; d <= last; ++d) {
        if (bf.test(d)) {
          exc << to_idx(b) << "," << to_ymd_str(d) << ",1\n";
        }
      }
      continue;
    }

    auto active_count = std::array<int, 7>{};
    auto total_count = std::array<int, 7>{};
    for (auto d = first; d <= last; ++d) {
      auto const wd = static_cast<std::size_t>(get_weekday(d));
      total_count[wd]++;
      active_count[wd] += static_cast<int>(bf.test(d));
    }

    auto best_map = uint8_t{0};
    for (auto wd = std::size_t{0}; wd < 7; ++wd) {
      if (total_count[wd] > 0 && active_count[wd] * 2 > total_count[wd]) {
        best_map |= static_cast<uint8_t>(1 << wd);
      }
    }

    if (best_map == 0) {
      for (auto d = first; d <= last; ++d) {
        if (bf.test(d)) {
          exc << to_idx(b) << "," << to_ymd_str(d) << ",1\n";
        }
      }
      continue;
    }

    auto const am_start = first - static_cast<std::size_t>(get_weekday(first));
    auto const am_end = last + static_cast<std::size_t>(6 - get_weekday(last));
    auto const l = static_cast<int>((am_end - am_start) / 7) + 1;

    auto best_e = std::numeric_limits<uint32_t>::max();
    auto best_a = int{0};
    auto best_b = int{l - 1};

    for (auto a = int{0}; a < l; ++a) {
      for (auto bb = int{l - 1}; bb >= a; --bb) {
        auto e = uint32_t{0};
        for (auto d = first; d <= last; ++d) {
          auto const week = static_cast<int>((d - am_start) / 7);
          auto const in_span = (week >= a && week <= bb);
          auto const in_pattern = in_span && ((best_map >> get_weekday(d)) & 1);
          auto const active = bf.test(d);
          if (active != in_pattern) {
            e++;
          }
        }
        if (e < best_e) {
          best_e = e;
          best_a = a;
          best_b = bb;
          if (e == 0) {
            goto done;
          }
        }
      }
    }
  done:

    auto new_begin = static_cast<std::ptrdiff_t>(
        am_start + static_cast<std::size_t>(best_a) * 7);
    auto new_end = static_cast<std::ptrdiff_t>(
        am_start + static_cast<std::size_t>(best_b) * 7 + 6);

    while (new_begin <= new_end &&
           !bf.test(static_cast<std::size_t>(new_begin)))
      ++new_begin;
    while (new_end >= new_begin && !bf.test(static_cast<std::size_t>(new_end)))
      --new_end;

    if (new_end < new_begin ||
        static_cast<std::size_t>(new_end - new_begin) < 7) {
      for (auto d = first; d <= last; ++d) {
        if (bf.test(d)) {
          exc << to_idx(b) << "," << to_ymd_str(d) << ",1\n";
        }
      }
      continue;
    }

    auto const ub_new_begin = static_cast<std::size_t>(new_begin);
    auto const ub_new_end = static_cast<std::size_t>(new_end);

    cal << to_idx(b) << "," << ((best_map >> 1) & 1) << ","  // Monday
        << ((best_map >> 2) & 1) << ","  // Tuesday
        << ((best_map >> 3) & 1) << ","  // Wednesday
        << ((best_map >> 4) & 1) << ","  // Thursday
        << ((best_map >> 5) & 1) << ","  // Friday
        << ((best_map >> 6) & 1) << ","  // Saturday
        << ((best_map >> 0) & 1) << ","  // Sunday
        << to_ymd_str(ub_new_begin) << "," << to_ymd_str(ub_new_end) << "\n";

    for (auto d = first; d <= last; ++d) {
      auto const active = bf.test(d);
      auto const week = static_cast<int>((d - am_start) / 7);
      auto const in_span = (week >= best_a && week <= best_b);
      auto const in_pattern = in_span && ((best_map >> get_weekday(d)) & 1);

      if (active && !in_pattern) {
        exc << to_idx(b) << "," << to_ymd_str(d) << ",1\n";
      } else if (!active && in_pattern) {
        exc << to_idx(b) << "," << to_ymd_str(d) << ",2\n";
      }
    }
  }
  std::cout << "done.\n";
}
}  // namespace nigiri
