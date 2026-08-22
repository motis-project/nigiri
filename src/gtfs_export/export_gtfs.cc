#include "nigiri/export_gtfs.h"

#include <cctype>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>

#include "miniz.h"

#include "utl/verify.h"

#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/common/day_list.h"
#include "nigiri/timetable.h"

namespace nigiri {

namespace {

struct progress_timer {
  explicit progress_timer(std::string label)
      : label_{std::move(label)}, start_{std::chrono::steady_clock::now()} {
    std::cout << "writing " << label_ << " ... " << std::flush;
  }
  progress_timer(progress_timer const&) = delete;
  progress_timer(progress_timer&&) = delete;
  progress_timer& operator=(progress_timer const&) = delete;
  progress_timer& operator=(progress_timer&&) = delete;
  ~progress_timer() {
    auto const elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_)
            .count();
    std::cout << "done. (" << elapsed_ms << "ms)\n";
  }

  std::string label_;
  std::chrono::steady_clock::time_point start_;
};

// Some agencies do not have a name or / and an url
constexpr auto const kDummyAgencyName = std::string_view{"Unknown Agency"};
constexpr auto const kDummyAgencyUrl = std::string_view{"https://example.com"};

}  // namespace

gtfs_dir_export_target::gtfs_dir_export_target(std::filesystem::path dir)
    : dir_{std::move(dir)} {
  std::filesystem::create_directories(dir_);
}

std::ostream& gtfs_dir_export_target::create_file(std::string const& filename) {
  streams_.push_back(std::make_unique<std::ofstream>(dir_ / filename));
  utl::verify(streams_.back()->is_open(), "gtfs export: cannot create {}",
              (dir_ / filename).string());
  return *streams_.back();
}

namespace {
std::filesystem::path make_tmp_export_dir(
    std::filesystem::path const& zip_path) {
  auto dir = zip_path;
  dir += ".tmp_export";
  auto ec = std::error_code{};
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir);
  return dir;
}
}  // namespace

struct gtfs_zip_export_target::impl {
  explicit impl(std::filesystem::path const& zip_path) {
    auto const ok =
        mz_zip_writer_init_file(&ar_, zip_path.string().c_str(), 0U);
    utl::verify(ok == MZ_TRUE, "gtfs export: cannot create zip archive {}",
                zip_path.string());
  }
  ~impl() { mz_zip_writer_end(&ar_); }
  impl(impl const&) = delete;
  impl(impl&&) = delete;
  impl& operator=(impl const&) = delete;
  impl& operator=(impl&&) = delete;

  mz_zip_archive ar_{};
};

gtfs_zip_export_target::gtfs_zip_export_target(std::filesystem::path zip_path)
    : zip_path_{std::move(zip_path)},
      tmp_dir_{make_tmp_export_dir(zip_path_)},
      impl_{std::make_unique<impl>(zip_path_)} {}

gtfs_zip_export_target::~gtfs_zip_export_target() {
  auto ec = std::error_code{};
  std::filesystem::remove_all(tmp_dir_, ec);
}

std::ostream& gtfs_zip_export_target::create_file(std::string const& filename) {
  filenames_.push_back(filename);
  streams_.push_back(std::make_unique<std::ofstream>(tmp_dir_ / filename));
  utl::verify(streams_.back()->is_open(), "gtfs export: cannot create {}",
              (tmp_dir_ / filename).string());
  return *streams_.back();
}

void gtfs_zip_export_target::finalize() {
  for (auto& s : streams_) {
    s->flush();
    s->close();
  }

  for (auto const& name : filenames_) {
    auto const src = tmp_dir_ / name;
    auto const ok =
        mz_zip_writer_add_file(&impl_->ar_, name.c_str(), src.string().c_str(),
                               nullptr, 0, MZ_BEST_SPEED);
    utl::verify(ok == MZ_TRUE, "gtfs export: cannot add {} to zip archive {}",
                name, zip_path_.string());
  }

  auto const ok = mz_zip_writer_finalize_archive(&impl_->ar_);
  utl::verify(ok == MZ_TRUE, "gtfs export: cannot finalize zip archive {}",
              zip_path_.string());

  auto ec = std::error_code{};
  std::filesystem::remove_all(tmp_dir_, ec);
}

std::unique_ptr<gtfs_export_target> make_gtfs_export_target(
    std::filesystem::path const& output_path) {
  auto ext = output_path.extension().string();
  std::ranges::transform(ext, ext.begin(), [](unsigned char const c) {
    return static_cast<char>(std::tolower(c));
  });
  if (ext == ".zip") {
    return std::make_unique<gtfs_zip_export_target>(output_path);
  }
  return std::make_unique<gtfs_dir_export_target>(output_path);
}

std::string csv_escape(std::string_view input) {
  auto const had_quote = bool{input.find('"') != std::string_view::npos};
  auto const needs_quoting =
      bool{had_quote || input.find_first_of(",\n\r") != std::string_view::npos};

  if (!needs_quoting || input.empty()) {
    return std::string{input};
  }

  auto out = std::string{};
  out.reserve(input.size() + 2);
  out.push_back('"');
  for (auto const c : input) {
    if (c == '"') {
      continue;
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
  auto const total_seconds = int{static_cast<int>(d.count()) * 60};
  auto const h = int{total_seconds / 3600};
  auto const m = int{(total_seconds % 3600) / 60};
  auto const s = int{total_seconds % 60};
  return std::format("{:02}:{:02}:{:02}", h, m, s);
}

void export_gtfs(timetable const& tt,
                 std::filesystem::path const& output_path) {
  auto target = make_gtfs_export_target(output_path);

  auto route_offsets = std::vector<size_t>(tt.route_ids_.size());
  auto sum = std::size_t{0};

  for (auto s = source_idx_t{0}; s < tt.route_ids_.size(); ++s) {
    route_offsets[to_idx(s)] = sum;
    sum += tt.route_ids_[s].ids_.size();
  }

  write_feed_info(*target);
  write_agencies(tt, *target);
  write_stops(tt, *target);
  write_routes(tt, *target, route_offsets);
  write_trips(tt, *target, route_offsets);
  write_stop_times(tt, *target);
  write_calendar(tt, *target);
  write_transfers(tt, *target);

  target->finalize();
}

void write_feed_info(gtfs_export_target& out_target) {
  auto const timer = progress_timer{"feed_info.txt"};

  auto& out = out_target.create_file("feed_info.txt");
  out << "feed_publisher_name,feed_publisher_url,feed_lang,agency_timezone\n";
  out << "MOTIS - Export,github.com/motis-project/motis,EN,Etc/UTC\n";
}

void write_agencies(timetable const& tt, gtfs_export_target& out_target) {
  auto const timer = progress_timer{"agency.txt"};

  auto& out = out_target.create_file("agency.txt");
  out << "agency_id,agency_name,agency_url,agency_timezone\n";

  for (auto p = provider_idx_t{0}; p < tt.providers_.size(); ++p) {
    auto const& provider = tt.providers_[p];

    auto const raw_name = tt.get_default_translation(provider.name_);
    auto const raw_url = tt.get_default_translation(provider.url_);

    auto const name =
        std::string_view{raw_name.empty() ? kDummyAgencyName : raw_name};
    auto const url =
        std::string_view{raw_url.empty() ? kDummyAgencyUrl : raw_url};

    out << to_idx(p) << "," << csv_escape(name) << "," << csv_escape(url)
        << ",Etc/UTC\n";
  }
}

void write_stops(timetable const& tt, gtfs_export_target& out_target) {
  auto const timer = progress_timer{"stops.txt"};

  auto& out = out_target.create_file("stops.txt");
  out << "stop_id,original_stop_id,stop_name,stop_desc,stop_lat,stop_lon,"
         "location_type,"
         "parent_station\n";

  // precompute which locations are actually referenced as a root/parent
  auto is_parent = std::vector<bool>(tt.n_locations(), false);
  for (auto l = location_idx_t{stopOffset}; l < tt.n_locations(); ++l) {
    auto const root = tt.locations_.get_root_idx(l);
    if (root != l) {
      is_parent[to_idx(root)] = true;
    }
  }

  for (auto l = location_idx_t{stopOffset}; l < tt.n_locations(); ++l) {
    if (!is_parent[to_idx(l)]) {
      continue;
    }
    auto const id = std::size_t{to_idx(l) - stopOffset};
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
    auto const id = std::size_t{to_idx(l) - stopOffset};
    auto const original_id = tt.locations_.ids_[l].view();
    auto const name = tt.get_default_name(l);
    auto const desc =
        tt.get_default_translation(tt.locations_.descriptions_[l]);
    auto const coord = tt.locations_.coordinates_[l];
    auto const root = tt.locations_.get_root_idx(l);
    auto const has_parent = bool{root != l};

    if (is_parent[to_idx(l)] && !has_parent) {
      continue;
    }

    auto const parent_str = std::string{
        has_parent ? std::to_string(to_idx(root) - stopOffset) : ""};
    out << id << "," << csv_escape(original_id) << "," << csv_escape(name)
        << "," << csv_escape(desc) << "," << coord.lat_ << "," << coord.lng_
        << ",0," << parent_str << "\n";
  }
}

void write_stop_times(timetable const& tt, gtfs_export_target& out_target) {
  auto const timer = progress_timer{"stop_times.txt"};

  auto& out = out_target.create_file("stop_times.txt");
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
        auto const s_id = std::size_t{to_idx(loc) - stopOffset};
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
}

void write_trips(timetable const& tt,
                 gtfs_export_target& out_target,
                 std::vector<size_t> const& route_offsets) {
  auto const timer = progress_timer{"trips.txt"};

  auto& out = out_target.create_file("trips.txt");
  out << "route_id,service_id,trip_id,trip_headsign,trip_short_name,"
         "bikes_allowed,cars_allowed\n";

  auto const to_global_route_id = [&](source_idx_t s, route_id_idx_t r) {
    return route_offsets[to_idx(s)] + to_idx(r);
  };

  for (auto r = route_idx_t{0}; r < tt.n_routes(); ++r) {
    auto const transport_range = tt.route_transport_ranges_[r];
    auto const bikes_allowed = int{tt.has_bike_transport(r) ? 1 : 2};
    auto const cars_allowed = int{tt.has_car_transport(r) ? 1 : 2};

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
}

void write_routes(timetable const& tt,
                  gtfs_export_target& out_target,
                  std::vector<size_t> const& route_offsets) {
  auto const timer = progress_timer{"routes.txt"};

  auto& out = out_target.create_file("routes.txt");

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
}

void write_transfers(timetable const& tt, gtfs_export_target& out_target) {
  auto const timer = progress_timer{"transfers.txt"};

  auto& out = out_target.create_file("transfers.txt");
  out << "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n";

  for (auto l = location_idx_t{stopOffset}; l < tt.n_locations(); ++l) {
    for (auto const& fp : tt.locations_.footpaths_out_[0][l]) {
      auto const to = fp.target();
      out << (to_idx(l) - stopOffset) << "," << (to_idx(to) - stopOffset)
          << ",2," << (fp.duration_ * 60) << "\n";
    }
  }
}

void write_calendar(timetable const& tt, gtfs_export_target& out_target) {
  auto const timer = progress_timer{"calendar.txt and calendar_dates.txt"};

  auto& cal = out_target.create_file("calendar.txt");
  auto& exc = out_target.create_file("calendar_dates.txt");

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

  auto const get_weekday = [&](std::size_t day) -> unsigned {
    auto const sys_day = base + date::days{static_cast<int>(day)};
    return static_cast<unsigned>(date::weekday{sys_day}.c_encoding());
  };

  auto const write_as_exceptions_only =
      [&](bitfield const& bf, std::size_t const first, std::size_t const last,
          bitfield_idx_t const b) {
        for (auto d = first; d <= last; ++d) {
          if (bf.test(d)) {
            exc << to_idx(b) << "," << to_ymd_str(d) << ",1\n";
          }
        }
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

    if (last - first + 1 < 7) {
      write_as_exceptions_only(bf, first, last, b);
      continue;
    }

    auto active_count = std::array<int, 7>{};
    auto total_count = std::array<int, 7>{};
    for (auto d = first; d <= last; ++d) {
      auto const wd = get_weekday(d);
      ++total_count[wd];
      active_count[wd] += static_cast<int>(bf.test(d));
    }

    auto weekly_pattern = std::uint8_t{0};
    for (auto wd = 0U; wd < 7U; ++wd) {
      if (active_count[wd] * 2 > total_count[wd]) {
        weekly_pattern |= static_cast<std::uint8_t>(1U << wd);
      }
    }

    if (weekly_pattern == 0) {
      write_as_exceptions_only(bf, first, last, b);
      continue;
    }

    cal << to_idx(b) << ","  //
        << ((weekly_pattern >> 1U) & 1U) << ","  // Monday
        << ((weekly_pattern >> 2U) & 1U) << ","  // Tuesday
        << ((weekly_pattern >> 3U) & 1U) << ","  // Wednesday
        << ((weekly_pattern >> 4U) & 1U) << ","  // Thursday
        << ((weekly_pattern >> 5U) & 1U) << ","  // Friday
        << ((weekly_pattern >> 6U) & 1U) << ","  // Saturday
        << ((weekly_pattern >> 0U) & 1U) << ","  // Sunday
        << to_ymd_str(first) << "," << to_ymd_str(last) << "\n";

    for (auto d = first; d <= last; ++d) {
      auto const active = bf.test(d);
      auto const in_pattern = ((weekly_pattern >> get_weekday(d)) & 1U) != 0U;

      if (active && !in_pattern) {
        exc << to_idx(b) << "," << to_ymd_str(d) << ",1\n";
      } else if (!active && in_pattern) {
        exc << to_idx(b) << "," << to_ymd_str(d) << ",2\n";
      }
    }
  }
}
}  // namespace nigiri
