#include "nigiri/loader/hrd/load_timetable.h"

#include <execution>

#include "wyhash.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/loader/hrd/service/service_builder.h"
#include "nigiri/loader/hrd/stamm/stamm.h"
#include "nigiri/special_stations.h"

namespace nigiri::loader::hrd {

bool applicable(config const& c, dir const& d) {
  return utl::all_of(
      c.required_files_, [&](std::vector<std::string> const& alt) {
        return alt.empty() || utl::any_of(alt, [&](std::string const& file) {
                 auto const exists = d.exists(c.core_data_ / file);
                 if (!exists) {
                   log(log_lvl::info, "loader.hrd",
                       "missing file for config {}: {}", c.version_.view(),
                       (c.core_data_ / file));
                 }
                 return exists;
               });
      });
}

void build_route_stop_times(timetable& tt) {
  for (auto i = route_idx_t{0U}; i != tt.route_transport_ranges_.size(); ++i) {
    auto const transport_range = tt.route_transport_ranges_[i];
    auto const stop_seq = tt.route_location_seq_[i];

    auto const stop_times_begin = tt.route_stop_times_.size();
    for (auto const [from, to] : utl::pairwise(interval{0U, stop_seq.size()})) {
      for (auto const t : transport_range) {
        tt.route_stop_times_.emplace_back(
            tt.event_mam(t, from, event_type::kDep));
      }
      for (auto const t : transport_range) {
        tt.route_stop_times_.emplace_back(
            tt.event_mam(t, to, event_type::kArr));
      }
    }
    auto const stop_times_end = tt.route_stop_times_.size();
    tt.route_stop_time_ranges_.emplace_back(
        interval{stop_times_begin, stop_times_end});
  }
}

void build_route_traffic_days(timetable& tt) {
  tt.col_bitfields_.resize(kMaxDays * tt.bitfields_.size());
  for (auto day = 0U; day != kMaxDays; ++day) {
    for (auto const [bf_idx, bitfield] : utl::enumerate(tt.bitfields_)) {
      tt.col_bitfields_.set(
          static_cast<unsigned>(tt.bitfields_.size() * day + bf_idx),
          bitfield.test(day));
    }
  }
}

std::uint64_t hash(config const& c, dir const& d, std::uint64_t const seed) {
  auto h = seed;
  for (auto const& f : stamm::load_files(c, d)) {
    if (!f.has_value()) {
      h = wyhash64(h, _wyp[0]);
    } else {
      auto const data = f.data();
      h = wyhash(data.data(), data.size(), h, _wyp);
    }
  }
  for (auto const& path : d.list_files(c.fplan_)) {
    auto const f = d.get_file(path);
    auto const data = f.data();
    h = wyhash(data.data(), data.size(), h, _wyp);
  }
  return h;
}

void load_timetable(source_idx_t const src,
                    config const& c,
                    dir const& d,
                    timetable& tt,
                    interval<std::chrono::sys_days> selection) {
  (void)src;
  auto bars = utl::global_progress_bars{false};

  auto empty_idx_vec = vector<location_idx_t>{};
  auto empty_footpath_vec = vector<footpath>{};
  for (auto const& name : special_stations_names) {
    tt.locations_.register_location(location{name,
                                             name,
                                             {0.0, 0.0},
                                             source_idx_t{0U},
                                             location_type::kStation,
                                             osm_node_id_t::invalid(),
                                             location_idx_t ::invalid(),
                                             timezone_idx_t ::invalid(),
                                             0_minutes,
                                             it_range{empty_idx_vec},
                                             it_range{empty_footpath_vec},
                                             it_range{empty_footpath_vec}});
  }

  auto st = stamm{c, tt, d};
  service_builder sb{st, tt, selection};

  auto progress_tracker = utl::activate_progress_tracker("nigiri");
  progress_tracker->status("Read Services")
      .in_high(utl::all(d.list_files(c.fplan_))  //
               | utl::transform([&](auto&& f) { return d.file_size(f); })  //
               | utl::sum());
  auto total_bytes_processed = std::uint64_t{0U};

  for (auto const& path : d.list_files(c.fplan_)) {
    log(log_lvl::info, "loader.hrd.services", "loading {}", path);
    auto const file = d.get_file(path);
    sb.add_services(
        c, relative(path, c.fplan_).string().c_str(), file.data(),
        [&](std::size_t const bytes_processed) {
          progress_tracker->update(total_bytes_processed + bytes_processed);
        });
    sb.write_services(src);
    total_bytes_processed += file.data().size();
  }

  tt.location_routes_[location_idx_t{tt.locations_.src_.size() - 1}];

  scoped_timer sort_timer{"sorting trip ids"};

  std::sort(
#if __cpp_lib_execution
      std::execution::par_unseq,
#endif
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_),
      [&](pair<trip_id_idx_t, trip_idx_t> const& a,
          pair<trip_id_idx_t, trip_idx_t> const& b) {
        return tt.trip_id_strings_[a.first].view() <
               tt.trip_id_strings_[b.first].view();
      });
  build_footpaths(tt);
  build_lb_graph<direction::kForward>(tt);
  build_lb_graph<direction::kBackward>(tt);
  build_route_stop_times(tt);
  build_route_traffic_days(tt);
}

}  // namespace nigiri::loader::hrd
