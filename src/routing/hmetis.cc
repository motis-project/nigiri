#include "nigiri/routing/hmetis.h"

#include <fstream>

#include "geo/box.h"

#include "boost/thread/tss.hpp"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/insert_sorted.h"
#include "utl/parallel_for.h"
#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"
#include "utl/pipes/all.h"
#include "utl/pipes/remove_if.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "nigiri/routing/no_route_filter.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

using namespace std::string_view_literals;

namespace nigiri::routing {

struct state {
  explicit state(timetable const& tt, date::sys_days const base_day)
      : tt_{tt},
        results_(tt.n_locations()),
        base_day_{tt.day_idx(base_day)},
        lb_(tt.n_locations(), 0U),
        is_dest_(tt.n_locations(), false),
        dist_to_dest_(tt.n_locations(),
                      std::numeric_limits<std::uint16_t>::max()) {}

  void reset() {
    for (auto& r : results_) {
      r.clear();
    }

    raptor_.reset_arrivals();
    starts_.clear();
  }

  timetable const& tt_;

  std::vector<start> starts_;
  std::vector<pareto_set<journey>> results_{tt_.n_locations()};

  day_idx_t base_day_;
  std::vector<std::uint16_t> lb_;
  std::vector<bool> is_dest_;
  std::vector<std::uint16_t> dist_to_dest_;
  raptor_state raptor_state_;
  journey reconstruct_journey_;
  no_route_filter no_route_filter_;
  raptor<no_route_filter,
         direction::kBackward,
         /* Rt= */ false,
         /* OneToAll= */ true>
      raptor_{no_route_filter_, tt_,           nullptr, raptor_state_,
              is_dest_,         dist_to_dest_, lb_,     base_day_};
};

static boost::thread_specific_ptr<state> search_state;

template <typename Partitions>
void update_route_arc_flags(timetable& tt,
                            Partitions&& partitions,
                            journey const& j) {
  for (auto const& l : j.legs_) {
    if (!std::holds_alternative<journey::run_enter_exit>(l.uses_)) {
      continue;
    }

    auto const ree = std::get<journey::run_enter_exit>(l.uses_);
    auto const r = tt.transport_route_[ree.r_.t_.t_idx_];

    for (auto const p : partitions) {
      tt.arc_flags_[r].set(to_idx(p), true);
    }
  }
}

void arc_flags_search(timetable& tt, component_idx_t const c) {
  if (search_state.get() == nullptr) {
    search_state.reset(new state{tt, tt.internal_interval_days().from_});
  }

  auto& state = *search_state;
  state.reset();

  auto q = query{};
  q.start_match_mode_ = location_match_mode::kEquivalent;
  q.dest_match_mode_ = location_match_mode::kEquivalent;
  q.start_ = utl::to_vec(tt.locations_.component_locations_[c],
                         [](location_idx_t const l) {
                           return offset{l, 0_minutes, 0U};
                         });

  get_starts(direction::kBackward, tt, nullptr, tt.external_interval(),
             q.start_, location_match_mode::kEquivalent, true, state.starts_,
             false);

  utl::equal_ranges_linear(
      state.starts_,
      [](start const& a, start const& b) {
        return a.time_at_start_ == b.time_at_start_;
      },
      [&](auto&& from_it, auto&& to_it) {
        state.raptor_.next_start_time();
        auto const start_time = from_it->time_at_start_;

        q.start_time_ = start_time;

        for (auto const& st : it_range{from_it, to_it}) {
          state.raptor_.add_start(st.stop_, st.time_at_stop_);
        }

        auto const worst_time_at_dest = start_time - kMaxTravelTime;
        auto dummy = pareto_set<journey>{};
        state.raptor_.execute(start_time, kMaxTransfers, worst_time_at_dest,
                              dummy);

        // Reconstruct for each target.
        for (auto t = location_idx_t{0U}; t != tt.n_locations(); ++t) {
          if (tt.locations_.components_[t] == c) {
            continue;
          }

          // Collect journeys for each number of transfers.
          for (auto k = 1U; k != kMaxTransfers + 1U; ++k) {
            auto const dest_time =
                state.raptor_state_.round_times_[k][to_idx(t)];
            if (dest_time == kInvalidDelta<direction::kBackward>) {
              continue;
            }
            state.results_[to_idx(t)].add(journey{
                .legs_ = {},
                .start_time_ = start_time,
                .dest_time_ = delta_to_unix(state.raptor_.base(), dest_time),
                .dest_ = t,
                .transfers_ = static_cast<std::uint8_t>(k - 1)});
          }

          // Reconstruct journeys and update reach values.
          for (auto& j : state.results_[to_idx(t)]) {
            if (j.reconstructed_) {
              continue;
            }
            q.destination_ = {{t, 0_minutes, 0U}};
            state.reconstruct_journey_.copy_from(j);
            try {
              state.raptor_.reconstruct(q, state.reconstruct_journey_);
            } catch (std::exception const& e) {
              log(log_lvl::info, "routing.reach",
                  "reconstruct {}@{} to {}@{}: {}", c, j.start_time_,
                  location{tt, t}, j.dest_time_, e.what());
              //                          continue;
              throw;
            }
            update_route_arc_flags(tt, tt.component_partitions_[c],
                                   state.reconstruct_journey_);
            j.reconstructed_ = true;
          }
        }
      });
}

std::string exec(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

void compute_arc_flags(timetable& tt) {
  auto progress_tracker = utl::activate_progress_tracker("arcflags");

  // Write file for hMETIS
  progress_tracker->status("Write hMETIS");
  auto of = std::ofstream{"hmetis.txt"};
  write_hmetis_file(of, tt);

  // Compute partition with hMETIS
  progress_tracker->status("Exec hMETIS");
  auto const ret = exec(
      R"(/home/felix/Downloads/hmetis-1.5-linux/hmetis hmetis.txt 8 15 50 1 1 1 1 0)");
  log(log_lvl::info, "routing.arcflags", "hmetis returned \"{}\"", ret);

  // Read partitions.
  progress_tracker->status("Read hMETIS result");
  auto const file =
      cista::mmap{"hmetis.txt.part.8", cista::mmap::protection::READ};
  utl::for_each_line(file.view(), [&](utl::cstr line) {
    tt.route_partitions_.emplace_back(utl::parse<unsigned>(line));
  });

  // Write location partitions.
  auto component_partitions =
      mutable_fws_multimap<component_idx_t, partition_idx_t>{};
  for (auto const& [c, locations] :
       utl::enumerate(tt.locations_.component_locations_)) {
    for (auto const l : locations) {
      for (auto const r : tt.location_routes_[l]) {
        utl::insert_sorted(component_partitions[component_idx_t{c}],
                           tt.route_partitions_[r]);
      }
    }
  }
  for (auto const& el : component_partitions) {
    tt.component_partitions_.emplace_back(el);
  }

  std::vector<component_idx_t> cut_components;
  for (auto const& [c, cp] : utl::enumerate(tt.component_partitions_)) {
    if (cp.size() > 1U) {
      cut_components.emplace_back(component_idx_t{c});
    }
  }

  progress_tracker->status("Compute arc-flags");
  progress_tracker->out_bounds(0, 100);
  progress_tracker->in_high(cut_components.size());

  tt.arc_flags_.resize(tt.n_routes());

  utl::parallel_for(
      cut_components, [&](component_idx_t const c) { arc_flags_search(tt, c); },
      progress_tracker->increment_fn());
}

void write_hmetis_file(std::ostream& out, timetable const& tt) {
  auto const location_has_routes = [&](auto const& locations) {
    return std::any_of(begin(locations), end(locations),
                       [&](location_idx_t const l) {
                         return !tt.location_routes_[l].empty();
                       });
  };

  auto const n_non_empty_locations = std::count_if(
      begin(tt.locations_.component_locations_),
      end(tt.locations_.component_locations_),
      [&](auto const& locations) { return location_has_routes(locations); });

  out << n_non_empty_locations << " " << tt.n_routes() << " 11\n";

  for (auto const& locations : tt.locations_.component_locations_) {
    if (!location_has_routes(locations)) {
      continue;
    }

    auto n_stop_events = 0U;
    for (auto const l : locations) {
      for (auto const r : tt.location_routes_[l]) {
        for (auto const t : tt.route_transport_ranges_[r]) {
          n_stop_events += tt.bitfields_[tt.transport_traffic_days_[t]].count();
        }
      }
    }
    out << static_cast<unsigned>(std::round(std::log2(n_stop_events))) << " ";

    for (auto const l : locations) {
      for (auto const r : tt.location_routes_[l]) {
        out << (r + 1) << " ";
      }
    }
    out << "\n";
  }

  for (auto const route_transports : tt.route_transport_ranges_) {
    auto runs = 0U;
    for (auto const t : route_transports) {
      runs += tt.bitfields_[tt.transport_traffic_days_[t]].count();
    }
    out << runs << "\n";
  }
}

constexpr auto const marker_fmt_str_start = R"({{
"type": "Feature",
"properties": {{
  "stroke": "{}"
}},
"geometry": {{
  "coordinates": [)";
constexpr auto const marker_fmt_str_end = R"(],
  "type": "LineString"
}},
"id": {}
}},)";

constexpr std::array<std::string_view, 8> kColors = {
    "red"sv,  "blue"sv, "green"sv,  "yellow"sv,
    "cyan"sv, "pink"sv, "orange"sv, "black"sv};

void hmetis_out_to_geojson(std::string_view in,
                           std::ostream& out,
                           timetable const& tt) {
  auto const print_pos = [&](geo::latlng const p) {
    out << "[" << p.lng_ << ", " << p.lat_ << "]";
  };

  auto const print_marker = [&](geo::latlng const& pos) {
    fmt::print(out, R"({{
  "type": "Feature",
  "properties": {{}},
  "geometry": {{
    "coordinates": [ {}, {} ],
    "type": "Point"
  }}
}},)",
               pos.lng_, pos.lat_);
  };

  auto route_partitions = vector_map<route_idx_t, partition_idx_t>{};
  utl::for_each_line(in, [&](utl::cstr line) {
    route_partitions.emplace_back(utl::parse<unsigned>(line));
  });
  utl::verify(route_partitions.size() == tt.n_routes(),
              "invalid partitions size n_route_partitions={} vs n_routes={}",
              route_partitions.size(), tt.n_routes());

  out << R"({
  "type": "FeatureCollection",
  "features": [)";

  // Print line strings for route sequences.
  for (auto const [r, partition] : utl::enumerate(route_partitions)) {
    out << fmt::format(marker_fmt_str_start, kColors[to_idx(partition)]);
    auto first = true;
    for (auto const& stp : tt.route_location_seq_[route_idx_t{r}]) {
      if (!first) {
        out << ", ";
      }
      first = false;
      print_pos(tt.locations_.coordinates_[stop{stp}.location_idx()]);
    }
    out << fmt::format(marker_fmt_str_end, r);
  }

  // Print markers for cut locations.
  auto n_cut_components = 0U;
  for (auto const& locations : tt.locations_.component_locations_) {
    std::optional<partition_idx_t> partition = std::nullopt;
    for (auto const l : locations) {
      for (auto const r : tt.location_routes_[l]) {
        if (partition.has_value() && *partition != route_partitions[r]) {
          print_marker(tt.locations_.coordinates_[l]);
          ++n_cut_components;
          goto next;
        } else if (!partition.has_value()) {
          partition = route_partitions[r];
        }
      }
    }
  next:;
  }

  // Print bounding boxes of location components.
  for (auto const& locations : tt.locations_.component_locations_) {
    geo::box bbox;
    for (auto const l : locations) {
      bbox.extend(tt.locations_.coordinates_[l]);
    }

    out << R"({
"type": "Feature",
"properties": {
  "fill": "#000000",
  "fill-opacity": 0.7
},
"geometry": {
  "coordinates": [
    [)";
    print_pos(bbox.min_);
    out << ", ";
    print_pos({bbox.min_.lat_, bbox.max_.lng_});
    out << ", ";
    print_pos(bbox.max_);
    out << ", ";
    print_pos({bbox.max_.lat_, bbox.min_.lng_});
    out << ", ";
    print_pos(bbox.min_);

    out << R"(]
],
"type": "Polygon"
},
"id": 0
},)";
  }

  out << "  ]\n"
         "}";
}

}  // namespace nigiri::routing