#include <cstdio>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <thread>

#include "boost/program_options.hpp"

#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"
#include "utl/parser/cstr.h"
#include "utl/progress_tracker.h"

#include "nigiri/logging.h"
#include "nigiri/qa/qa.h"
#include "nigiri/query_generator/generator.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/raptor.h"

#ifndef _WIN32
#include <sys/resource.h>
#endif

using namespace nigiri;
using namespace nigiri::routing;

std::vector<std::string> tokenize(std::string_view const str,
                                  char const delimiter) {
  auto tokens = std::vector<std::string>{};
  utl::for_each_token(
      utl::cstr{str.data(), str.size()}, delimiter,
      [&](utl::cstr const t) { tokens.emplace_back(t.str, t.len); });
  return tokens;
}

std::optional<geo::box> parse_bbox(std::string const& str) {
  using namespace geo;

  if (str == "europe") {
    return box{latlng{36.0, -11.0}, latlng{72.0, 32.0}};
  }

  static auto const bbox_regex = std::regex{
      "^[-+]?[0-9]*\\.?[0-9]+,[-+]?[0-9]*\\.?[0-9]+,[-+]?[0-9]*\\.?[0-9]+,[-+]?"
      "[0-9]*\\.?[0-9]+$"};
  if (!std::regex_match(begin(str), end(str), bbox_regex)) {
    return std::nullopt;
  }
  auto const tokens = tokenize(str, ',');
  return box{latlng{std::stod(tokens[0]), std::stod(tokens[1])},
             latlng{std::stod(tokens[2]), std::stod(tokens[3])}};
}

std::optional<geo::latlng> parse_coord(std::string const& str) {
  using namespace geo;

  static auto const coord_regex =
      std::regex{R"(^\([-+]?[0-9]*\.?[0-9]+, [-+]?[0-9]*\.?[0-9]+\))"};
  if (!std::regex_match(begin(str), end(str), coord_regex)) {
    return std::nullopt;
  }
  auto const str_trimmed = std::string_view{begin(str) + 1, end(str) - 2};
  auto const tokens = tokenize(str_trimmed, ',');
  return latlng{std::stod(tokens[0]), std::stod(tokens[1])};
}

void generate_queries(
    std::vector<nigiri::query_generation::start_dest_query>& queries,
    std::uint32_t n_queries,
    nigiri::timetable const& tt,
    query_generation::generator_settings const& gs,
    std::int64_t const seed) {
  auto qg = seed > -1
                ? query_generation::generator{tt, gs,
                                              static_cast<std::uint32_t>(seed)}
                : query_generation::generator{tt, gs};
  queries.reserve(n_queries);
  for (auto i = 0U; i != n_queries; ++i) {
    auto const sdq = qg.random_query();
    if (sdq.has_value()) {
      queries.emplace_back(sdq.value());
    }
  }
}

// Journeys at/over this travel time are not counted as misses in the
// pong-vs-raptor common-interval check: the pong ping's travel time cap
// anchors at the probe, so a journey departing after the probe with travel
// time close to kMaxTravelTime can be invisible to every probe's ping while
// search.h (departure-anchored) still finds it.
constexpr auto const kCheckedMaxTravelTime = routing::kMaxTravelTime - 1_days;

struct compare_stats {
  std::uint64_t missed_by_cmp_{0U};  // ref journeys not covered by cmp pareto
  std::uint64_t missed_by_ref_{0U};  // cmp journeys not covered by ref pareto
  bool compared_{false};  // a cross-check ran -> the counts are meaningful
};

// cross-check two result sets over the same queries: every journey of one
// side must be covered (dominated or equaled) by the other side's pareto set
compare_stats cross_check(
    timetable const& tt,
    std::string const& ref_name,
    std::vector<pareto_set<routing::journey>> const& ref,
    std::string const& cmp_name,
    std::vector<pareto_set<routing::journey>> const& cmp) {
  auto stats = compare_stats{};
  stats.compared_ = true;

  auto const covered = [&](journey const& j, auto const& by) {
    // journey::dominates is direction-aware (bwd journeys have
    // start_time_ > dest_time_)
    return utl::any_of(by, [&](journey const& o) { return o.dominates(j); });
  };
  auto const key = [](journey const& j) {
    return fmt::format("dep={} arr={} transfers={}", j.start_time_,
                       j.dest_time_, j.transfers_);
  };
  for (auto i = std::size_t{0U}; i != ref.size(); ++i) {
    auto cmp_misses = 0, ref_misses = 0;
    for (auto const& r : ref[i]) {
      if (!covered(r, cmp[i])) {
        ++cmp_misses;
      }
    }
    for (auto const& c : cmp[i]) {
      if (!covered(c, ref[i])) {
        ++ref_misses;
      }
    }
    if (cmp_misses != 0 || ref_misses != 0) {
      std::cout << "query #" << i << " " << cmp_name << "_MISSES_" << ref_name
                << "=" << cmp_misses << " " << ref_name << "_MISSES_"
                << cmp_name << "=" << ref_misses << "\n  " << ref_name
                << "-pareto: ";
      for (auto const& r : ref[i]) {
        std::cout << "[" << key(r) << "] ";
      }
      std::cout << "\n  " << cmp_name << "-pareto: ";
      for (auto const& c : cmp[i]) {
        std::cout << "[" << key(c) << "] ";
      }
      std::cout << "\n";
      for (auto const& r : ref[i]) {
        if (!covered(r, cmp[i])) {
          std::cout << "  === MISSED-BY-" << cmp_name << ": " << key(r)
                    << " ===\n";
          r.print(std::cout, tt);
          std::cout << "\n";
        }
      }
      for (auto const& c : cmp[i]) {
        if (!covered(c, ref[i])) {
          std::cout << "  === " << cmp_name << "-EXTRA (not covered by "
                    << ref_name << "): " << key(c) << " ===\n";
          c.print(std::cout, tt);
          std::cout << "\n";
        }
      }
    }
    stats.missed_by_cmp_ += static_cast<std::uint64_t>(cmp_misses);
    stats.missed_by_ref_ += static_cast<std::uint64_t>(ref_misses);
  }
  return stats;
}

// Compare search.h vs pong results for min(N, M) journeys after T.
compare_stats common_interval_check(
    timetable const& tt,
    std::string const& ref_name,
    std::vector<pareto_set<routing::journey>> const& ref,
    std::vector<interval<unixtime_t>> const& ref_iv,
    std::string const& cmp_name,
    std::vector<pareto_set<routing::journey>> const& cmp,
    std::vector<interval<unixtime_t>> const& cmp_iv) {
  auto stats = compare_stats{};
  stats.compared_ = true;

  auto const covered = [&](journey const& j, auto const& by) {
    return utl::any_of(by, [&](journey const& o) { return o.dominates(j); });
  };
  auto const key = [](journey const& j) {
    return fmt::format("dep={} arr={} transfers={}", j.departure_time(),
                       j.arrival_time(), j.transfers_);
  };

  for (auto i = std::size_t{0U}; i != ref.size(); ++i) {
    auto const iv =
        interval<unixtime_t>{std::max(ref_iv[i].from_, cmp_iv[i].from_),
                             std::min(ref_iv[i].to_, cmp_iv[i].to_)};
    if (iv.from_ >= iv.to_) {
      continue;  // nothing searched by both -> nothing to compare
    }
    auto misses = 0U;
    auto const check_side = [&](auto const& side, auto const& other,
                                std::string const& side_name,
                                std::string const& other_name) {
      for (auto const& j : side) {
        if (j.travel_time() >= kCheckedMaxTravelTime) {
          continue;  // probe-anchored pong may miss these, see constant above
        }
        if (iv.contains(j.start_time_) && !covered(j, other)) {
          ++misses;
          std::cout << "query #" << i << " " << other_name << " misses ["
                    << key(j) << "] found by " << side_name
                    << " (common interval " << iv << ", " << ref_name << " "
                    << ref_iv[i] << ", " << cmp_name << " " << cmp_iv[i]
                    << ")\n";
          j.print(std::cout, tt);
          std::cout << "\n";
        }
      }
    };
    check_side(ref[i], cmp[i], ref_name, cmp_name);
    check_side(cmp[i], ref[i], cmp_name, ref_name);
    stats.missed_by_cmp_ += misses;
  }
  return stats;
}

// one worker thread per state, pulling queries from a shared counter
template <typename WS, typename SearchFn>
std::vector<double> run_load(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    std::string const& tag,
    std::vector<WS*> const& states,
    SearchFn search_one) {
  if (!queries.empty()) {
    // Warm up (allocate search state).
    for (auto* s : states) {
      search_one(*s, queries.front().q_, std::size_t{0U});
    }
  }

  auto next = std::atomic<std::size_t>{0};
  auto done = std::atomic<std::size_t>{0};
  auto lat = std::vector<double>(queries.size(), -1.0);
  auto const t0 = std::chrono::steady_clock::now();
  auto workers = std::vector<std::thread>{};
  for (auto* ws : states) {
    workers.emplace_back([&, ws]() {
      for (auto i = next.fetch_add(1); i < queries.size();
           i = next.fetch_add(1)) {
        try {
          auto const q0 = std::chrono::steady_clock::now();
          search_one(*ws, queries[i].q_, i);
          lat[i] = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - q0)
                       .count();
          done.fetch_add(1);
        } catch (std::exception const& e) {
          std::cerr << "q#" << i << " FAILED: " << e.what() << std::endl;
        }
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  auto const ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  auto const d = done.load();
  auto const qps =
      d * 1000.0 / static_cast<double>(std::max<std::int64_t>(ms, 1));

  auto l = lat;
  std::erase_if(l, [](double const x) { return x < 0.0; });
  std::sort(begin(l), end(l));
  auto const q = [&](double const p) {
    return l.empty() ? 0.0
                     : l[std::min(l.size() - 1,
                                  static_cast<std::size_t>(p * l.size()))];
  };
  auto const avg =
      l.empty() ? 0.0 : std::accumulate(begin(l), end(l), 0.0) / l.size();
  fmt::print(
      "| {:<36} | {:>6.1f} | {:>6.0f} | {:>6.0f} | {:>6.0f} | "
      "{:>6.0f} |\n",
      tag, qps, avg, q(0.50), q(0.90), q(0.99));
  return lat;
}

struct cell_result {
  compare_stats stats_;
  std::vector<pareto_set<routing::journey>> cpu_res_;
  std::vector<interval<unixtime_t>> cpu_iv_;  // final searched interval
  std::vector<double> cpu_lat_;
};

cell_result run_config(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    timetable const& tt,
    std::string const& label,
    std::string const& algo,
    direction const dir,
    bool const run_cpu,
    [[maybe_unused]] bool const run_gpu,
    std::vector<unsigned> const& cpu_threads_v,
    [[maybe_unused]] std::vector<unsigned> const& gpu_states_v) {
  auto out = cell_result{};
  auto& stats = out.stats_;

  auto const use_pong = algo != "raptor";

  // the sweep's state pool is allocated once at the maximum count; each
  // sweep point borrows a prefix of it
  auto const prefix = []<typename WS>(
                          std::vector<std::unique_ptr<WS>> const& owned,
                          unsigned const n) {
    auto pool = std::vector<WS*>{};
    for (auto i = 0U; i != n; ++i) {
      pool.push_back(owned[i].get());
    }
    return pool;
  };

  struct cpu_ws {
    search_state ss_;
    routing::raptor_state rs_;
  };
  auto& cpu_res = out.cpu_res_;
  cpu_res.resize(queries.size());
  out.cpu_iv_.resize(queries.size());
  if (run_cpu) {
    auto states = std::vector<std::unique_ptr<cpu_ws>>{};
    for (auto i = 0U;
         i != *std::max_element(begin(cpu_threads_v), end(cpu_threads_v));
         ++i) {
      states.push_back(std::make_unique<cpu_ws>());
    }

    for (auto const t : cpu_threads_v) {
      out.cpu_lat_ = run_load<cpu_ws>(
          queries, label + "-cpu-" + std::to_string(t), prefix(states, t),
          [&](cpu_ws& w, routing::query q, std::size_t const i) {
            auto const r =
                use_pong ? routing::pong_search(tt, nullptr, w.ss_, w.rs_,
                                                std::move(q), dir)
                         : routing::raptor_search(tt, nullptr, w.ss_, w.rs_,
                                                  std::move(q), dir);
            cpu_res[i] = *r.journeys_;
            out.cpu_iv_[i] = r.interval_;
          });
    }
  }

#if defined(NIGIRI_CUDA)
  auto gpu_res = std::vector<pareto_set<routing::journey>>(queries.size());
  if (run_gpu) {
    auto const gpu_tt = routing::gpu::gpu_timetable{tt};
    struct gpu_ws {
      search_state ss_;
      std::unique_ptr<routing::gpu::gpu_raptor_state> rs_;
    };
    auto states = std::vector<std::unique_ptr<gpu_ws>>{};
    for (auto i = 0U;
         i != *std::max_element(begin(gpu_states_v), end(gpu_states_v)); ++i) {
      states.push_back(std::make_unique<gpu_ws>(
          gpu_ws{search_state{},
                 std::make_unique<routing::gpu::gpu_raptor_state>(gpu_tt)}));
    }
    for (auto const s : gpu_states_v) {
      run_load<gpu_ws>(
          queries, label + "-gpu-" + std::to_string(s), prefix(states, s),
          [&](gpu_ws& w, routing::query q, std::size_t const i) {
            auto const r =
                use_pong ? routing::pong_search(tt, nullptr, w.ss_, *w.rs_,
                                                std::move(q), dir)
                         : routing::raptor_search(tt, nullptr, w.ss_, *w.rs_,
                                                  std::move(q), dir);
            gpu_res[i] = *r.journeys_;
          });
    }
  }

  if (!run_cpu || !run_gpu) {
    return out;  // single engine -> nothing to compare
  }
  stats = cross_check(tt, "CPU", cpu_res, "GPU", gpu_res);
#endif

  return out;
}

void print_memory_usage() {
#ifndef _WIN32
  auto r = rusage{};
  getrusage(RUSAGE_SELF, &r);
  std::cout << "\n--- memory usage ---\nrusage.ru_maxrss: "
            << static_cast<double>(r.ru_maxrss) / (1024 * 1024) << " GiB\n";
#endif
}

int main(int argc, char* argv[]) {
  // CI pipes stdout -> fully buffered by default; line-buffer so progress
  // (table rows) appears immediately (std::cout syncs with stdio)
  setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);

  namespace bpo = boost::program_options;

  auto tt_path = std::filesystem::path{};
  auto n_queries = std::uint32_t{100U};
  auto gs = query_generation::generator_settings{};
  auto interval_size = duration_t::rep{};
  auto bbox_str = std::string{};
  auto intermodal_start_str = std::string{};
  auto intermodal_dest_str = std::string{};
  auto max_transfers = std::uint32_t{kMaxTransfers};
  auto prf_idx = std::uint32_t{0};
  auto start_coord_str = std::string{};
  auto dest_coord_str = std::string{};
  auto start_loc_val = location_idx_t::value_t{0U};
  auto dest_loc_val = location_idx_t::value_t{0U};
  auto seed = std::int64_t{0};
  auto min_transfer_time = duration_t::rep{};
  auto qa_path = std::filesystem::path{};
  auto engines = std::vector<std::string>{};
  auto algos = std::vector<std::string>{};
  auto modes = std::vector<std::string>{};
  auto dirs = std::vector<std::string>{};
  auto threads_v = std::vector<unsigned>{};
  auto gpu_states_v = std::vector<unsigned>{};

  bpo::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce this help message")  //
      ("tt_path,p", bpo::value(&tt_path)->required(),
       "path to a binary file containing a serialized nigiri timetable")  //
      ("engines", bpo::value(&engines)->multitoken(),
       "engines to benchmark (default: cpu gpu); every axis is a vector -- "
       "the run is the full cross product of engines x algos x modes (x "
       "threads/states within an engine), all against the once-loaded "
       "timetable, with one PROFILE throughput/latency line per point; "
       "whenever BOTH engines ran a (mode, algo) cell, their pareto sets are "
       "cross-checked per query and the process exits non-zero on any "
       "divergence")  //
      ("algo,a", bpo::value(&algos)->multitoken(),
       "algorithms: raptor | pong (default: both); if both ran with the cpu "
       "engine, the pong cell of each (mode, dir) is checked against raptor "
       "for agreement on the intersection of the final search intervals")  //
      ("modes", bpo::value(&modes)->multitoken(),
       "<start>-<dest> query modes with station | coordinate, e.g. "
       "station-station coordinate-coordinate (default: those two); "
       "coordinate = intermodal offsets (walk)")  //
      ("dirs", bpo::value(&dirs)->multitoken(),
       "search directions: fwd | bwd (default: fwd); bwd flips the generated "
       "queries (start/dest swapped, vias reversed) and searches backward = "
       "arriveBy with the interval as arrival window; the interval extension "
       "flags are forced to the search direction (fwd: later only, bwd: "
       "earlier only), overriding -e/-l")  //
      ("threads", bpo::value(&threads_v)->multitoken(),
       "CPU worker thread counts to sweep (default: hardware "
       "concurrency)")  //
      ("gpu_states", bpo::value(&gpu_states_v)->multitoken(),
       "concurrent GPU pipeline counts to sweep (default: 2)")  //
      ("seed,s", bpo::value<std::int64_t>(&seed)->default_value(seed),
       "query generator RNG seed, -1 for a random seed")  //
      ("num_queries,n", bpo::value(&n_queries)->default_value(n_queries),
       "number of queries to generate/process")(
          "interval_size,i",
          bpo::value<duration_t::rep>(&interval_size)->default_value(60U, "60"),
          "the initial size of the search interval in minutes, set to 0 for "
          "ontrip queries")  //
      ("bounding_box,b", bpo::value<std::string>(&bbox_str),
       "limit randomized locations to a bounding box, "
       "format: lat_min,lon_min,lat_max,lon_max\ne.g., 36.0,-11.0,72.0,32.0\n"
       "(available via \"-b europe\")")  //
      ("intermodal_start",
       bpo::value<std::string>(&intermodal_start_str)->default_value("walk"),
       "first-mile transport mode for coordinate-* --modes: "
       "walk | bicycle | car")  //
      ("intermodal_dest",
       bpo::value<std::string>(&intermodal_dest_str)->default_value("walk"),
       "last-mile transport mode for *-coordinate --modes: "
       "walk | bicycle | car")  //
      ("use_start_footpaths",
       bpo::value<bool>(&gs.use_start_footpaths_)->default_value(true),
       "")  //
      ("max_transfers,t",
       bpo::value<std::uint32_t>(&max_transfers)->default_value(kMaxTransfers),
       "maximum number of transfers during routing")  //
      ("min_connection_count,m",
       bpo::value<std::uint32_t>(&gs.min_connection_count_)->default_value(5U),
       "the minimum number of connections to find with each query")  //
      ("extend_interval_earlier,e",
       bpo::value<bool>(&gs.extend_interval_earlier_)
           ->default_value(true, "true"),
       "allows extension of the search interval into the past")  //
      ("extend_interval_later,l",
       bpo::value<bool>(&gs.extend_interval_later_)
           ->default_value(true, "true"),
       "allows extension of the search interval into the future")  //
      ("profile_idx", bpo::value<std::uint32_t>(&prf_idx)->default_value(0U),
       "footpath profile index")  //
      ("allowed_claszes",
       bpo::value<clasz_mask_t>(&gs.allowed_claszes_)
           ->default_value(routing::all_clasz_allowed()),
       "")  //
      ("min_transfer_time",
       bpo::value<duration_t::rep>(&min_transfer_time)->default_value(0U),
       "minimum transfer time in minutes")  //
      ("transfer_time_factor",
       bpo::value<float>(&gs.transfer_time_settings_.factor_)
           ->default_value(1.0F),
       "multiply all transfer times by this factor")  //
      ("vias", bpo::value<unsigned>(&gs.n_vias_)->default_value(0U),
       "number of via stops")  //
      ("start_coord", bpo::value<std::string>(&start_coord_str),
       "start coordinate for random queries, format: \"(LAT, LON)\", "  //
       "where LAT/LON are given in decimal degrees")  //
      ("dest_coord", bpo::value<std::string>(&dest_coord_str),
       "destination coordinate for random queries, format: \"(LAT, LON)\", "  //
       "where LAT/LON are given in decimal degrees")  //
      ("start_loc", bpo::value<location_idx_t::value_t>(&start_loc_val),
       "start location for random queries")  //
      ("dest_loc", bpo::value<location_idx_t::value_t>(&dest_loc_val),
       "destination location for random queries")  //
      ("qa_path,q", bpo::value(&qa_path),
       "path to write the journey criteria to for qa");
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);

  // process program options - begin
  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  bpo::notify(vm);

  std::cout << "loading timetable...\n";
  auto tt = *nigiri::timetable::read(tt_path);
  tt.resolve();

  gs.interval_size_ = duration_t{interval_size};

  if (!bbox_str.empty()) {
    gs.bbox_ = parse_bbox(bbox_str);
    if (!gs.bbox_.has_value()) {
      std::cout << "Error: malformed bounding box input\n";
      return 1;
    }
  }

  // transport modes of the first/last mile for coordinate-* / *-coordinate
  // --modes (the match modes themselves come from the mode tokens)
  auto const intermodal_start_mode =
      query_generation::to_transport_mode(intermodal_start_str);
  auto const intermodal_dest_mode =
      query_generation::to_transport_mode(intermodal_dest_str);
  if (!intermodal_start_mode || !intermodal_dest_mode) {
    std::cerr << "Error: unknown intermodal start/dest mode\n";
    return 1;
  }
  gs.start_mode_ = *intermodal_start_mode;
  gs.dest_mode_ = *intermodal_dest_mode;

  gs.max_transfers_ = max_transfers > std::numeric_limits<std::uint8_t>::max()
                          ? std::numeric_limits<std::uint8_t>::max()
                          : max_transfers;

  gs.transfer_time_settings_.min_transfer_time_ = duration_t{min_transfer_time};
  gs.transfer_time_settings_.default_ =
      min_transfer_time == 0U && gs.transfer_time_settings_.factor_ == 1.0F;

  if (vm.count("profile_idx") != 0) {
    if (prf_idx >= kNProfiles) {
      std::cout << "Error: profile idx exceeds numeric limits\n";
      return 1;
    }
    gs.prf_idx_ = prf_idx;
  }

  if (!start_coord_str.empty()) {
    gs.start_match_mode_ = location_match_mode::kIntermodal;
    auto const start_coord = parse_coord(start_coord_str);
    if (start_coord.has_value()) {
      gs.start_ = start_coord.value();
    } else {
      std::cout << "Error: Invalid start coordinate\n";
      return 1;
    }
  }

  if (!dest_coord_str.empty()) {
    gs.dest_match_mode_ = location_match_mode::kIntermodal;
    auto const dest_coord = parse_coord(dest_coord_str);
    if (dest_coord.has_value()) {
      gs.dest_ = dest_coord.value();
    } else {
      std::cout << "Error: Invalid destination coordinate\n";
      return 1;
    }
  }

  if (start_loc_val != 0U) {
    gs.start_match_mode_ = location_match_mode::kEquivalent;
    gs.start_ = location_idx_t{start_loc_val};
  }

  if (dest_loc_val != 0U) {
    gs.dest_match_mode_ = location_match_mode::kEquivalent;
    gs.dest_ = location_idx_t{dest_loc_val};
  }
  // process program options - end

  // ---- benchmark matrix: engines x algos x modes (x threads/states) ----
  if (engines.empty()) {
    engines = {"cpu", "gpu"};
  }
  if (algos.empty()) {
    algos = {"raptor", "pong"};
  }
  if (modes.empty()) {
    modes = {"station-station", "coordinate-coordinate"};
  }
  if (dirs.empty()) {
    dirs = {"fwd"};
  }
  for (auto const& d : dirs) {
    if (d != "fwd" && d != "bwd") {
      std::cerr << "invalid dir \"" << d << "\", expected fwd | bwd\n";
      return 1;
    }
  }
  if (threads_v.empty()) {
    threads_v = {std::max(std::thread::hardware_concurrency(), 1U)};
  }
  if (gpu_states_v.empty()) {
    gpu_states_v = {2U};
  }

  auto run_cpu = false, run_gpu = false;
  for (auto const& e : engines) {
    if (e == "cpu") {
      run_cpu = true;
    } else if (e == "gpu") {
      run_gpu = true;
    } else {
      std::cerr << "invalid engine \"" << e << "\", expected cpu | gpu\n";
      return 1;
    }
  }
#if !defined(NIGIRI_CUDA)
  if (run_gpu) {
    if (!run_cpu) {
      std::cerr << "--engines gpu requires a NIGIRI_CUDA build\n";
      return 1;
    }
    std::cout << "NIGIRI_CUDA not enabled -> running CPU only\n";
    run_gpu = false;
  }
#endif
  for (auto const& a : algos) {
    if (a != "raptor" && a != "pong") {
      std::cerr << "invalid algo \"" << a << "\", expected raptor | pong\n";
      return 1;
    }
  }

  // apply one end of a <start>-<dest> mode token to the generator settings
  // (the first/last-mile transport modes come from --intermodal_start/_dest)
  auto const apply_mode = [](std::string const& m, location_match_mode& match) {
    if (m == "station") {
      match = location_match_mode::kEquivalent;
      return true;
    }
    if (m == "coordinate" || m == "intermodal") {
      match = location_match_mode::kIntermodal;
      return true;
    }
    return false;
  };

  // padded markdown: renders as a table AND stays aligned as plain text;
  // one table for the whole matrix
  fmt::print("| {:<36} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} |\n",  //
             "config", "q/s", "avg ms", "median", "q90", "q99");
  fmt::print(
      "| {0:-<36} | {0:->5}: | {0:->5}: | {0:->5}: | {0:->5}: | "
      "{0:->5}: |\n",
      "");

  auto mode_queries =
      std::map<std::string,
               std::vector<nigiri::query_generation::start_dest_query>>{};
  auto summary = std::vector<std::string>{};
  auto total = compare_stats{};
  auto qa_cell = std::optional<cell_result>{};
  auto qa_n_cells = 0U;

  for (auto const& mode : modes) {
    auto rs = gs;
    auto const sep = mode.find('-');
    if (sep == std::string::npos ||
        !apply_mode(mode.substr(0, sep), rs.start_match_mode_) ||
        !apply_mode(mode.substr(sep + 1U), rs.dest_match_mode_)) {
      std::cerr << "invalid mode \"" << mode
                << "\", expected <station|coordinate>-<station|coordinate>\n";
      return 1;
    }
    if (rs.start_match_mode_ == location_match_mode::kIntermodal) {
      rs.use_start_footpaths_ = false;  // first mile is in the start offsets
    }

    auto& fwd_qs = mode_queries[mode];
    if (fwd_qs.empty()) {
      generate_queries(fwd_qs, n_queries, tt, rs, seed);
    }

    for (auto const& dir_str : dirs) {
      auto const dir =
          dir_str == "fwd" ? direction::kForward : direction::kBackward;

      // bwd = the same workload flipped: start/dest swapped, vias reversed,
      // the generated interval becomes the arrival window (arriveBy).
      // The interval only ever extends in the search direction (later for
      // fwd, earlier for bwd) -- production never sets both flags and the
      // pong probes only march that way.
      auto qs = fwd_qs;
      for (auto& sdq : qs) {
        if (dir == direction::kBackward) {
          sdq.q_.flip_dir();
        }
        sdq.q_.extend_interval_earlier_ = dir == direction::kBackward;
        sdq.q_.extend_interval_later_ = dir == direction::kForward;
      }

      // per (mode, dir): raptor cell = common-interval check reference
      struct check_ref {
        std::string label_;
        std::vector<pareto_set<routing::journey>> res_;
        std::vector<interval<unixtime_t>> iv_;
      };
      auto raptor_ref = std::optional<check_ref>{};

      for (auto const& algo : algos) {
        auto const label = mode + "-" + dir_str + "-" + algo;
        auto cell = cell_result{};
        try {
          cell = run_config(qs, tt, label, algo, dir, run_cpu, run_gpu,
                            threads_v, gpu_states_v);
        } catch (std::exception const& e) {
          // e.g. GPU state allocation OOM -- report + fail instead of dying
          std::cerr << "RUN " << label << " failed: " << e.what() << "\n";
          summary.push_back(
              fmt::format("{:<44} EXCEPTION: {}", label, e.what()));
          total.missed_by_cmp_ += 1U;
          continue;
        }

        summary.push_back(
            cell.stats_.compared_
                ? fmt::format(
                      "{:<44} n={:<6} gpu_misses_cpu={:<4} "
                      "cpu_misses_gpu={:<4} {}",
                      label, qs.size(), cell.stats_.missed_by_cmp_,
                      cell.stats_.missed_by_ref_,
                      cell.stats_.missed_by_cmp_ + cell.stats_.missed_by_ref_ ==
                              0U
                          ? "PASS"
                          : "FAIL")
                : fmt::format("{:<44} n={:<6} benchmark only ({})", label,
                              qs.size(), run_cpu ? "cpu" : "gpu"));
        total.missed_by_cmp_ += cell.stats_.missed_by_cmp_;
        total.missed_by_ref_ += cell.stats_.missed_by_ref_;

        if (run_cpu) {
          if (algo == "raptor") {
            raptor_ref.emplace(check_ref{label, cell.cpu_res_, cell.cpu_iv_});
          } else if (raptor_ref.has_value()) {
            // strict agreement with search.h raptor on the region both
            // searched (intersection of the final intervals)
            auto const st = common_interval_check(
                tt, raptor_ref->label_, raptor_ref->res_, raptor_ref->iv_,
                label, cell.cpu_res_, cell.cpu_iv_);
            summary.push_back(
                fmt::format("{:<44} vs {:<40} common-interval misses={:<4} {}",
                            label, raptor_ref->label_, st.missed_by_cmp_,
                            st.missed_by_cmp_ == 0U ? "PASS" : "FAIL"));
            total.missed_by_cmp_ += st.missed_by_cmp_;
          }
        }

        ++qa_n_cells;
        qa_cell = std::move(cell);
      }
    }
  }

  std::cout << "\n=== SUMMARY ===\n";
  for (auto const& s : summary) {
    std::cout << s << "\n";
  }
  print_memory_usage();

  if (vm.count("qa_path")) {
    // qa export needs ONE well-defined result set: exactly one (mode, algo)
    // cell with the CPU engine.
    if (qa_n_cells != 1U || !run_cpu || !qa_cell.has_value()) {
      std::cerr << "--qa_path requires exactly one (mode, algo) cell with the "
                   "cpu engine (single-element --algo/--modes)\n";
      return 1;
    }
    auto bm_crit = nigiri::qa::benchmark_criteria{};
    for (auto i = std::size_t{0U}; i != qa_cell->cpu_res_.size(); ++i) {
      auto jc = vector<nigiri::qa::criteria_t>{};
      for (auto const& j : qa_cell->cpu_res_[i]) {
        jc.emplace_back(
            static_cast<double>(j.start_time_.time_since_epoch().count()),
            static_cast<double>(j.dest_time_.time_since_epoch().count()),
            static_cast<double>(j.transfers_));
      }
      utl::sort(jc);
      bm_crit.qc_.emplace_back(
          i,
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::duration<double, std::milli>{std::max(
                  qa_cell->cpu_lat_.size() > i ? qa_cell->cpu_lat_[i] : 0.0,
                  0.0)}),
          jc);
    }
    bm_crit.write(qa_path);
  }

  return total.missed_by_cmp_ + total.missed_by_ref_ == 0U ? 0 : 1;
}
