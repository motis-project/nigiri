#include <cstdio>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <span>
#include <thread>

#include "boost/program_options.hpp"

#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"
#include "utl/parser/cstr.h"
#include "utl/progress_tracker.h"
#include "utl/zip.h"

#include "nigiri/logging.h"
#include "nigiri/qa/qa.h"
#include "nigiri/query_generator/generator.h"
#include "nigiri/routing/interval_estimate.h"
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

// Range-RAPTOR start time == first trip's departure time
// Pong start time != first trip's departure time
// -> travel time is not measured from departure
// -> give Pong some slack
constexpr auto const kCheckedMaxTravelTime = routing::kMaxTravelTime - 1_days;

std::uint64_t compare_results(
    timetable const& tt,
    std::string const& ref_name,
    std::vector<pareto_set<routing::journey>> const& ref,
    std::string const& cmp_name,
    std::vector<pareto_set<routing::journey>> const& cmp,
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    direction const search_dir,
    unsigned const min_connection_count) {
  auto mismatches = std::uint64_t{0U};

  auto const equal = [](journey const& a, journey const& b) {
    return a.start_time_ == b.start_time_ && a.dest_time_ == b.dest_time_ &&
           a.transfers_ == b.transfers_;
  };

  auto const key = [](journey const& j) {
    return fmt::format("dep={} arr={} transfers={}", j.departure_time(),
                       j.arrival_time(), j.transfers_);
  };

  auto const max_window = [&](query const& q) {
    return search_dir == direction::kForward
               ? interval_estimator<direction::kForward>{tt, q}.max_interval()
               : interval_estimator<direction::kBackward>{tt, q}.max_interval();
  };

  auto const filtered = [](pareto_set<routing::journey> const& set,
                           interval<unixtime_t> const& window) {
    auto v = std::vector<journey const*>{};
    for (auto const& j : set) {
      if (j.travel_time() < kCheckedMaxTravelTime /* slack for pong */ &&
          window.contains(j.start_time_) /* range raptor search limit */) {
        v.push_back(&j);
      }
    }
    return v;
  };

  auto const print_set = [&](std::string const& name, auto const& journeys) {
    fmt::print("  {}: ", name);
    for (auto const* j : journeys) {
      fmt::print("[{}] ", key(*j));
    }
    fmt::println("");
  };

  for (auto i = std::size_t{0U}; i != ref.size(); ++i) {
    auto const window = max_window(queries[i].q_);
    auto r = filtered(ref[i], window);
    auto c = filtered(cmp[i], window);
    if (search_dir == direction::kBackward) {
      std::reverse(begin(r), end(r));
      std::reverse(begin(c), end(c));
    }

    auto const r_size = r.size();
    auto const c_size = c.size();
    auto const n = std::min(r_size, c_size);
    auto const r_zip = std::span{r.data(), n};
    auto const c_zip = std::span{c.data(), n};

    // Count under-deliverying results:
    // >= min_connection_count reached by one but not the other
    auto const raw_r = ref[i].size();
    auto const raw_c = cmp[i].size();
    auto misses = std::uint64_t{0U};
    if (r_size >= min_connection_count && raw_c < min_connection_count) {
      misses += min_connection_count - raw_c;
    }
    if (c_size >= min_connection_count && raw_r < min_connection_count) {
      misses += min_connection_count - raw_r;
    }

    // Count inequalities.
    for (auto const [a, b] : utl::zip(r_zip, c_zip)) {
      if (!equal(*a, *b)) {
        ++misses;
      }
    }

    if (misses != 0U) {
      fmt::println("query #{} mismatches={} ({} n={}, {} n={})", i, misses,
                   ref_name, r_size, cmp_name, c_size);
      auto const loc_str = [&](std::variant<location_idx_t, geo::latlng> const&
                                   v) {
        if (std::holds_alternative<location_idx_t>(v)) {
          auto const l = std::get<location_idx_t>(v);
          return fmt::format("{} idx={}", tt.locations_.ids_[l].view(), l);
        }
        return fmt::format("({}, {})", std::get<geo::latlng>(v).lat_,
                           std::get<geo::latlng>(v).lng_);
      };
      auto const& q = queries[i].q_;
      auto const time_str =
          std::holds_alternative<interval<unixtime_t>>(q.start_time_)
              ? fmt::format(
                    "[{}, {}] epoch=[{}, {}]",
                    std::get<interval<unixtime_t>>(q.start_time_).from_,
                    std::get<interval<unixtime_t>>(q.start_time_).to_,
                    std::get<interval<unixtime_t>>(q.start_time_)
                        .from_.time_since_epoch()
                        .count(),
                    std::get<interval<unixtime_t>>(q.start_time_)
                        .to_.time_since_epoch()
                        .count())
              : fmt::format("{}", std::get<unixtime_t>(q.start_time_));
      fmt::println("  QUERY from={} to={} start_time={} window=[{}, {}]",
                   loc_str(queries[i].start_), loc_str(queries[i].dest_),
                   time_str, window.from_, window.to_);
      fmt::print("  RAW {}: ", ref_name);
      for (auto const& j : ref[i]) {
        fmt::print("[{}] ", key(j));
      }
      fmt::println("");
      fmt::print("  RAW {}: ", cmp_name);
      for (auto const& j : cmp[i]) {
        fmt::print("[{}] ", key(j));
      }
      fmt::println("");
      print_set(ref_name, r);
      print_set(cmp_name, c);
      for (auto const [a, b] : utl::zip(r_zip, c_zip)) {
        if (!equal(*a, *b)) {
          fmt::println("  === MISMATCH: {} [{}] vs {} [{}] ===", ref_name,
                       key(*a), cmp_name, key(*b));
          a->print(std::cout, tt);
          fmt::println("");
          b->print(std::cout, tt);
          fmt::println("");
        }
      }
    }

    mismatches += misses;
  }

  return mismatches;
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

struct result_set {
  std::string label_;
  std::vector<pareto_set<routing::journey>> res_;
  std::vector<std::array<std::uint64_t, 2>> stats_;  // routes, fps visited
  std::vector<double> latencies_;
};

struct cpu_ws {
  search_state ss_;
  routing::raptor_state rs_;
};

#if defined(NIGIRI_CUDA)
struct gpu_ws {
  explicit gpu_ws(routing::gpu::gpu_timetable const& gtt)
      : rs_{std::make_unique<routing::gpu::gpu_raptor_state>(gtt)} {}
  search_state ss_;
  std::unique_ptr<routing::gpu::gpu_raptor_state> rs_;
};
#endif

// one (engine, algo) cell: runs the queries once per n_parallel value (each
// worker borrows a state from a pool allocated once at the maximum count)
// and keeps the last run's journeys + latencies
template <typename WS, typename Search, typename... StateArgs>
result_set run_cell(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    std::string const& label,
    std::vector<unsigned> const& n_parallel,
    Search&& search,
    StateArgs const&... state_args) {
  auto out = result_set{.label_ = label};
  out.res_.resize(queries.size());
  out.stats_.resize(queries.size());

  auto states = std::vector<std::unique_ptr<WS>>{};
  for (auto i = 0U; i != *std::max_element(begin(n_parallel), end(n_parallel));
       ++i) {
    states.push_back(std::make_unique<WS>(state_args...));
  }

  for (auto const n : n_parallel) {
    auto pool = std::vector<WS*>{};
    for (auto i = 0U; i != n; ++i) {
      pool.push_back(states[i].get());
    }
    out.latencies_ =
        run_load<WS>(queries, label + "-" + std::to_string(n), pool,
                     [&](WS& w, routing::query q, std::size_t const i) {
                       auto [js, st] = search(w, std::move(q));
                       out.res_[i] = std::move(js);
                       out.stats_[i] = st;
                     });
  }

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
  setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);  // line buffering for CI

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
  auto engines = std::vector<std::string>{"cpu", "gpu"};
  auto algos = std::vector<std::string>{"range", "pong"};
  auto modes = std::vector<std::string>{"s2s", "c2c"};
  auto dirs = std::vector<std::string>{"fwd"};
  auto threads_v =
      std::vector<unsigned>{std::max(std::thread::hardware_concurrency(), 1U)};
  auto gpu_states_v = std::vector<unsigned>{2U};

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
       "<start>2<dest> query modes with s = station, c = coordinate: "
       "s2s | s2c | c2s | c2c (default: s2s c2c); c = intermodal offsets "
       "(walk)")  //
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
  if (auto const* id = std::getenv("NIGIRI_FPAT"); id != nullptr) {
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.ids_[l].view() != std::string_view{id}) { continue; }
      for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
        if (tt.locations_.footpaths_out_[p].size() <= to_idx(l)) { continue; }
        auto n = 0U;
        for (auto const& fp : tt.locations_.footpaths_out_[p][l]) {
          ++n;
          if (tt.locations_.ids_[fp.target()].view().find("3000") != std::string_view::npos) {
            fmt::print("prf={} {} -> {} ({} min)\n", p, id,
                       tt.locations_.ids_[fp.target()].view(), fp.duration().count());
          }
        }
        fmt::print("prf={} total_out={}\n", p, n);
      }
    }
    return 0;
  }

  fmt::print("timetable: locations={} routes={} transports={}\n",
             tt.n_locations(), tt.n_routes(), tt.transport_traffic_days_.size());

  if (std::getenv("NIGIRI_SCAN_STAT") != nullptr) {
    auto loc_routes = std::size_t{0U};
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      loc_routes += tt.location_routes_[l].size();
    }
    auto seq = std::size_t{0U};
    for (auto r = route_idx_t{0U}; r != route_idx_t{tt.n_routes()}; ++r) {
      seq += tt.route_location_seq_[r].size();
    }
    fmt::print("locations={} routes={} location_route_entries={} route_stops={}\n",
               tt.n_locations(), tt.n_routes(), loc_routes, seq);
    for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
      auto n = std::size_t{0U}, longest = std::size_t{0U};
      for (auto l = location_idx_t{0U};
           l != location_idx_t{tt.locations_.footpaths_out_[p].size()}; ++l) {
        n += tt.locations_.footpaths_out_[p][l].size();
        longest = std::max(longest,
                           static_cast<std::size_t>(
                               tt.locations_.footpaths_out_[p][l].size()));
      }
      if (n != 0U) {
        fmt::print("  prf={} footpaths={} longest_list={}\n", p, n, longest);
      }
    }
    return 0;
  }
  if (auto const* id = std::getenv("NIGIRI_FP_DUMP"); id != nullptr &&
      std::string_view{id} == "ALL") {
    auto const nm = [&](location_idx_t const l) {
      auto const v = tt.locations_.ids_[l].view();
      return v.empty() ? fmt::format("#{}", to_idx(l)) : std::string{v};
    };
    fmt::print("locations={}\n", tt.n_locations());
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      auto const p = tt.locations_.parents_[l];
      fmt::print("loc {:<3} {:<44} type={} parent={:<14} xfer={} base={}\n",
                 to_idx(l), nm(l),
                 static_cast<int>(tt.locations_.types_[l]),
                 p == location_idx_t::invalid()
                     ? std::string{"-"}
                     : std::string{tt.locations_.ids_[p].view()},
                 tt.locations_.transfer_time_[l].count(),
                 l < location_idx_t{tt.locations_.base_transfer_time_.size()}
                     ? tt.locations_.base_transfer_time_[l].count()
                     : -1);
    }
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      for (auto const& fp : tt.locations_.footpaths_out_[kDefaultProfile][l]) {
        fmt::print("fp  {} -> {} ({} min)\n", nm(l), nm(fp.target()),
                   fp.duration().count());
      }
    }
    fmt::print("hubs={}\n", tt.locations_.hub_time_[kDefaultProfile].size());
    for (auto h = hub_idx_t{0U};
         h != hub_idx_t{tt.locations_.hub_time_[kDefaultProfile].size()}; ++h) {
      auto in = std::string{};
      for (auto const m : tt.locations_.hub_in_[kDefaultProfile][h]) {
        in += nm(m) + " ";
      }
      auto out = std::string{};
      for (auto const m : tt.locations_.hub_out_[kDefaultProfile][h]) {
        out += nm(m) + " ";
      }
      fmt::print("hub {} d={} in=[ {}] out=[ {}]\n", to_idx(h),
                 tt.locations_.hub_time_[kDefaultProfile][h].count(), in, out);
    }
    return 0;
  }
  if (auto const* id = std::getenv("NIGIRI_FP_DUMP");
      id != nullptr && std::string_view{id}.starts_with("IDX:")) {
    auto const l = location_idx_t{static_cast<std::uint32_t>(
        std::atoi(std::string{std::string_view{id}.substr(4)}.c_str()))};
    fmt::print("loc {} parent={} xfer={}\n", to_idx(l),
               tt.locations_.ids_[tt.locations_.parents_[l]].view(),
               tt.locations_.transfer_time_[l].count());
    for (auto const& fp : tt.locations_.footpaths_out_[kDefaultProfile][l]) {
      fmt::print("  fp -> {} ({} min)\n", to_idx(fp.target()),
                 fp.duration().count());
    }
    for (auto h = hub_idx_t{0U};
         h != hub_idx_t{tt.locations_.hub_time_[kDefaultProfile].size()}; ++h) {
      for (auto const u : tt.locations_.hub_in_[kDefaultProfile][h]) {
        if (u != l) {
          continue;
        }
        for (auto const v : tt.locations_.hub_out_[kDefaultProfile][h]) {
          fmt::print("  hub{} -> {} ({} min)\n", to_idx(h), to_idx(v),
                     tt.locations_.hub_time_[kDefaultProfile][h].count());
        }
      }
    }
    // incoming: sources that reach l, by hub and by stored footpath
    for (auto h = hub_idx_t{0U};
         h != hub_idx_t{tt.locations_.hub_time_[kDefaultProfile].size()}; ++h) {
      for (auto const v : tt.locations_.hub_out_[kDefaultProfile][h]) {
        if (v != l) {
          continue;
        }
        for (auto const u : tt.locations_.hub_in_[kDefaultProfile][h]) {
          fmt::print("  in-hub{} <- {} ({} min)\n", to_idx(h), to_idx(u),
                     tt.locations_.hub_time_[kDefaultProfile][h].count());
        }
      }
    }
    for (auto const& fp : tt.locations_.footpaths_in_[kDefaultProfile][l]) {
      fmt::print("  in-fp <- {} ({} min)\n", to_idx(fp.target()),
                 fp.duration().count());
    }
    return 0;
  }
  if (auto const* id = std::getenv("NIGIRI_FP_DUMP"); id != nullptr &&
      std::string_view{id} == "RELATION") {
    // the transfers the routing can actually make: stored footpaths plus the
    // pairs the hubs derive, minimum per pair. Two timetables that agree here
    // must answer every query the same way.
    auto rel = std::vector<hash_map<std::uint32_t, int>>(
        static_cast<std::size_t>(cista::to_idx(tt.n_locations())));
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      for (auto const& fp : tt.locations_.footpaths_out_[kDefaultProfile][l]) {
        auto const [it, ins] =
            rel[to_idx(l)].emplace(to_idx(fp.target()), fp.duration().count());
        if (!ins) {
          it->second = std::min(it->second,
                                static_cast<int>(fp.duration().count()));
        }
      }
    }
    for (auto h = hub_idx_t{0U};
         h != hub_idx_t{tt.locations_.hub_time_[kDefaultProfile].size()}; ++h) {
      auto const w = tt.locations_.hub_time_[kDefaultProfile][h].count();
      for (auto const u : tt.locations_.hub_in_[kDefaultProfile][h]) {
        for (auto const v : tt.locations_.hub_out_[kDefaultProfile][h]) {
          if (u == v) {
            continue;
          }
          auto const [it, ins] = rel[to_idx(u)].emplace(to_idx(v), w);
          if (!ins) {
            it->second = std::min(it->second, static_cast<int>(w));
          }
        }
      }
    }
    auto total = std::size_t{0U};
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      auto v = std::vector<std::pair<std::uint32_t, int>>{begin(rel[to_idx(l)]),
                                                          end(rel[to_idx(l)])};
      utl::sort(v);
      total += v.size();
      auto h = cista::BASE_HASH;
      for (auto const& [t, d] : v) {
        h = cista::hash_combine(cista::hash_combine(h, t),
                                static_cast<std::uint64_t>(d));
      }
      fmt::print("{} {} {}\n", to_idx(l), v.size(), h);
    }
    fmt::print("TOTAL {}\n", total);
    return 0;
  }
  if (auto const* id = std::getenv("NIGIRI_FP_DUMP"); id != nullptr &&
      std::string_view{id} == "CHECK") {
    auto self = 0U, dup = 0U, total = 0U, worse_dup = 0U;
    auto seen = hash_map<std::uint32_t, int>{};
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      seen.clear();
      for (auto const& fp : tt.locations_.footpaths_out_[kDefaultProfile][l]) {
        ++total;
        if (fp.target() == l) {
          ++self;
        }
        auto const [it, ins] =
            seen.emplace(to_idx(fp.target()), fp.duration().count());
        if (!ins) {
          ++dup;
          if (it->second != fp.duration().count()) {
            ++worse_dup;
          }
        }
      }
    }
    fmt::print("footpaths={} self_loops={} duplicate_targets={} "
               "duplicates_with_different_duration={}\n",
               total, self, dup, worse_dup);
    return 0;
  }
  if (auto const* id = std::getenv("NIGIRI_FP_DUMP"); id != nullptr &&
      std::string_view{id} == "VIRTS") {
    auto n_virt = 0U, no_routes = 0U, empty_id = 0U, no_fp = 0U, dead = 0U;
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.types_[l] != location_type::kVirt) {
        continue;
      }
      ++n_virt;
      auto const r = tt.location_routes_[l].size();
      auto const f = tt.locations_.footpaths_out_[kDefaultProfile][l].size() +
                     tt.locations_.footpaths_in_[kDefaultProfile][l].size();
      if (r == 0U) { ++no_routes; }
      if (f == 0U) { ++no_fp; }
      if (r == 0U && f == 0U) { ++dead; }
      if (tt.locations_.ids_[l].view().empty()) { ++empty_id; }
    }
    fmt::print("kVirt={} no_routes={} no_footpaths={} dead(both)={} empty_id={}\n",
               n_virt, no_routes, no_fp, dead, empty_id);
    return 0;
  }
  if (auto const* id = std::getenv("NIGIRI_FP_DUMP"); id != nullptr &&
      std::string_view{id} == "SYM") {
    for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
      if (tt.locations_.footpaths_out_[p].size() == 0U) {
        continue;
      }
      auto out = std::vector<std::array<std::uint64_t, 3>>{};
      auto in = std::vector<std::array<std::uint64_t, 3>>{};
      for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
        for (auto const& fp : tt.locations_.footpaths_out_[p][l]) {
          out.push_back({to_idx(l), to_idx(fp.target()),
                         static_cast<std::uint64_t>(fp.duration().count())});
        }
        for (auto const& fp : tt.locations_.footpaths_in_[p][l]) {
          in.push_back({to_idx(fp.target()), to_idx(l),
                        static_cast<std::uint64_t>(fp.duration().count())});
        }
      }
      std::sort(begin(out), end(out));
      std::sort(begin(in), end(in));
      fmt::print("prf={} out_edges={} in_edges={} identical={}\n", p,
                 out.size(), in.size(), out == in);
      if (out != in) {
        auto n = 0U;
        for (auto i = std::size_t{0U}; i < std::min(out.size(), in.size()); ++i) {
          if (out[i] != in[i]) {
            fmt::print("  first diff at {}: out=({} -> {}, {}) in=({} -> {}, {})\n",
                       i, tt.locations_.ids_[location_idx_t{out[i][0]}].view(),
                       tt.locations_.ids_[location_idx_t{out[i][1]}].view(), out[i][2],
                       tt.locations_.ids_[location_idx_t{in[i][0]}].view(),
                       tt.locations_.ids_[location_idx_t{in[i][1]}].view(), in[i][2]);
            if (++n == 3U) { break; }
          }
        }
      }
    }
    return 0;
  }
  if (auto const* id = std::getenv("NIGIRI_FP_DUMP"); id != nullptr &&
      std::string_view{id} == "EMPTY") {
    auto n = 0U;
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.ids_[l].view().empty()) {
        ++n;
        if (n <= 8U) {
          auto const p = tt.locations_.parents_[l];
          fmt::print("empty id: idx={} type={} parent={} n_routes={}\n", l,
                     static_cast<int>(tt.locations_.types_[l]),
                     p == location_idx_t::invalid()
                         ? std::string{"-"}
                         : std::string{tt.locations_.ids_[p].view()},
                     tt.location_routes_[l].size());
        }
      }
    }
    fmt::print("total empty ids: {} / {}\n", n, tt.n_locations());
    return 0;
  }
  // VIRTSTATS - how many virtual locations carry an own transfer time that
  // differs from their stop's. Those are the ones the merge cannot fold away
  // (an unstated pair between them would change price), so they set the floor
  // on how many locations the search has to walk.
  if (auto const* spec = std::getenv("NIGIRI_FP_DUMP");
      spec != nullptr && std::string_view{spec} == "VIRTSTATS") {
    auto n_virt = 0U, differs = 0U, zero = 0U, no_routes = 0U;
    auto by_parent = hash_map<location_idx_t, unsigned>{};
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.types_[l] != location_type::kVirt) {
        continue;
      }
      ++n_virt;
      ++by_parent[tt.locations_.parents_[l]];
      auto const own = tt.locations_.transfer_time_[l].count();
      auto const dflt =
          tt.locations_.transfer_time_[tt.locations_.parents_[l]].count();
      if (own != dflt) {
        ++differs;
      }
      if (own == 0) {
        ++zero;
      }
      if (tt.location_routes_[l].empty()) {
        ++no_routes;
      }
    }
    fmt::print(
        "virts={} own!=stop_default={} own==0={} without_routes={} "
        "stops_with_virts={}\n",
        n_virt, differs, zero, no_routes, by_parent.size());
    // footpath volume per profile, and how much of it survives projecting
    // virtual locations onto their stop (duplicate targets collapse)
    for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
      auto const& fps = tt.locations_.footpaths_out_[p];
      if (fps.size() == 0U) {
        continue;
      }
      auto total = std::size_t{0U}, projected = std::size_t{0U};
      auto longest = std::size_t{0U};
      auto seen = hash_map<std::uint32_t, int>{};
      for (auto l = location_idx_t{0U}; l != location_idx_t{fps.size()}; ++l) {
        total += fps[l].size();
        longest = std::max(longest, static_cast<std::size_t>(fps[l].size()));
        seen.clear();
        for (auto const& fp : fps[l]) {
          auto const t = tt.locations_.types_[fp.target()] == location_type::kVirt
                             ? tt.locations_.parents_[fp.target()]
                             : fp.target();
          auto const [it, ins] = seen.emplace(to_idx(t), fp.duration().count());
          if (!ins) {
            it->second = std::min(it->second,
                                  static_cast<int>(fp.duration().count()));
          }
        }
        projected += seen.size();
      }
      fmt::print("prf={} footpaths={} after_projection={} longest_list={}\n", p,
                 total, projected, longest);
    }
    return 0;
  }
  // ROUTEPROJ - routes are trips grouped by identical stop sequences, so a
  // stop split into virts keeps trips apart that would otherwise share a
  // route. This counts how many routes would collapse if every virt were
  // projected back onto its stop - the route-side cost of the split.
  if (auto const* spec = std::getenv("NIGIRI_FP_DUMP");
      spec != nullptr && std::string_view{spec} == "ROUTEPROJ") {
    auto seqs = hash_set<cista::hash_t>{};
    auto with_virt = 0U;
    auto virt_stops = std::size_t{0U}, total_stops = std::size_t{0U};
    for (auto r = route_idx_t{0U}; r != route_idx_t{tt.n_routes()}; ++r) {
      auto h = cista::BASE_HASH;
      auto has_virt = false;
      for (auto const s : tt.route_location_seq_[r]) {
        auto const stp = stop{s};
        auto const l = stp.location_idx();
        ++total_stops;
        auto const proj =
            tt.locations_.types_[l] == location_type::kVirt
                ? (++virt_stops, has_virt = true, tt.locations_.parents_[l])
                : l;
        h = cista::hash_combine(h, to_idx(proj));
        h = cista::hash_combine(h, stp.in_allowed() ? 1U : 0U);
        h = cista::hash_combine(h, stp.out_allowed() ? 1U : 0U);
      }
      seqs.insert(h);
      if (has_virt) {
        ++with_virt;
      }
    }
    fmt::print(
        "routes={} distinct_projected_seqs={} routes_touching_a_virt={} "
        "route_stops={} of_which_virt={}\n",
        tt.n_routes(), seqs.size(), with_virt, total_stops, virt_stops);
    return 0;
  }
  // PRUNE - how much of the stored foot layer the hubs already derive at the
  // same weight. Those edges are walked twice by the search: once as a
  // footpath, once through the hub that stands for the same pair.
  if (auto const* spec = std::getenv("NIGIRI_FP_DUMP");
      spec != nullptr && std::string_view{spec} == "PRUNE") {
    constexpr auto const p = kDefaultProfile;
    // (from, to) -> best weight a hub derives
    auto derived = hash_map<std::uint64_t, int>{};
    for (auto h = hub_idx_t{0U};
         h != hub_idx_t{tt.locations_.hub_time_[p].size()}; ++h) {
      auto const w = static_cast<int>(tt.locations_.hub_time_[p][h].count());
      for (auto const u : tt.locations_.hub_in_[p][h]) {
        for (auto const v : tt.locations_.hub_out_[p][h]) {
          if (u == v) {
            continue;
          }
          auto const k = (static_cast<std::uint64_t>(to_idx(u)) << 32) | to_idx(v);
          auto const [it, ins] = derived.emplace(k, w);
          if (!ins) {
            it->second = std::min(it->second, w);
          }
        }
      }
    }
    auto stored = std::size_t{0U}, covered_eq = std::size_t{0U},
         covered_worse = std::size_t{0U}, uncovered = std::size_t{0U};
    for (auto l = location_idx_t{0U};
         l != location_idx_t{tt.locations_.footpaths_out_[p].size()}; ++l) {
      for (auto const& fp : tt.locations_.footpaths_out_[p][l]) {
        ++stored;
        auto const k =
            (static_cast<std::uint64_t>(to_idx(l)) << 32) | to_idx(fp.target());
        auto const it = derived.find(k);
        if (it == end(derived)) {
          ++uncovered;
        } else if (it->second == static_cast<int>(fp.duration().count())) {
          ++covered_eq;
        } else if (it->second < static_cast<int>(fp.duration().count())) {
          ++covered_worse;  // hub is faster: the footpath never wins
        } else {
          ++uncovered;  // footpath is faster: it has to stay
        }
      }
    }
    fmt::print(
        "stored={} hub_derived_pairs={} covered_same_weight={} "
        "covered_hub_faster={} must_stay={}\n",
        stored, derived.size(), covered_eq, covered_worse, uncovered);
    // A real-time track change moves a trip onto a base location, which the
    // schedule timetable may never use because the rules split every trip
    // onto virts. Edges touching a base therefore have to survive whatever
    // the hubs derive - so count how much of the prunable set that is.
    auto virt_virt = std::size_t{0U}, base_incident = std::size_t{0U};
    auto const is_virt = [&](location_idx_t const l) {
      return tt.locations_.types_[l] == location_type::kVirt;
    };
    for (auto l = location_idx_t{0U};
         l != location_idx_t{tt.locations_.footpaths_out_[p].size()}; ++l) {
      for (auto const& fp : tt.locations_.footpaths_out_[p][l]) {
        auto const it = derived.find(
            (static_cast<std::uint64_t>(to_idx(l)) << 32) | to_idx(fp.target()));
        if (it == end(derived) ||
            it->second > static_cast<int>(fp.duration().count())) {
          continue;  // not prunable anyway
        }
        if (is_virt(l) && is_virt(fp.target())) {
          ++virt_virt;
        } else {
          ++base_incident;
        }
      }
    }
    fmt::print("  prunable virt->virt={} base_incident={}\n", virt_virt,
               base_incident);
    // The other direction: is every pair a hub derives also stored as a
    // footpath, at least as fast? Only then can the hubs be switched off
    // without losing or slowing a transfer (the materialized reference is
    // supposed to guarantee exactly this).
    auto stored_w = hash_map<std::uint64_t, int>{};
    for (auto l = location_idx_t{0U};
         l != location_idx_t{tt.locations_.footpaths_out_[p].size()}; ++l) {
      for (auto const& fp : tt.locations_.footpaths_out_[p][l]) {
        auto const k =
            (static_cast<std::uint64_t>(to_idx(l)) << 32) | to_idx(fp.target());
        auto const d = static_cast<int>(fp.duration().count());
        auto const [it, ins] = stored_w.emplace(k, d);
        if (!ins) {
          it->second = std::min(it->second, d);
        }
      }
    }
    auto missing = std::size_t{0U}, slower = std::size_t{0U},
         ok = std::size_t{0U};
    for (auto const& [k, w] : derived) {
      auto const it = stored_w.find(k);
      if (it == end(stored_w)) {
        ++missing;
      } else if (it->second > w) {
        ++slower;
      } else {
        ++ok;
      }
    }
    fmt::print(
        "  hubs droppable? derived={} stored_at_least_as_fast={} "
        "stored_slower={} not_stored={}\n",
        derived.size(), ok, slower, missing);
    return 0;
  }
  // REFINE - what partition refinement (HRD's approach) would yield on the
  // structure the GTFS loader actually produced. The merge there compares
  // each virt pairwise against a representative, which is one refinement
  // step; refining to a fixpoint can merge classes that only become
  // equivalent after their neighbours merged. This measures the difference
  // without changing the loader.
  if (auto const* spec = std::getenv("NIGIRI_FP_DUMP");
      spec != nullptr && std::string_view{spec} == "REFINE") {
    constexpr auto const p = kDefaultProfile;
    constexpr auto const kInf = std::numeric_limits<int>::max();
    // hub-derived pairs, min per (from,to)
    auto derived = hash_map<std::uint64_t, int>{};
    for (auto h = hub_idx_t{0U}; h != hub_idx_t{tt.locations_.hub_time_[p].size()}; ++h) {
      auto const w = static_cast<int>(tt.locations_.hub_time_[p][h].count());
      for (auto const u : tt.locations_.hub_in_[p][h]) {
        for (auto const v : tt.locations_.hub_out_[p][h]) {
          if (u == v) { continue; }
          auto const k = (static_cast<std::uint64_t>(to_idx(u)) << 32) | to_idx(v);
          auto const [it, ins] = derived.emplace(k, w);
          if (!ins) { it->second = std::min(it->second, w); }
        }
      }
    }
    auto members_of = hash_map<location_idx_t, std::vector<location_idx_t>>{};
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.types_[l] == location_type::kVirt) {
        members_of[tt.locations_.parents_[l]].push_back(l);
      }
    }
    auto total_members = std::size_t{0U}, total_classes = std::size_t{0U};
    auto stops_shrunk = 0U;
    for (auto& [base, virts] : members_of) {
      auto m = std::vector<location_idx_t>{base};
      m.insert(end(m), begin(virts), end(virts));
      auto const n = m.size();
      auto const pos = [&](location_idx_t const x) {
        return static_cast<std::size_t>(utl::find(m, x) - begin(m));
      };
      auto mat = std::vector<int>(n * n, kInf);
      for (auto i = std::size_t{0U}; i != n; ++i) {
        mat[i * n + i] = tt.locations_.transfer_time_[m[i]].count();
        for (auto const& fp : tt.locations_.footpaths_out_[p][m[i]]) {
          if (utl::find(m, fp.target()) == end(m)) { continue; }
          auto& c = mat[i * n + pos(fp.target())];
          c = std::min(c, static_cast<int>(fp.duration().count()));
        }
        for (auto j = std::size_t{0U}; j != n; ++j) {
          if (i == j) { continue; }
          auto const it = derived.find(
              (static_cast<std::uint64_t>(to_idx(m[i])) << 32) | to_idx(m[j]));
          if (it != end(derived)) {
            mat[i * n + j] = std::min(mat[i * n + j], it->second);
          }
        }
      }
      // refine to fixpoint on (own time, row, col) relative to classes
      auto cls = std::vector<unsigned>(n, 0U);
      for (auto changed = true; changed;) {
        changed = false;
        auto sig = std::map<std::vector<std::int64_t>, unsigned>{};
        auto next = std::vector<unsigned>(n, 0U);
        for (auto i = std::size_t{0U}; i != n; ++i) {
          auto key = std::vector<std::int64_t>{cls[i], mat[i * n + i]};
          for (auto j = std::size_t{0U}; j != n; ++j) {
            if (j == i) { continue; }
            key.push_back((static_cast<std::int64_t>(cls[j]) << 34) |
                          (static_cast<std::int64_t>(mat[i * n + j] == kInf ? 1023 : mat[i * n + j]) << 12) |
                          (mat[j * n + i] == kInf ? 1023 : mat[j * n + i]));
          }
          next[i] = sig.emplace(key, static_cast<unsigned>(sig.size())).first->second;
        }
        changed = next != cls;
        cls = std::move(next);
      }
      // exactness: same class => pair must cost the class's own time
      auto bad = true;
      while (bad) {
        bad = false;
        auto n_cls = 1U + *std::max_element(begin(cls), end(cls));
        for (auto i = std::size_t{0U}; i != n && !bad; ++i) {
          for (auto j = std::size_t{0U}; j != n; ++j) {
            if (i != j && cls[i] == cls[j] && mat[i * n + j] != mat[i * n + i]) {
              cls[j] = n_cls;
              bad = true;
              break;
            }
          }
        }
      }
      auto uniq = std::vector<unsigned>{begin(cls), end(cls)};
      std::sort(begin(uniq), end(uniq));
      uniq.erase(std::unique(begin(uniq), end(uniq)), end(uniq));
      total_members += n;
      total_classes += uniq.size();
      if (uniq.size() < n) { ++stops_shrunk; }
    }
    fmt::print(
        "stops_with_virts={} members(now)={} classes(refined)={} "
        "stops_that_would_shrink={}\n",
        members_of.size(), total_members, total_classes, stops_shrunk);
    return 0;
  }
  // MERGEBASE - virts that are indistinguishable from their own stop. The
  // merge in apply_rules only ever compares virt against virt, so these are
  // never folded back into the base even though nothing can tell them apart.
  if (auto const* spec = std::getenv("NIGIRI_FP_DUMP");
      spec != nullptr && std::string_view{spec} == "MERGEBASE") {
    constexpr auto const p = kDefaultProfile;
    auto hub_mem = hash_map<location_idx_t, std::vector<std::uint32_t>>{};
    for (auto h = 0U; h != tt.locations_.hub_in_[p].size(); ++h) {
      for (auto const m : tt.locations_.hub_in_[p][hub_idx_t{h}]) {
        hub_mem[m].push_back(h * 2U);
      }
      for (auto const m : tt.locations_.hub_out_[p][hub_idx_t{h}]) {
        hub_mem[m].push_back(h * 2U + 1U);
      }
    }
    auto const hubs_of = [&](location_idx_t const l) {
      auto v = std::vector<std::uint32_t>{};
      if (auto const it = hub_mem.find(l); it != end(hub_mem)) { v = it->second; }
      utl::sort(v);
      return v;
    };
    auto const edges = [&](auto const& vv, location_idx_t const l,
                           location_idx_t const partner) {
      auto v = std::vector<std::pair<std::uint32_t, int>>{};
      if (to_idx(l) < vv.size()) {
        for (auto const& fp : vv[l]) {
          if (fp.target() != partner) {
            v.emplace_back(to_idx(fp.target()), fp.duration().count());
          }
        }
      }
      utl::sort(v);
      return v;
    };
    auto n_virt = 0U, same_time = 0U, mergeable = 0U;
    auto by_parent = hash_map<location_idx_t, unsigned>{};
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.types_[l] != location_type::kVirt) { continue; }
      ++n_virt;
      auto const par = tt.locations_.parents_[l];
      if (tt.locations_.transfer_time_[l] != tt.locations_.transfer_time_[par]) {
        continue;
      }
      ++same_time;
      if (hubs_of(l) == hubs_of(par) &&
          edges(tt.locations_.footpaths_out_[p], l, par) ==
              edges(tt.locations_.footpaths_out_[p], par, l) &&
          edges(tt.locations_.footpaths_in_[p], l, par) ==
              edges(tt.locations_.footpaths_in_[p], par, l)) {
        ++mergeable;
        ++by_parent[par];
      }
    }
    fmt::print(
        "virts={} same_own_time_as_stop={} indistinguishable_from_stop={} "
        "stops_affected={}\n",
        n_virt, same_time, mergeable, by_parent.size());
    return 0;
  }
  // HUBDEG - how uneven the hub work is: the scatter costs one relaxation per
  // out-list entry, so the tail of this distribution is what a thread-per-hub
  // kernel makes a single lane walk alone.
  if (auto const* spec = std::getenv("NIGIRI_FP_DUMP");
      spec != nullptr && std::string_view{spec} == "HUBDEG") {
    for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
      auto const n = tt.locations_.hub_time_[p].size();
      if (n == 0U) {
        continue;
      }
      auto degs = std::vector<std::size_t>{};
      auto total = std::size_t{0U};
      for (auto h = hub_idx_t{0U}; h != hub_idx_t{n}; ++h) {
        auto const d = tt.locations_.hub_out_[p][h].size();
        degs.push_back(d);
        total += d;
      }
      utl::sort(degs);
      auto const q = [&](double const f) {
        return degs[std::min(degs.size() - 1U,
                             static_cast<std::size_t>(f * degs.size()))];
      };
      // work above each threshold: what a cooperative path would take over
      auto over = [&](std::size_t const t) {
        auto s = std::size_t{0U};
        for (auto const d : degs) {
          if (d > t) {
            s += d;
          }
        }
        return s;
      };
      auto in_total = std::size_t{0U};
      auto pairs = std::size_t{0U};
      for (auto h = hub_idx_t{0U}; h != hub_idx_t{n}; ++h) {
        auto const i = tt.locations_.hub_in_[p][h].size();
        in_total += i;
        pairs += i * tt.locations_.hub_out_[p][h].size();
      }
      auto fps = std::size_t{0U};
      for (auto l = location_idx_t{0U};
           l != location_idx_t{tt.locations_.footpaths_out_[p].size()}; ++l) {
        fps += tt.locations_.footpaths_out_[p][l].size();
      }
      fmt::print(
          "prf={} hubs={} in_total={} out_total={} pairs_derived={} "
          "stored_footpaths={} min={} q50={} q90={} q99={} max={}\n",
          p, n, in_total, total, pairs, fps, degs.front(), q(0.5), q(0.9),
          q(0.99), degs.back());
      fmt::print(
          "  work in lists >8: {} ({:.1f}%), >32: {} ({:.1f}%)\n", over(8),
          100.0 * static_cast<double>(over(8)) /
              static_cast<double>(std::max(total, std::size_t{1})),
          over(32),
          100.0 * static_cast<double>(over(32)) /
              static_cast<double>(std::max(total, std::size_t{1})));
    }
    return 0;
  }
  // STOP:<id> - the stop, every virtual location split off it, and the edges
  // each of them carries (footpaths plus the pairs their hubs derive). Two
  // timetables that encode the same rules must agree here per (from, to) pair,
  // whatever their virts are numbered.
  if (auto const* spec = std::getenv("NIGIRI_FP_DUMP");
      spec != nullptr && std::string_view{spec}.starts_with("STOP:")) {
    auto const want = std::string_view{spec}.substr(5);
    auto const nm = [&](location_idx_t const l) {
      auto const v = tt.locations_.ids_[l].view();
      return v.empty() ? fmt::format("virt#{}(of {})", to_idx(l),
                                     tt.locations_.ids_
                                         [tt.locations_.parents_[l]]
                                             .view())
                       : std::string{v};
    };
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.ids_[l].view() != want) {
        continue;
      }
      auto members = std::vector<location_idx_t>{l};
      for (auto const c : tt.locations_.children_[l]) {
        if (tt.locations_.types_[c] == location_type::kVirt) {
          members.push_back(c);
        }
      }
      fmt::print("stop {} idx={} members={} xfer={}\n", want, to_idx(l),
                 members.size(), tt.locations_.transfer_time_[l].count());
      // effective pairs among the members, minimum per (from, to)
      auto pairs = std::map<std::pair<std::uint32_t, std::uint32_t>, int>{};
      auto const add = [&](location_idx_t const a, location_idx_t const b,
                           int const d) {
        auto const k = std::pair{to_idx(a), to_idx(b)};
        auto const it = pairs.find(k);
        if (it == end(pairs)) {
          pairs.emplace(k, d);
        } else {
          it->second = std::min(it->second, d);
        }
      };
      auto const is_member = [&](location_idx_t const x) {
        return utl::find(members, x) != end(members);
      };
      for (auto const m : members) {
        add(m, m, tt.locations_.transfer_time_[m].count());
        for (auto const& fp : tt.locations_.footpaths_out_[kDefaultProfile][m]) {
          if (is_member(fp.target())) {
            add(m, fp.target(), fp.duration().count());
          }
        }
      }
      for (auto h = hub_idx_t{0U};
           h != hub_idx_t{tt.locations_.hub_time_[kDefaultProfile].size()};
           ++h) {
        auto const w = tt.locations_.hub_time_[kDefaultProfile][h].count();
        for (auto const u : tt.locations_.hub_in_[kDefaultProfile][h]) {
          if (!is_member(u)) {
            continue;
          }
          for (auto const v : tt.locations_.hub_out_[kDefaultProfile][h]) {
            if (is_member(v) && u != v) {
              add(u, v, static_cast<int>(w));
            }
          }
        }
      }
      auto hist = std::map<int, unsigned>{};
      for (auto const& [k, d] : pairs) {
        ++hist[d];
      }
      fmt::print("  internal pairs={} durations:", pairs.size());
      for (auto const& [d, c] : hist) {
        fmt::print(" {}min x{}", d, c);
      }
      fmt::print("\n");

      // events in a window: which member each trip uses, so a connection can
      // be priced against the pair table above
      if (auto const* w = std::getenv("NIGIRI_STOP_WINDOW"); w != nullptr) {
        auto const comma = std::string_view{w}.find(',');
        auto const lo = std::stoi(std::string{std::string_view{w}.substr(0, comma)});
        auto const hi = std::stoi(std::string{std::string_view{w}.substr(comma + 1)});
        for (auto const m : members) {
          for (auto const r : tt.location_routes_[m]) {
            auto const seq = tt.route_location_seq_[r];
            for (auto i = 0U; i != seq.size(); ++i) {
              if (stop{seq[i]}.location_idx() != m) {
                continue;
              }
              for (auto t = tt.route_transport_ranges_[r].from_;
                   t != tt.route_transport_ranges_[r].to_; ++t) {
                auto const dep =
                    i + 1U < seq.size()
                        ? tt.event_mam(r, t, static_cast<stop_idx_t>(i),
                                       event_type::kDep)
                              .count() % 1440
                        : -1;
                auto const arr =
                    i > 0U ? tt.event_mam(r, t, static_cast<stop_idx_t>(i),
                                          event_type::kArr)
                                 .count() % 1440
                           : -1;
                if ((arr >= lo && arr <= hi) || (dep >= lo && dep <= hi)) {
                  fmt::print(
                      "  event member={} xfer={} trip={} arr={:02d}:{:02d} "
                      "dep={:02d}:{:02d}\n",
                      to_idx(m), tt.locations_.transfer_time_[m].count(),
                      tt.transport_name(t), arr / 60, arr % 60, dep / 60,
                      dep % 60);
                }
              }
            }
          }
        }
      }
      // which trips share a member: a transfer between two trips costs what
      // the pair of members they use costs, so trips on the same member
      // transfer at that member's own time
      for (auto const m : members) {
        auto names = std::vector<std::string>{};
        for (auto const r : tt.location_routes_[m]) {
          auto const range = tt.route_transport_ranges_[r];
          if (range.size() == 0U) {
            continue;
          }
          names.emplace_back(tt.transport_name(range.from_));
        }
        utl::sort(names);
        names.erase(std::unique(begin(names), end(names)), end(names));
        if (names.empty()) {
          continue;
        }
        fmt::print("  member {:<7} xfer={} routes={} trips=[{}]\n",
                   to_idx(m) == to_idx(l) ? std::string{"stop"}
                                          : fmt::format("virt{}", to_idx(m)),
                   tt.locations_.transfer_time_[m].count(),
                   tt.location_routes_[m].size(), fmt::join(names, " "));
      }
    }
    return 0;
  }
  if (auto const* id = std::getenv("NIGIRI_FP_DUMP"); id != nullptr) {
    for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
      fmt::print("prf={} fp_out={} fp_in={}\n", p,
                 tt.locations_.footpaths_out_[p].size(),
                 tt.locations_.footpaths_in_[p].size());
    }
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.ids_[l].view() != std::string_view{id}) {
        continue;
      }
      fmt::print("{} idx={} type={} transfer_time={} base={}\n",
                 tt.locations_.ids_[l].view(), l,
                 static_cast<int>(tt.locations_.types_[l]),
                 tt.locations_.transfer_time_[l].count(),
                 l < location_idx_t{tt.locations_.base_transfer_time_.size()}
                     ? tt.locations_.base_transfer_time_[l].count()
                     : -1);
      auto n_virt = 0U;
      for (auto const c : tt.locations_.children_[l]) {
        if (tt.locations_.types_[c] == location_type::kVirt) {
          ++n_virt;
        }
      }
      auto n_by_parent = 0U;
      for (auto x = location_idx_t{0U}; x != tt.n_locations(); ++x) {
        if (tt.locations_.types_[x] == location_type::kVirt &&
            tt.locations_.parents_[x] == l) {
          ++n_by_parent;
        }
      }
      fmt::print("  children={} virt_children={} virts_by_parents={} n_routes={}\n",
                 tt.locations_.children_[l].size(), n_virt, n_by_parent,
                 tt.location_routes_[l].size());
      for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
        if (tt.locations_.footpaths_out_[p].size() <= to_idx(l)) {
          continue;
        }
        for (auto const& fp : tt.locations_.footpaths_out_[p][l]) {
          fmt::print("  prf={} -> {} ({} min)\n", p,
                     tt.locations_.ids_[fp.target()].view(),
                     fp.duration().count());
        }
        for (auto const& fp : tt.locations_.footpaths_in_[p][l]) {
          fmt::print("  prf={} <- {} ({} min)\n", p,
                     tt.locations_.ids_[fp.target()].view(),
                     fp.duration().count());
        }
      }
    }
    return 0;
  }

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
  for (auto const& d : dirs) {
    if (d != "fwd" && d != "bwd") {
      std::cerr << "invalid dir \"" << d << "\", expected fwd | bwd\n";
      return 1;
    }
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
    if (a != "range" && a != "pong") {
      std::cerr << "invalid algo \"" << a << "\", expected raptor | pong\n";
      return 1;
    }
  }

  // apply one end of a <start>2<dest> mode token to the generator settings
  // (the first/last-mile transport modes come from --intermodal_start/_dest)
  auto const apply_mode = [](char const m, location_match_mode& match) {
    switch (m) {
      case 's': match = location_match_mode::kEquivalent; return true;
      case 'c': match = location_match_mode::kIntermodal; return true;
      default: return false;
    }
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
  auto total = std::uint64_t{0U};
  auto qa_cell = std::optional<result_set>{};
  auto qa_n_cpu_cells = 0U;

#if defined(NIGIRI_CUDA)
  auto gpu_tt = std::optional<routing::gpu::gpu_timetable>{};
  if (run_gpu) {
    gpu_tt.emplace(tt);
  }
#endif

  for (auto const& mode : modes) {
    auto rs = gs;
    if (mode.size() != 3U || mode[1] != '2' ||
        !apply_mode(mode[0], rs.start_match_mode_) ||
        !apply_mode(mode[2], rs.dest_match_mode_)) {
      std::cerr << "invalid mode \"" << mode
                << "\", expected s2s | s2c | c2s | c2c\n";
      return 1;
    }
    if (rs.start_match_mode_ == location_match_mode::kIntermodal) {
      rs.use_start_footpaths_ = false;  // first mile is in the start offsets
    }

    auto& fwd_qs = mode_queries[mode];
    if (fwd_qs.empty()) {
      generate_queries(fwd_qs, n_queries, tt, rs, seed);
      if (auto const* only = std::getenv("NIGIRI_ONLY_QUERY");
          only != nullptr) {
        auto const idx = static_cast<std::size_t>(std::atoll(only));
        if (idx < fwd_qs.size()) {
          auto const q = fwd_qs[idx];
          fwd_qs.clear();
          fwd_qs.push_back(q);
        }
      }
      auto const *ef = std::getenv("NIGIRI_QUERY_FROM"),
                 *et = std::getenv("NIGIRI_QUERY_TO"),
                 *eb = std::getenv("NIGIRI_QUERY_BEGIN"),
                 *ee = std::getenv("NIGIRI_QUERY_END");
      if (ef && et && eb && ee) {
        auto const resolve = [&](std::string_view const id) {
          for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
            if (tt.locations_.ids_[l].view() == id) {
              return l;
            }
          }
          fmt::println("cannot resolve location id \"{}\"", id);
          std::exit(1);
        };
        auto const from_l = resolve(ef);
        auto const to_l = resolve(et);
        auto const t = [](char const* e) {
          return unixtime_t{std::chrono::minutes{std::atoll(e)}};
        };
        auto sdq = nigiri::query_generation::start_dest_query{};
        sdq.start_ = from_l;
        sdq.dest_ = to_l;
        sdq.q_ = routing::query{
            .start_time_ = interval<unixtime_t>{t(eb), t(ee)},
            .start_match_mode_ = rs.start_match_mode_,
            .dest_match_mode_ = rs.dest_match_mode_,
            .use_start_footpaths_ = rs.use_start_footpaths_,
            .start_ = {routing::offset{from_l, duration_t{0U}, 0U}},
            .destination_ = {routing::offset{to_l, duration_t{0U}, 0U}},
            .min_connection_count_ = rs.min_connection_count_,
            .extend_interval_earlier_ = rs.extend_interval_earlier_,
            .extend_interval_later_ = rs.extend_interval_later_,
            .prf_idx_ = rs.prf_idx_,
            .allowed_claszes_ = rs.allowed_claszes_,
            .transfer_time_settings_ = rs.transfer_time_settings_};
        sdq.q_.max_transfers_ = rs.max_transfers_;
        fwd_qs.clear();
        fwd_qs.push_back(sdq);
      }
    }

    // (mode, dir) are the incomparable dimensions -- within one (mode, dir),
    // every (engine, algo) combination has to agree
    for (auto const& dir_str : dirs) {
      auto const dir =
          dir_str == "fwd" ? direction::kForward : direction::kBackward;

      auto qs = fwd_qs;
      for (auto& sdq : qs) {
        if (dir == direction::kBackward) {
          sdq.q_.flip_dir();
        }
        sdq.q_.extend_interval_earlier_ = dir == direction::kBackward;
        sdq.q_.extend_interval_later_ = dir == direction::kForward;
      }

      auto cells = std::vector<result_set>{};
      for (auto const& algo : algos) {
        auto const use_pong = algo == "pong";
        auto const label = mode + "-" + dir_str + "-" + algo;

        try {
          if (run_cpu) {
            cells.push_back(run_cell<cpu_ws>(
                qs, label + "-cpu", threads_v,
                [&](cpu_ws& w, routing::query q) {
                  auto const r =
                      use_pong
                          ? routing::pong_search(tt, nullptr, w.ss_, w.rs_,
                                                 std::move(q), dir)
                          : routing::raptor_search(tt, nullptr, w.ss_, w.rs_,
                                                   std::move(q), dir);
                  auto const stat = [&](char const* k) {
                    auto const it = r.algo_stats_.find(k);
                    return it == end(r.algo_stats_) ? std::uint64_t{0U}
                                                    : it->second;
                  };
                  return std::pair{*r.journeys_,
                                   std::array<std::uint64_t, 2>{
                                       stat("n_routes_visited"),
                                       stat("n_footpaths_visited")}};
                }));
            ++qa_n_cpu_cells;
            if (vm.count("qa_path")) {
              qa_cell = cells.back();
            }
          }

#if defined(NIGIRI_CUDA)
          if (run_gpu) {
            cells.push_back(run_cell<gpu_ws>(
                qs, label + "-gpu", gpu_states_v,
                [&](gpu_ws& w, routing::query q) {
                  auto const r =
                      use_pong
                          ? routing::pong_search(tt, nullptr, w.ss_, *w.rs_,
                                                 std::move(q), dir)
                          : routing::raptor_search(tt, nullptr, w.ss_, *w.rs_,
                                                   std::move(q), dir);
                  auto const stat = [&](char const* k) {
                    auto const it = r.algo_stats_.find(k);
                    return it == end(r.algo_stats_) ? std::uint64_t{0U}
                                                    : it->second;
                  };
                  return std::pair{*r.journeys_,
                                   std::array<std::uint64_t, 2>{
                                       stat("n_routes_visited"),
                                       stat("n_footpaths_visited")}};
                },
                *gpu_tt));
          }
#endif
        } catch (std::exception const& e) {
          // e.g. GPU state allocation OOM -- report + fail instead of dying
          std::cerr << "RUN " << label << " failed: " << e.what() << "\n";
          summary.push_back(
              fmt::format("{:<24} EXCEPTION: {}", label, e.what()));
          ++total;
        }
      }

      if (cells.size() == 1U) {
        summary.push_back(fmt::format("{:<24} n={:<6} benchmark only",
                                      cells.front().label_, qs.size()));
      }
      for (auto const& c : cells) {
        auto rv = std::uint64_t{0U};
        auto fv = std::uint64_t{0U};
        for (auto const& st : c.stats_) {
          rv += st[0];
          fv += st[1];
        }
        summary.push_back(fmt::format(
            "{:<24} n={:<6} routes_visited/q={:<9.0f} fps_visited/q={:<11.0f}",
            c.label_, c.stats_.size(),
            c.stats_.empty() ? 0.0
                             : static_cast<double>(rv) /
                                   static_cast<double>(c.stats_.size()),
            c.stats_.empty() ? 0.0
                             : static_cast<double>(fv) /
                                   static_cast<double>(c.stats_.size())));
      }
      for (auto const& c : cells) {
        auto n_journeys = std::size_t{0U};
        auto n_empty = std::size_t{0U};
        for (auto const& r : c.res_) {
          n_journeys += r.size();
          n_empty += r.size() == 0U ? 1U : 0U;
        }
        summary.push_back(fmt::format(
            "{:<24} n={:<6} journeys={:<7} avg={:<5.2f} no-journey-queries={}",
            c.label_, c.res_.size(), n_journeys,
            c.res_.empty()
                ? 0.0
                : static_cast<double>(n_journeys) /
                      static_cast<double>(c.res_.size()),
            n_empty));
      }
      for (auto a = std::size_t{0U}; a < cells.size(); ++a) {
        for (auto b = a + 1U; b < cells.size(); ++b) {
          auto const mismatches = compare_results(
              tt, cells[a].label_, cells[a].res_, cells[b].label_,
              cells[b].res_, qs, dir, gs.min_connection_count_);
          summary.push_back(
              fmt::format("{:<24} vs {:<24} n={:<6} mismatches={:<4} {}",
                          cells[a].label_, cells[b].label_, qs.size(),
                          mismatches, mismatches == 0U ? "PASS" : "FAIL"));
          total += mismatches;
        }
      }
    }
  }

  std::cout << "\n=== SUMMARY ===\n";
  for (auto const& s : summary) {
    std::cout << s << "\n";
  }
  print_memory_usage();

  if (vm.count("qa_path")) {
    if (qa_n_cpu_cells != 1U || !qa_cell.has_value()) {
      std::cerr << "--qa_path requires exactly one cpu (mode, dir, algo) cell "
                   "(single-element --algo/--modes/--dirs)\n";
      return 1;
    }
    auto bm_crit = nigiri::qa::benchmark_criteria{};
    for (auto i = std::size_t{0U}; i != qa_cell->res_.size(); ++i) {
      auto jc = vector<nigiri::qa::criteria_t>{};
      for (auto const& j : qa_cell->res_[i]) {
        jc.emplace_back(
            static_cast<double>(j.start_time_.time_since_epoch().count()),
            static_cast<double>(j.dest_time_.time_since_epoch().count()),
            static_cast<double>(j.transfers_));
      }
      utl::sort(jc);
      auto const latency =
          i < qa_cell->latencies_.size() ? qa_cell->latencies_[i] : 0.0;
      bm_crit.qc_.emplace_back(
          i,
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::duration<double, std::milli>{
                  std::max(latency, 0.0)}),
          jc);
    }
    bm_crit.write(qa_path);
  }

  return total == 0U ? 0 : 1;
}
