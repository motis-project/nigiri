#include <cstdio>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <span>
#include <optional>
#include <random>
#include <sstream>
#include <thread>

#include "boost/program_options.hpp"

#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
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
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
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

// Synthetic real-time timetable for `day`. The defaults mimic the DELFI
// GTFS-RT feed (snapshot of 2026-10-01 08:48 UTC against the 2026-09-28
// schedule, 165,604 trip updates; see rt_synth_params): a share of the
// transports active on that day gets an rt instance; `p_cancel` of those are
// cancelled, `punctual` reported on time at minute resolution (the real feed
// carries second-level delays, most of which round to zero), the rest shifted
// by a lognormal delay with the measured median and p90, rounded up to whole
// minutes and capped at `max_delay`. Deterministic in `seed`.
struct rt_synth_params {
  double share_{0.55};  // covered share of the transports running in the
                        // feed's 6 h window (165 k of ~300 k)
  double p_cancel_{0.035};  // CANCELED trip updates
  double punctual_{0.45};  // max |delay| < 60 s or no delay field at all
  double delay_median_{3.4};  // of the late trips, minutes
  double delay_p90_{15.0};
  int max_delay_{120};
  std::uint32_t seed_{1U};
};

rt_timetable make_synthetic_rtt(timetable const& tt,
                                date::sys_days const day,
                                double const share,
                                rt_synth_params const& p) {
  auto rtt = rt::create_rt_timetable(tt, day);
  auto const day_idx = tt.day_idx(day);
  auto rng = std::mt19937{p.seed_};
  auto u = std::uniform_real_distribution<double>{0.0, 1.0};
  // lognormal: median m, p90 q  =>  mu = ln m, sigma = ln(q / m) / 1.2816
  auto const sigma =
      std::log(std::max(p.delay_p90_, p.delay_median_ * 1.01) /
               p.delay_median_) /
      1.2816;
  auto delay_d = std::lognormal_distribution<double>{
      std::log(std::max(p.delay_median_, 0.01)), sigma};
  auto n_unchanged = 0U, n_cancelled = 0U, n_delayed = 0U;
  for (auto i = 0U; i != tt.transport_traffic_days_.size(); ++i) {
    auto const t = transport_idx_t{i};
    if (!tt.is_transport_active(t, day_idx) || u(rng) >= share) {
      continue;
    }
    auto const tr = transport{t, day_idx};
    if (u(rng) < p.p_cancel_) {
      auto const n_stops = static_cast<stop_idx_t>(
          tt.route_location_seq_[tt.transport_route_[t]].size());
      rtt.cancel_run(rt::run{.t_ = tr, .stop_range_ = {0U, n_stops}});
      ++n_cancelled;
      continue;
    }
    auto delay = 0;
    if (u(rng) >= p.punctual_) {
      delay = std::clamp(static_cast<int>(std::ceil(delay_d(rng))), 1,
                         std::max(1, p.max_delay_));
      ++n_delayed;
    }
    auto const rt_t =
        rtt.add_rt_transport(source_idx_t{0U}, tt, tr, {}, {}, {}, {},
                             direction_id_t{}, {}, static_cast<delta_t>(delay));
    n_unchanged += rtt.is_unchanged(rt_t) ? 1U : 0U;
  }
  rtt.update_lbs(tt);
  fmt::print(
      "synthetic rt: day={}, share={:.2f}, punctual={:.2f}, p_cancel={:.3f}, "
      "delay median {:.1f} / p90 {:.1f} min -> {} rt transports, {} "
      "unchanged (static scan), {} delayed, {} cancelled, deviation interval "
      "{}\n",
      day, share, p.punctual_, p.p_cancel_, p.delay_median_, p.delay_p90_,
      rtt.n_rt_transports(), n_unchanged, n_delayed, n_cancelled,
      rtt.deviation_interval_);
  return rtt;
}

// Moves a query's start time (point or interval) onto `day`, keeping the time
// of day, so that it can see the synthetic real-time data.
void move_to_day(routing::start_time_t& start_time, date::sys_days const day) {
  auto const shift = [&](unixtime_t const t) {
    return unixtime_t{day + (t - std::chrono::floor<date::days>(t))};
  };
  std::visit(utl::overloaded{[&](unixtime_t& t) { t = shift(t); },
                             [&](interval<unixtime_t>& i) {
                               i = {shift(i.from_), shift(i.to_)};
                             }},
             start_time);
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
  std::vector<double> latencies_;
};

// "<rt cell> rt overhead vs <static cell>: ..." -- latency ratios of the two
// cells and the number of queries whose result set changed under real-time
std::string rt_overhead_line(result_set const& rt, result_set const& st) {
  auto const valid = [](std::vector<double> v) {
    std::erase_if(v, [](double const x) { return x < 0.0; });
    std::sort(begin(v), end(v));
    return v;
  };
  auto const avg = [](std::vector<double> const& v) {
    return v.empty() ? 0.0
                     : std::accumulate(begin(v), end(v), 0.0) /
                           static_cast<double>(v.size());
  };
  auto const q = [](std::vector<double> const& v, double const p) {
    return v.empty() ? 0.0
                     : v[std::min(v.size() - 1U,
                                  static_cast<std::size_t>(p * v.size()))];
  };
  auto const pct = [](double const a, double const b) {
    return b <= 0.0 ? 0.0 : (a / b - 1.0) * 100.0;
  };
  auto const keys = [](pareto_set<routing::journey> const& js) {
    auto v = std::vector<std::tuple<unixtime_t, unixtime_t, std::uint8_t>>{};
    for (auto const& j : js) {
      v.emplace_back(j.start_time_, j.dest_time_, j.transfers_);
    }
    std::sort(begin(v), end(v));
    return v;
  };
  auto const a = valid(rt.latencies_), b = valid(st.latencies_);
  auto n_changed = 0U;
  for (auto i = std::size_t{0U};
       i != std::min(rt.res_.size(), st.res_.size()); ++i) {
    n_changed += keys(rt.res_[i]) != keys(st.res_[i]) ? 1U : 0U;
  }
  return fmt::format(
      "{:<36} rt overhead vs {:<24} avg {:+.1f} %  median {:+.1f} %  q90 "
      "{:+.1f} %  (ms: {:.0f} vs {:.0f})  result sets changed {}/{}",
      rt.label_, st.label_, pct(avg(a), avg(b)), pct(q(a, 0.5), q(b, 0.5)),
      pct(q(a, 0.9), q(b, 0.9)), avg(a), avg(b), n_changed, rt.res_.size());
}


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
                       out.res_[i] = search(w, std::move(q));
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
  auto rt_shares = std::vector<double>{};
  auto rt_params = rt_synth_params{};
  auto rt_day_str = std::string{};
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
      ("rt_share", bpo::value(&rt_shares)->multitoken(),
       "additionally run the matrix against synthetic real-time "
       "timetable(s) and report their overhead against the static run: "
       "share(s) of the transports active on --rt_day that get an rt "
       "instance (DELFI feed: ~0.55 of the trips running in its window)")  //
      ("rt_day", bpo::value(&rt_day_str),
       "day (YYYY-MM-DD) the synthetic real-time data is generated for; "
       "query start times are moved onto this day (time of day kept)")  //
      ("rt_punctual",
       bpo::value(&rt_params.punctual_)->default_value(0.45),
       "share of the covered transports reported on time at minute "
       "resolution (DELFI: 0.45)")  //
      ("rt_p_cancel",
       bpo::value(&rt_params.p_cancel_)->default_value(0.035),
       "share of the covered transports that are cancelled (DELFI: 0.035)")  //
      ("rt_delay_median",
       bpo::value(&rt_params.delay_median_)->default_value(3.4),
       "median delay of the late transports in minutes, lognormal "
       "(DELFI: 3.4)")  //
      ("rt_delay_p90",
       bpo::value(&rt_params.delay_p90_)->default_value(15.0),
       "90th percentile of that delay in minutes (DELFI: 15)")  //
      ("rt_max_delay",
       bpo::value(&rt_params.max_delay_)->default_value(120),
       "delays are capped at this many minutes")  //
      ("rt_seed", bpo::value(&rt_params.seed_)->default_value(1U),
       "seed of the synthetic real-time timetable")  //
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

  auto rt_day = std::optional<date::sys_days>{};
  if (!rt_day_str.empty()) {
    auto d = date::sys_days{};
    auto ss = std::stringstream{rt_day_str};
    ss >> date::parse("%F", d);
    if (ss.fail() || !tt.internal_interval_days().contains(d)) {
      std::cerr << "--rt_day " << rt_day_str
                << " is not a day inside the timetable\n";
      return 1;
    }
    rt_day = d;
  }
  // the static run first (nullptr), then one real-time timetable per
  // --rt_share value; without --rt_share this is the plain benchmark
  auto rtts = std::vector<std::unique_ptr<rt_timetable>>{};
  rtts.push_back(nullptr);
  std::erase_if(rt_shares, [](double const x) { return x <= 0.0; });
  for (auto const share : rt_shares) {
    if (!rt_day.has_value()) {
      std::cerr << "--rt_share requires --rt_day\n";
      return 1;
    }
    rtts.push_back(std::make_unique<rt_timetable>(
        make_synthetic_rtt(tt, *rt_day, share, rt_params)));
  }
  rt_shares.insert(begin(rt_shares), 0.0);  // index 0 = static run
  auto const rt_label = [&](std::size_t const i) {
    return i == 0U ? std::string{} : fmt::format("-rt{:.2f}", rt_shares[i]);
  };

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
    for (auto& r : rtts) {
      if (r != nullptr) {
        r->gpu_rtt_.ptr_ = routing::gpu::make_gpu_rtt(tt, *r);
      }
    }
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
      if (rt_day.has_value()) {
        for (auto& sdq : fwd_qs) {
          move_to_day(sdq.q_.start_time_, *rt_day);
        }
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

      // one cell set per --rt_share value: engines/algos are compared within
      // a set, every set with real-time data is measured against the static
      // one
      auto cells_by_rt = std::vector<std::vector<result_set>>(rtts.size());
      for (auto rt_i = std::size_t{0U}; rt_i != rtts.size(); ++rt_i) {
      auto const* const rtt_ptr = rtts[rt_i].get();
      auto& cells = cells_by_rt[rt_i];
      for (auto const& algo : algos) {
        auto const use_pong = algo == "pong";
        auto const label =
            mode + "-" + dir_str + "-" + algo + rt_label(rt_i);

        try {
          if (run_cpu) {
            cells.push_back(run_cell<cpu_ws>(
                qs, label + "-cpu", threads_v,
                [&](cpu_ws& w, routing::query q) {
                  auto const r =
                      use_pong
                          ? routing::pong_search(tt, rtt_ptr, w.ss_, w.rs_,
                                                 std::move(q), dir)
                          : routing::raptor_search(tt, rtt_ptr, w.ss_, w.rs_,
                                                   std::move(q), dir);
                  return *r.journeys_;
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
                          ? routing::pong_search(tt, rtt_ptr, w.ss_, *w.rs_,
                                                 std::move(q), dir)
                          : routing::raptor_search(tt, rtt_ptr, w.ss_, *w.rs_,
                                                   std::move(q), dir);
                  return *r.journeys_;
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
      }  // rt_i

      for (auto const& cells : cells_by_rt) {
        if (cells.size() == 1U) {
          summary.push_back(fmt::format("{:<24} n={:<6} benchmark only",
                                        cells.front().label_, qs.size()));
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

      // real-time overhead: every real-time cell against the static cell of
      // the same (mode, dir, algo, engine); result sets are expected to
      // differ, their count is reported, not failed
      {
        auto const& static_cells = cells_by_rt[0U];
        for (auto rt_i = std::size_t{1U}; rt_i != rtts.size(); ++rt_i) {
          for (auto const& c : cells_by_rt[rt_i]) {
            auto base = c.label_;
            base.erase(base.find(rt_label(rt_i)), rt_label(rt_i).size());
            auto const twin =
                utl::find_if(static_cells, [&](result_set const& x) {
                  return x.label_ == base;
                });
            if (twin == end(static_cells)) {
              continue;
            }
            summary.push_back(rt_overhead_line(c, *twin));
          }
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
