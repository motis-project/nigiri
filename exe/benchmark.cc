#include <condition_variable>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <regex>
#include <thread>

#include "boost/program_options.hpp"

#include "utl/parallel_for.h"
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

std::vector<std::string> tokenize(std::string_view const& str,
                                  char delimiter,
                                  std::uint32_t n_tokens) {
  auto tokens = std::vector<std::string>{};
  tokens.reserve(n_tokens);
  auto start = 0U;
  for (auto i = 0U; i != n_tokens; ++i) {
    auto end = str.find(delimiter, start);
    if (end == std::string::npos && i != n_tokens - 1U) {
      break;
    }
    tokens.emplace_back(str.substr(start, end - start));
    start = end + 1U;
  }
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
  auto const tokens = tokenize(str, ',', 4U);
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
  auto const tokens = tokenize(str_trimmed, ',', 2U);
  return latlng{std::stod(tokens[0]), std::stod(tokens[1])};
}

struct benchmark_result {
  friend std::ostream& operator<<(std::ostream& out,
                                  benchmark_result const& br) {
    using double_seconds_t = std::chrono::duration<double, std::ratio<1>>;
    out << "(t_total: " << std::fixed << std::setprecision(3) << std::setw(9)
        << std::chrono::duration_cast<double_seconds_t>(br.total_time_).count()
        << "s, t_exec: " << std::setw(9)
        << std::chrono::duration_cast<double_seconds_t>(
               br.routing_result_.search_stats_.execute_time_)
               .count()
        << "s, t_lb: " << std::setw(6)
        << (static_cast<double>(br.routing_result_.search_stats_.lb_time_) /
            1000.0)
        << "s, intvl_ext: " << std::setw(2)
        << br.routing_result_.search_stats_.interval_extensions_
        << ", intvl_size: " << std::setw(5)
        << std::chrono::duration_cast<std::chrono::hours>(
               br.routing_result_.interval_.size())
               .count()
        << "h" << ", #jrny: " << std::setfill(' ') << std::setw(2)
        << br.journeys_.size() << ")";
    return out;
  }

  std::uint64_t q_idx_;
  routing_result routing_result_;
  pareto_set<journey> journeys_;
  std::chrono::milliseconds total_time_;
};

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
  auto query_generation_timer = scoped_timer(fmt::format(
      "generation of {} queries using seed {}", n_queries, qg.seed_));
  std::cout << "--- Query generator settings ---\n" << gs << "\n--- --- ---\n";
  queries.reserve(n_queries);
  for (auto i = 0U; i != n_queries; ++i) {
    auto const sdq = qg.random_query();
    if (sdq.has_value()) {
      queries.emplace_back(sdq.value());
    }
  }
  std::cout << queries.size() << " queries generated successfully\n";
}

std::string to_str(timetable const& tt,
                   pareto_set<nigiri::routing::journey> const& results) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n\n";
  }
  return ss.str();
}

nigiri::pareto_set<nigiri::routing::journey> raptor_search(
    nigiri::timetable const& tt, nigiri::routing::query q) {
  using namespace nigiri;
  using algo_state_t = routing::raptor_state;
  static auto search_state = routing::search_state{};
  static auto algo_state = algo_state_t{};

  return *(routing::raptor_search(tt, nullptr, search_state, algo_state,
                                  std::move(q), nigiri::direction::kForward)
               .journeys_);
}

// CPU-vs-GPU pareto disagreements accumulated over a process_queries run;
// used as the CI gate (any nonzero count = engines diverged).
struct compare_stats {
  std::uint64_t gpu_misses_{0U};  // CPU journeys not covered by GPU pareto
  std::uint64_t cpu_misses_{0U};  // GPU journeys not covered by CPU pareto
};

compare_stats process_queries(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    std::vector<benchmark_result>& results,
    nigiri::timetable const& tt,
    bool const use_pong) {
  results.reserve(queries.size());
  auto stats = compare_stats{};

  auto mutex = std::mutex{};
  {
#if defined(NIGIRI_CUDA)
    std::cout << "creating GPU timetable\n";
    auto const gpu_tt = routing::gpu::gpu_timetable{tt};
    std::cout << "creating GPU timetable finished\n";
#endif

    auto query_processing_timer =
        scoped_timer(fmt::format("processing of {} queries", queries.size()));

    auto const progress_tracker = utl::activate_progress_tracker("benchmark");
    progress_tracker->status("processing queries").in_high(queries.size());

    struct query_state {
      search_state ss_;
      raptor_state rs_;
#if defined(NIGIRI_CUDA)
      search_state gpu_ss_;  // separate required for result comparison
      std::unique_ptr<routing::gpu::gpu_raptor_state> gpu_rs_;
#endif
    } query_state;

    for (auto q_idx = 0U; q_idx != queries.size(); ++q_idx) {
#if defined(NIGIRI_CUDA)
      if (query_state.gpu_rs_.get() == nullptr) {
        query_state.gpu_rs_ =
            std::make_unique<routing::gpu::gpu_raptor_state>(gpu_tt);
      }
#endif
      try {
        auto const total_time_start = std::chrono::steady_clock::now();
        auto const result =
            use_pong ? routing::pong_search(tt, nullptr, query_state.ss_,
                                            query_state.rs_, queries[q_idx].q_,
                                            direction::kForward)
                     : routing::raptor_search(
                           tt, nullptr, query_state.ss_, query_state.rs_,
                           queries[q_idx].q_, direction::kForward);
        auto const total_time_stop = std::chrono::steady_clock::now();
        auto const total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                total_time_stop - total_time_start);

#if defined(NIGIRI_CUDA)
        auto const gpu_total_time_start = std::chrono::steady_clock::now();
        auto const gpu_result =
            use_pong
                ? routing::pong_search(tt, nullptr, query_state.gpu_ss_,
                                       *query_state.gpu_rs_, queries[q_idx].q_,
                                       direction::kForward)
                : routing::raptor_search(
                      tt, nullptr, query_state.gpu_ss_, *query_state.gpu_rs_,
                      queries[q_idx].q_, direction::kForward);
        auto const gpu_total_time_stop = std::chrono::steady_clock::now();

        auto const total_gpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                gpu_total_time_stop - gpu_total_time_start);

        std::cout << "cpu=" << total_us.count()
                  << "us, gpu=" << total_gpu_us.count() << "us\n";

        auto const dominates = [](journey const& a, journey const& b) {
          return a.start_time_ >= b.start_time_ &&
                 a.dest_time_ <= b.dest_time_ && a.transfers_ <= b.transfers_;
        };
        auto const covered = [&](journey const& j, auto const& by) {
          return std::any_of(begin(by), end(by),
                             [&](journey const& o) { return dominates(o, j); });
        };
        auto gpu_misses = 0, cpu_misses = 0;
        for (auto const& c : *result.journeys_) {
          if (!covered(c, *gpu_result.journeys_)) {
            ++gpu_misses;
          }
        }
        for (auto const& g : *gpu_result.journeys_) {
          if (!covered(g, *result.journeys_)) {
            ++cpu_misses;
          }
        }
        if (gpu_misses != 0 || cpu_misses != 0) {
          auto const key = [](journey const& j) {
            return fmt::format("dep={} arr={} transfers={}", j.start_time_,
                               j.dest_time_, j.transfers_);
          };
          std::cout << "query #" << q_idx << " GPU_MISSES_CPU=" << gpu_misses
                    << " CPU_MISSES_GPU=" << cpu_misses
                    << " cpu_intvl=" << result.interval_
                    << " gpu_intvl=" << gpu_result.interval_ << "\n";
          if (gpu_misses != 0) {
            std::cout << "  GPU-pareto: ";
            for (auto const& g : *gpu_result.journeys_) {
              std::cout << "[" << key(g) << "] ";
            }
            std::cout << "\n";
            for (auto const& c : *result.journeys_) {
              if (!covered(c, *gpu_result.journeys_)) {
                std::cout << "  === MISSED-BY-GPU: " << key(c) << " ===\n";
                c.print(std::cout, tt);
                std::cout << "\n";
              }
            }
          }
          if (cpu_misses != 0) {
            std::cout << "  CPU-pareto: ";
            for (auto const& c : *result.journeys_) {
              std::cout << "[" << key(c) << "] ";
            }
            std::cout << "\n";
            for (auto const& g : *gpu_result.journeys_) {
              if (!covered(g, *result.journeys_)) {
                std::cout << "  === GPU-EXTRA (not dominated by CPU): "
                          << key(g) << " ===\n";
                g.print(std::cout, tt);
                std::cout << "\n";
              }
            }
          }
        }
        stats.gpu_misses_ += static_cast<std::uint64_t>(gpu_misses);
        stats.cpu_misses_ += static_cast<std::uint64_t>(cpu_misses);
#else
        std::cout << "cpu=" << total_us.count() << "us\n";
#endif
        auto const guard = std::lock_guard{mutex};
        results.emplace_back(benchmark_result{
            q_idx, result, *result.journeys_,
            std::chrono::duration_cast<std::chrono::milliseconds>(
                total_time_stop - total_time_start)});
        progress_tracker->increment();
      } catch (std::exception const& e) {
        std::cerr << "query #" << q_idx << " FAILED: " << e.what() << std::endl;
      }
    }
  }
  return stats;
}

// needs sorted vector
template <typename T>
T quantile(std::vector<T> const& v, double q) {
  q = q < 0.0 ? 0.0 : q;
  q = 1.0 < q ? 1.0 : q;
  if (q == 1.0) {
    return v.back();
  }
  return v[static_cast<std::size_t>(v.size() * q)];
}

// GPU-only throughput test: run pong on `n_threads` worker threads, each with
// its own gpu_raptor_state + stream, all sharing the read-only gpu_timetable.
// Measures sustained queries/sec (how much the per-query GPU-idle slack can be
// reclaimed by concurrency, up to the DRAM-bandwidth ceiling).
// per-query latency (ms) percentiles under concurrent load; `lat` may contain
// unset (-1) slots for queries that were never run / failed.
void print_load_latency(std::vector<double> lat, std::string const& tag) {
  std::erase_if(lat, [](double const x) { return x < 0.0; });
  if (lat.empty()) {
    return;
  }
  std::sort(begin(lat), end(lat));
  auto const pct = [&](double const p) {
    return lat[std::min(lat.size() - 1,
                        static_cast<std::size_t>(p * lat.size()))];
  };
  auto const avg = std::accumulate(begin(lat), end(lat), 0.0) /
                   static_cast<double>(lat.size());
  std::cout << tag << " under-load latency ms: avg=" << avg
            << " min=" << lat.front() << " p10=" << pct(0.1)
            << " p25=" << pct(0.25) << " median=" << pct(0.5)
            << " p75=" << pct(0.75) << " p90=" << pct(0.9)
            << " p99=" << pct(0.99) << " max=" << lat.back() << "\n";
}

// ---- profiling matrix: N worker threads sharing a pool of M states ----
// A "state" is one search pipeline (search_state + algo state); the GPU variant
// additionally owns a CUDA stream + device buffers. n_states caps the number of
// concurrent searches; n_threads that exceed it block on the pool. Lets us
// measure how throughput/latency scale with request concurrency vs. the number
// of GPU pipelines (or CPU worker states) provisioned.
template <typename WS, typename Factory, typename SearchFn>
void run_load(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    std::string const& tag,
    unsigned const n_threads,
    unsigned const n_states,
    Factory make_state,
    SearchFn search_one) {
  auto states = std::vector<std::unique_ptr<WS>>{};
  states.reserve(n_states);
  for (auto i = 0U; i != n_states; ++i) {
    states.push_back(make_state());
  }
  // Warm up each state so lazy device-buffer allocation is excluded from the
  // timing, and -- crucially for the GPU state sweep -- so an out-of-memory
  // failure surfaces HERE (propagating to the caller, which stops the sweep)
  // instead of as per-query worker failures once threads start.
  if (!queries.empty()) {
    for (auto& s : states) {
      search_one(*s, queries.front().q_);
    }
  }

  auto free = std::vector<WS*>{};
  for (auto& s : states) {
    free.push_back(s.get());
  }
  auto pm = std::mutex{};
  auto pcv = std::condition_variable{};

  auto next = std::atomic<std::size_t>{0};
  auto done = std::atomic<std::size_t>{0};
  auto lat = std::vector<double>(queries.size(), -1.0);
  auto const t0 = std::chrono::steady_clock::now();
  auto workers = std::vector<std::thread>{};
  for (auto t = 0U; t != n_threads; ++t) {
    workers.emplace_back([&]() {
      for (auto i = next.fetch_add(1); i < queries.size();
           i = next.fetch_add(1)) {
        WS* ws = nullptr;
        {
          auto lk = std::unique_lock{pm};
          pcv.wait(lk, [&]() { return !free.empty(); });
          ws = free.back();
          free.pop_back();
        }
        try {
          auto const q0 = std::chrono::steady_clock::now();
          search_one(*ws, queries[i].q_);
          lat[i] = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - q0)
                       .count();
          done.fetch_add(1);
        } catch (std::exception const& e) {
          std::cerr << "q#" << i << " FAILED: " << e.what() << std::endl;
        }
        {
          auto lk = std::lock_guard{pm};
          free.push_back(ws);
        }
        pcv.notify_one();
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
      "PROFILE tag={} threads={:>2} states={} | {:>9.1f} q/s | ms avg={:>7.2f} "
      "min={:>6.2f} p50={:>6.2f} p75={:>6.2f} p90={:>7.2f} p95={:>7.2f} "
      "p99={:>7.2f} p99.9={:>8.2f} max={:>8.2f}\n",
      tag, n_threads, n_states, qps, avg, l.empty() ? 0.0 : l.front(), q(0.50),
      q(0.75), q(0.90), q(0.95), q(0.99), q(0.999), l.empty() ? 0.0 : l.back());
}

void profile_matrix(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    timetable const& tt,
    bool const use_pong) {
  auto const threads = std::vector<unsigned>{1U, 8U, 16U, 32U};
  auto const states = std::vector<unsigned>{1U, 2U, 5U};

  struct cpu_ws {
    search_state ss_;
    routing::raptor_state rs_;
  };
  auto const cpu_search = [&](cpu_ws& w, routing::query q) {
    use_pong ? routing::pong_search(tt, nullptr, w.ss_, w.rs_, std::move(q),
                                    direction::kForward)
             : routing::raptor_search(tt, nullptr, w.ss_, w.rs_, std::move(q),
                                      direction::kForward);
  };

  // CPU: a raptor_state is just per-worker scratch (no device pipeline / stream
  // limit), so the natural config is one state per thread -- concurrency is
  // bounded only by cores. Sweep threads with states == threads.
  std::cout << "=== CPU profile matrix (" << (use_pong ? "pong" : "raptor")
            << ") ===\n";
  for (auto const t : threads) {
    run_load<cpu_ws>(
        queries, "cpu", t, t, [] { return std::make_unique<cpu_ws>(); },
        cpu_search);
  }

#if defined(NIGIRI_CUDA)
  auto const gpu_tt = routing::gpu::gpu_timetable{tt};
  struct gpu_ws {
    search_state ss_;
    std::unique_ptr<routing::gpu::gpu_raptor_state> rs_;
  };
  auto const gpu_search = [&](gpu_ws& w, routing::query const& qq) {
    auto q = qq;
    use_pong ? routing::pong_search(tt, nullptr, w.ss_, *w.rs_, std::move(q),
                                    direction::kForward)
             : routing::raptor_search(tt, nullptr, w.ss_, *w.rs_, std::move(q),
                                      direction::kForward);
  };

  // GPU: "states" == concurrent device pipelines (each its own stream + device
  // buffers). More states hide the CPU-side prep/reconstruct behind other
  // states' kernels, up to the point the GPU (or the CPU feeding it) saturates.
  // Drive with a constant, generously-oversubscribed thread count so every
  // state count is fully fed; blocked threads just wait on the pool condvar.
  auto const gpu_states = std::vector<unsigned>{1U, 2U, 5U};
  auto const gpu_feed_threads = 32U;
  std::cout << "=== GPU profile matrix (" << (use_pong ? "pong" : "raptor")
            << ") ===\n";
  for (auto const s : gpu_states) {
    try {
      run_load<gpu_ws>(
          queries, "gpu", gpu_feed_threads, s,
          [&] {
            return std::make_unique<gpu_ws>(gpu_ws{
                search_state{},
                std::make_unique<routing::gpu::gpu_raptor_state>(gpu_tt)});
          },
          gpu_search);
    } catch (std::exception const& e) {
      std::cout << "gpu states=" << s << ": setup failed (" << e.what()
                << ") -- stopping sweep (GPU RAM limit reached)\n";
      break;
    }
  }
#endif
}

#if defined(NIGIRI_CUDA)
void throughput_test(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    timetable const& tt,
    unsigned const n_threads,
    bool const use_pong) {
  auto const gpu_tt = routing::gpu::gpu_timetable{tt};
  std::cout << "throughput: " << n_threads << " threads, " << queries.size()
            << " queries\n";
  auto next = std::atomic<std::size_t>{0};
  auto done = std::atomic<std::size_t>{0};
  auto lat = std::vector<double>(queries.size(), -1.0);
  auto lat_lb = std::vector<double>(queries.size(), -1.0);
  auto const t0 = std::chrono::steady_clock::now();
  auto workers = std::vector<std::thread>{};
  for (auto t = 0U; t != n_threads; ++t) {
    workers.emplace_back([&]() {
      auto ss = search_state{};
      auto gpu_rs = std::make_unique<routing::gpu::gpu_raptor_state>(gpu_tt);
      for (auto i = next.fetch_add(1); i < queries.size();
           i = next.fetch_add(1)) {
        try {
          auto const q0 = std::chrono::steady_clock::now();
          auto const r =
              use_pong
                  ? routing::pong_search(tt, nullptr, ss, *gpu_rs,
                                         queries[i].q_, direction::kForward)
                  : routing::raptor_search(tt, nullptr, ss, *gpu_rs,
                                           queries[i].q_, direction::kForward);
          lat[i] = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - q0)
                       .count();
          lat_lb[i] = static_cast<double>(r.search_stats_.lb_time_);
          if (!r.journeys_->empty() || true) {
            done.fetch_add(1);
          }
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
  std::cout << "throughput: " << d << " queries in " << ms << "ms = "
            << (d * 1000.0 / static_cast<double>(std::max<std::int64_t>(ms, 1)))
            << " queries/sec (" << n_threads << " threads)\n";
  print_load_latency(std::move(lat), "gpu");
  print_load_latency(std::move(lat_lb), "lb(cpu)");
}
#endif

// CPU counterpart of throughput_test: N worker threads, each its own
// raptor_state, running CPU pong over the shared timetable. For a fair
// CPU-vs-GPU sustained-throughput comparison.
void throughput_test_cpu(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    timetable const& tt,
    unsigned const n_threads,
    bool const use_pong) {
  std::cout << "throughput(cpu): " << n_threads << " threads, "
            << queries.size() << " queries\n";
  auto next = std::atomic<std::size_t>{0};
  auto done = std::atomic<std::size_t>{0};
  auto lat = std::vector<double>(queries.size(), -1.0);
  auto const t0 = std::chrono::steady_clock::now();
  auto workers = std::vector<std::thread>{};
  for (auto t = 0U; t != n_threads; ++t) {
    workers.emplace_back([&]() {
      auto ss = search_state{};
      auto rs = routing::raptor_state{};
      for (auto i = next.fetch_add(1); i < queries.size();
           i = next.fetch_add(1)) {
        try {
          auto const q0 = std::chrono::steady_clock::now();
          use_pong ? routing::pong_search(tt, nullptr, ss, rs, queries[i].q_,
                                          direction::kForward)
                   : routing::raptor_search(tt, nullptr, ss, rs, queries[i].q_,
                                            direction::kForward);
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
  std::cout << "throughput(cpu): " << d << " queries in " << ms << "ms = "
            << (d * 1000.0 / static_cast<double>(std::max<std::int64_t>(ms, 1)))
            << " queries/sec (" << n_threads << " threads)\n";
  print_load_latency(std::move(lat), "cpu");
}

void print_result(std::vector<benchmark_result> const& var,
                  std::string const& var_name) {
  if (var.empty()) {
    std::cout << "\n--- " << var_name
              << " --- (n = 0, no successful queries)\n";
    return;
  }
  std::cout << "\n--- " << var_name << " --- (n = " << var.size() << ")"
            << "\n  10%: " << quantile(var, 0.1)
            << "\n  20%: " << quantile(var, 0.2)
            << "\n  30%: " << quantile(var, 0.3)
            << "\n  40%: " << quantile(var, 0.4)
            << "\n  50%: " << quantile(var, 0.5)
            << "\n  60%: " << quantile(var, 0.6)
            << "\n  70%: " << quantile(var, 0.7)
            << "\n  80%: " << quantile(var, 0.8)
            << "\n  90%: " << quantile(var, 0.9)
            << "\n  99%: " << quantile(var, 0.99)
            << "\n99.9%: " << quantile(var, 0.999) << "\n  max: " << var.back()
            << "\n----------------------------------\n";
}

void print_results(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    std::vector<benchmark_result>& results,
    nigiri::timetable const& tt,
    nigiri::query_generation::generator_settings const& gs,
    std::filesystem::path const& tt_path) {
  utl::sort(results, [](auto const& a, auto const& b) {
    return a.total_time_ < b.total_time_;
  });
  print_result(results, "total_time");

  auto const visit_coord = [](geo::latlng const& coord) {
    std::stringstream ss;
    ss << coord;
    return ss.str();
  };

  auto const visit_loc_idx = [&](location_idx_t const loc_idx) {
    std::stringstream ss;
    ss << "loc_idx: " << loc_idx.v_
       << ", name: " << tt.get_default_name(loc_idx)
       << ", coord: " << tt.locations_.coordinates_[loc_idx];
    return ss.str();
  };

  auto const print_slow_result = [&](auto const& br) {
    std::cout << br << "\nstart: "
              << std::visit(utl::overloaded{visit_loc_idx, visit_coord},
                            queries[br.q_idx_].start_)
              << "\ndest: "
              << std::visit(utl::overloaded{visit_loc_idx, visit_coord},
                            queries[br.q_idx_].dest_)
              << "\n";
  };
  std::cout << "\nSlowest Queries:\n";
  for (auto i = 0; i != results.size() && i != 10; ++i) {
    std::cout << "\n--- " << i + 1
              << " ---\nquery_idx: " << rbegin(results)[i].q_idx_ << '\n';
    print_slow_result(rbegin(results)[i]);
  }
  std::cout << "\n";

  auto ss = std::stringstream{};
  ss << "Re-run the slowest source-destination "
        "combination:\n./nigiri-benchmark -p "
     << tt_path.string() << " -n 1 -i " << gs.interval_size_.count();
  if (gs.start_match_mode_ == location_match_mode::kIntermodal) {
    ss << " --start_mode intermodal --intermodal_start "
       << to_string(gs.start_mode_).value();
  } else {
    ss << " --start_mode station";
  }
  if (gs.dest_match_mode_ == location_match_mode::kIntermodal) {
    ss << " --dest_mode intermodal --intermodal_dest "
       << to_string(gs.dest_mode_).value();
  } else {
    ss << " --dest_mode station";
  }
  ss << " --use_start_footpaths " << gs.use_start_footpaths_ << " -t "
     << std::uint32_t{gs.max_transfers_} << " -m " << gs.min_connection_count_
     << " -e " << gs.extend_interval_earlier_ << " -l "
     << gs.extend_interval_later_ << " --profile_idx "
     << std::uint32_t{gs.prf_idx_} << " --allowed_claszes "
     << gs.allowed_claszes_;
  if (gs.start_match_mode_ == location_match_mode::kIntermodal) {
    ss << " --start_coord \""
       << get<geo::latlng>(queries[rbegin(results)[0].q_idx_].start_) << "\"";
  } else {
    ss << " --start_loc "
       << get<location_idx_t>(queries[rbegin(results)[0].q_idx_].start_);
  }
  if (gs.dest_match_mode_ == location_match_mode::kIntermodal) {
    ss << " --dest_coord \""
       << get<geo::latlng>(queries[rbegin(results)[0].q_idx_].dest_) << "\"";
  } else {
    ss << " --dest_loc "
       << get<location_idx_t>(queries[rbegin(results)[0].q_idx_].dest_) << "\n";
  }
  std::cout << ss.str() << "\n";

  utl::sort(results, [](auto const& a, auto const& b) {
    return a.routing_result_.search_stats_.execute_time_ <
           b.routing_result_.search_stats_.execute_time_;
  });
  print_result(results, "execute_time");

  utl::sort(results, [](auto const& a, auto const& b) {
    return a.routing_result_.search_stats_.interval_extensions_ <
           b.routing_result_.search_stats_.interval_extensions_;
  });
  print_result(results, "interval_extensions");

  utl::sort(results, [](auto const& a, auto const& b) {
    return a.routing_result_.interval_.size() <
           b.routing_result_.interval_.size();
  });
  print_result(results, "interval_size");

  utl::sort(results, [](auto const& a, auto const& b) {
    return a.journeys_.size() < b.journeys_.size();
  });
  print_result(results, "#journeys");
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
  namespace bpo = boost::program_options;

  auto tt_path = std::filesystem::path{};
  auto n_queries = std::uint32_t{100U};
  auto gs = query_generation::generator_settings{};
  auto interval_size = duration_t::rep{};
  auto bbox_str = std::string{};
  auto start_mode_str = std::string{};
  auto dest_mode_str = std::string{};
  auto intermodal_start_str = std::string{};
  auto intermodal_dest_str = std::string{};
  auto max_transfers = std::uint32_t{kMaxTransfers};
  auto gpu_threads = std::uint32_t{0};
  auto cpu_threads = std::uint32_t{0};
  auto profile_mat = false;
  auto prf_idx = std::uint32_t{0};
  auto start_coord_str = std::string{};
  auto dest_coord_str = std::string{};
  auto start_loc_val = location_idx_t::value_t{0U};
  auto dest_loc_val = location_idx_t::value_t{0U};
  auto seed = std::int64_t{-1};
  auto min_transfer_time = duration_t::rep{};
  auto qa_path = std::filesystem::path{};
  auto algorithm = std::string{"raptor"};
  auto runs_str = std::string{};

  bpo::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce this help message")  //
      ("tt_path,p", bpo::value(&tt_path)->required(),
       "path to a binary file containing a serialized nigiri timetable")  //
      ("algorithm,a", bpo::value(&algorithm)->default_value(algorithm),
       "routing algorithm: 'raptor' (interval extension) or 'pong'")  //
      ("runs", bpo::value(&runs_str),
       "comma-separated list of <base>-<algo> configurations to run on the "
       "once-loaded timetable (base: station | intermodal, algo: raptor | "
       "pong), e.g. station-pong,intermodal-raptor; query sets are generated "
       "once per base and shared across its algorithms; exits non-zero if CPU "
       "and GPU results diverge")  //
      ("seed,s", bpo::value<std::int64_t>(&seed),
       "value to seed the RNG of the query generator with, "
       "omit for random seed")  //
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
      ("start_mode",
       bpo::value<std::string>(&start_mode_str)->default_value("intermodal"),
       "intermodal | station")  //
      ("dest_mode",
       bpo::value<std::string>(&dest_mode_str)->default_value("intermodal"),
       "intermodal | station")  //
      ("intermodal_start",
       bpo::value<std::string>(&intermodal_start_str)->default_value("walk"),
       "walk | bicycle | car")  //
      ("intermodal_dest",
       bpo::value<std::string>(&intermodal_dest_str)->default_value("walk"),
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
       "path to write the journey criteria to for qa")  //
      ("gpu_threads",
       bpo::value<std::uint32_t>(&gpu_threads)->default_value(0U),
       "if >0: run GPU-only pong throughput test with this many worker "
       "threads (each its own state) instead of the cpu-vs-gpu benchmark")  //
      ("cpu_threads",
       bpo::value<std::uint32_t>(&cpu_threads)->default_value(0U),
       "if >0: run CPU-only pong throughput test with this many worker "
       "threads (each its own raptor_state)")  //
      ("profile_matrix", bpo::bool_switch(&profile_mat),
       "run the CPU-vs-GPU throughput/latency matrix over {1,8,16,32} threads "
       "x {1,2,5} states and exit");
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

  {
    auto fp_counts = std::vector<std::uint32_t>{};
    fp_counts.reserve(tt.n_locations());
    for (auto i = 0U; i != tt.n_locations(); ++i) {
      fp_counts.push_back(static_cast<std::uint32_t>(
          tt.locations_.footpaths_out_[0][location_idx_t{i}].size()));
    }
    utl::sort(fp_counts);
    auto const q = [&](double const p) {
      return fp_counts[static_cast<std::size_t>(
          p * static_cast<double>(fp_counts.size() - 1U))];
    };
    std::cout << "footpaths_out[0] per location: p50=" << q(.5)
              << " p90=" << q(.9) << " p99=" << q(.99) << " p99.9=" << q(.999)
              << " max=" << fp_counts.back() << "\n";
  }

  gs.interval_size_ = duration_t{interval_size};

  if (!bbox_str.empty()) {
    gs.bbox_ = parse_bbox(bbox_str);
    if (!gs.bbox_.has_value()) {
      std::cout << "Error: malformed bounding box input\n";
      return 1;
    }
  }

  if (start_mode_str == "intermodal") {
    gs.start_match_mode_ = location_match_mode::kIntermodal;
    if (!intermodal_start_str.empty()) {
      auto const intermodal_start_mode =
          query_generation::to_transport_mode(intermodal_start_str);
      if (intermodal_start_mode.has_value()) {
        gs.start_mode_ = intermodal_start_mode.value();
      } else {
        std::cout << "Error: Unknown intermodal start mode\n";
        return 1;
      }
    }
  } else if (start_mode_str == "station") {
    gs.start_match_mode_ = location_match_mode::kEquivalent;
  } else {
    std::cout << "Error: Invalid start mode\n";
    return 1;
  }

  if (dest_mode_str == "intermodal") {
    gs.dest_match_mode_ = location_match_mode::kIntermodal;
    if (!intermodal_dest_str.empty()) {
      auto const intermodal_dest_mode =
          query_generation::to_transport_mode(intermodal_dest_str);
      if (intermodal_dest_mode.has_value()) {
        gs.dest_mode_ = intermodal_dest_mode.value();
      } else {
        std::cout << "Error: Unknown intermodal start mode\n";
        return 1;
      }
    }
  } else if (dest_mode_str == "station") {
    gs.dest_match_mode_ = location_match_mode::kEquivalent;
  } else {
    std::cout << "Error: Invalid destination mode\n";
    return 1;
  }

  // An intermodal start encodes the first mile in the start offsets; enabling
  // start footpaths would double-count it. search.h rejects the combination via
  // utl::verify, and motis sets it the same way, so force it off here.
  if (gs.start_match_mode_ == location_match_mode::kIntermodal) {
    gs.use_start_footpaths_ = false;
  }

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

  // multi-configuration mode: several <base>-<algo> runs against the ONE
  // loaded timetable (loading dominates wall time for big datasets); query
  // sets are generated once per base and shared across its algorithms.
  if (!runs_str.empty()) {
    auto base_queries =
        std::map<std::string,
                 std::vector<nigiri::query_generation::start_dest_query>>{};
    auto summary = std::vector<std::string>{};
    auto total = compare_stats{};
    for (auto start = std::size_t{0U}; start < runs_str.size();) {
      auto const end = std::min(runs_str.find(',', start), runs_str.size());
      auto const run = runs_str.substr(start, end - start);
      start = end + 1U;

      auto const sep = run.rfind('-');
      auto const base = run.substr(0, sep);
      auto const algo = sep == std::string::npos ? "" : run.substr(sep + 1U);
      if ((base != "station" && base != "intermodal") ||
          (algo != "raptor" && algo != "pong")) {
        std::cerr << "invalid run \"" << run
                  << "\", expected <station|intermodal>-<raptor|pong>\n";
        return 1;
      }

      auto rs = gs;
      if (base == "station") {
        rs.start_match_mode_ = location_match_mode::kEquivalent;
        rs.dest_match_mode_ = location_match_mode::kEquivalent;
      } else {
        rs.start_match_mode_ = location_match_mode::kIntermodal;
        rs.dest_match_mode_ = location_match_mode::kIntermodal;
        rs.start_mode_ = *query_generation::to_transport_mode("walk");
        rs.dest_mode_ = *query_generation::to_transport_mode("walk");
        rs.use_start_footpaths_ = false;  // first mile is in the start offsets
      }

      auto& qs = base_queries[base];
      if (qs.empty()) {
        generate_queries(qs, n_queries, tt, rs, seed);
      }

      std::cout << "\n=== RUN " << run << " ===\n";
      auto results = std::vector<benchmark_result>{};
      auto const cs = process_queries(qs, results, tt, algo == "pong");
      print_results(qs, results, tt, rs, tt_path);

      summary.push_back(fmt::format(
          "{:<24} n={:<6} gpu_misses_cpu={:<4} cpu_misses_gpu={:<4} {}", run,
          qs.size(), cs.gpu_misses_, cs.cpu_misses_,
          cs.gpu_misses_ + cs.cpu_misses_ == 0U ? "PASS" : "FAIL"));
      total.gpu_misses_ += cs.gpu_misses_;
      total.cpu_misses_ += cs.cpu_misses_;
    }

    std::cout << "\n=== RUNS SUMMARY ===\n";
    for (auto const& s : summary) {
      std::cout << s << "\n";
    }
    print_memory_usage();
    return total.gpu_misses_ + total.cpu_misses_ == 0U ? 0 : 1;
  }

  auto queries = std::vector<nigiri::query_generation::start_dest_query>{};
  generate_queries(queries, n_queries, tt, gs, seed);

  auto const use_pong = algorithm == "pong";
  std::cout << "algorithm: " << (use_pong ? "pong" : "raptor") << "\n";

  if (profile_mat) {
    profile_matrix(queries, tt, use_pong);
    return 0;
  }
  if (cpu_threads > 0U) {
    throughput_test_cpu(queries, tt, cpu_threads, use_pong);
  }
  if (gpu_threads > 0U) {
#if defined(NIGIRI_CUDA)
    throughput_test(queries, tt, gpu_threads, use_pong);
#else
    std::cerr << "--gpu_threads requires a NIGIRI_CUDA build\n";
    return 1;
#endif
  }
  if (cpu_threads > 0U || gpu_threads > 0U) {
    return 0;
  }

  auto results = std::vector<benchmark_result>{};
  process_queries(queries, results, tt, use_pong);

  print_results(queries, results, tt, gs, tt_path);

  print_memory_usage();

  auto total = std::chrono::milliseconds{0U};
  for (auto const& res : results) {
    total += res.total_time_;
  }
  std::cout << "AVG: " << (static_cast<double>(total.count()) / results.size())
            << "ms\n";

  if (vm.count("qa_path")) {
    auto bm_crit = nigiri::qa::benchmark_criteria{};
    for (auto const& res : results) {
      auto jc = vector<nigiri::qa::criteria_t>{};
      for (auto const& j : res.journeys_) {
        jc.emplace_back(
            static_cast<double>(j.start_time_.time_since_epoch().count()),
            static_cast<double>(j.dest_time_.time_since_epoch().count()),
            static_cast<double>(j.transfers_));
      }
      utl::sort(jc);
      bm_crit.qc_.emplace_back(res.q_idx_, res.total_time_, jc);
    }
    bm_crit.write(qa_path);
  }

  return 0;
}
