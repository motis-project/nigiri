#include <benchmark/benchmark.h>

#include "nigiri/query_generator/generator.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"

static std::filesystem::path tt_path;

void generate_queries(
    std::vector<nigiri::query_generation::start_dest_query>& queries,
    std::uint32_t n_queries,
    nigiri::timetable const& tt,
    nigiri::query_generation::generator_settings const& gs,
    std::int64_t const seed) {
  auto qg = nigiri::query_generation::generator{
      tt, gs, static_cast<std::uint32_t>(seed)};
  queries.reserve(n_queries);
  for (auto i = 0U; i != n_queries; ++i) {
    auto const sdq = qg.random_query();
    if (sdq.has_value()) {
      queries.emplace_back(sdq.value());
    }
  }
}

struct benchmark_result {
  std::uint64_t q_idx_;
  nigiri::routing::routing_result<nigiri::routing::raptor_stats>
      routing_result_;
  nigiri::pareto_set<nigiri::routing::journey> journeys_;
  std::chrono::milliseconds total_time_;
};

void process_queries(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    std::vector<benchmark_result>& results,
    nigiri::timetable const& tt) {
  results.reserve(queries.size());
  std::mutex mutex;
  {
    struct query_state {
      nigiri::routing::search_state ss_;
      nigiri::routing::raptor_state rs_;
    };
    for (auto i = 0U; i < queries.size(); ++i) {
      try {
        query_state qs;
        auto const total_time_start = std::chrono::steady_clock::now();
        auto const result = nigiri::routing::raptor_search(
            tt, nullptr, qs.ss_, qs.rs_, queries[i].q_,
            nigiri::direction::kForward);
        auto const total_time_stop = std::chrono::steady_clock::now();
        auto const guard = std::lock_guard{mutex};
        results.emplace_back(benchmark_result{
            i, result, *result.journeys_,
            std::chrono::duration_cast<std::chrono::milliseconds>(
                total_time_stop - total_time_start)});
      } catch (const std::exception& e) {
        std::cout << e.what();
      }
    }
  }
}

static void benchmark_random_queries(benchmark::State& state) {
  std::vector<benchmark_result> results;
  ::benchmark::DoNotOptimize(results);
  std::vector<nigiri::query_generation::start_dest_query> queries;
  ::benchmark::DoNotOptimize(queries);
  auto tt = *nigiri::timetable::read(tt_path);
  tt.resolve();
  ::benchmark::DoNotOptimize(tt);
  nigiri::query_generation::generator_settings gs;
  ::benchmark::DoNotOptimize(gs);
  generate_queries(queries, 1U, tt, gs, 22);

  for (auto _ : state) {
    process_queries(queries, results, tt);
    ::benchmark::DoNotOptimize(results);
    ::benchmark::DoNotOptimize(queries);
    ::benchmark::DoNotOptimize(tt);
    ::benchmark::DoNotOptimize(gs);
  }
}
BENCHMARK(benchmark_random_queries)
    ->MinTime(20.0)
    ->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  if (argc <= 1) {
    fmt::println("Usage: ./nigiri-gbm [TIMETABLE PATH]");
    return 1;
  }
  tt_path = argv[1];
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}