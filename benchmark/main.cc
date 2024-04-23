#include <algorithm>
#include <filesystem>
#include <iostream>

#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/load.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/logging.h"
#include "nigiri/query_generator/generator.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

nigiri::pareto_set<nigiri::routing::journey> raptor_search(
    nigiri::timetable const& tt, nigiri::routing::query q) {
  using namespace nigiri;
  using algo_state_t = routing::raptor_state;
  static auto search_state = routing::search_state{};
  static auto algo_state = algo_state_t{};

  using algo_t = routing::raptor<nigiri::direction::kForward, false>;
  return *(routing::search<nigiri::direction::kForward, algo_t>{
      tt, nullptr, search_state, algo_state, std::move(q)}
               .execute()
               .journeys_);
}

std::unique_ptr<cista::wrapped<nigiri::timetable>> load_timetable(
    std::filesystem::path const& input_path) {
  auto load_timetable_timer = nigiri::scoped_timer("loading timetable");

  // gather paths of input files in target folder
  std::vector<std::filesystem::path> input_files;
  if (std::filesystem::is_regular_file(input_path) &&
      input_path.has_extension() && input_path.extension() == ".zip") {
    // input path directly to GTFS zip file
    input_files.emplace_back(input_path);
  } else if (std::filesystem::is_directory(input_path)) {
    // input path to directory
    for (auto const& dir_entry :
         std::filesystem::directory_iterator(input_path)) {
      std::filesystem::path file_path(dir_entry);
      if (std::filesystem::is_regular_file(dir_entry) &&
          file_path.has_extension() && file_path.extension() == ".zip") {
        input_files.emplace_back(file_path);
      }
    }
  } else {
    std::cout << "path provided is invalid\n";
  }

  auto const config = nigiri::loader::loader_config{100U, "Europe/Berlin"};
  auto const tt_interval = nigiri::interval<date::sys_days>{
      {date::January / 1 / 2024}, {date::December / 31 / 2024}};

  // hash tt settings and input files
  auto h = cista::hash_combine(
      cista::BASE_HASH, tt_interval.from_.time_since_epoch().count(),
      tt_interval.to_.time_since_epoch().count(), config.link_stop_distance_,
      cista::hash(config.default_tz_));
  for (auto const& input_file : input_files) {
    h = cista::hash_combine(h, cista::hash(input_file.string()));
  }

  auto const data_dir = std::filesystem::is_directory(input_files[0])
                            ? input_files[0]
                            : input_files[0].parent_path();
  auto const dump_file_path = data_dir / fmt::to_string(h);

  std::unique_ptr<cista::wrapped<nigiri::timetable>> tt;
  auto loaded = false;

  // try to load timetable from cache
  if (exists(dump_file_path)) {
    log(nigiri::log_lvl::info, "benchmark.load", "loading cached timetable {}",
        fmt::to_string(h));
    try {
      tt = std::make_unique<cista::wrapped<nigiri::timetable>>(
          nigiri::timetable::read(cista::memory_holder{
              cista::file{dump_file_path.c_str(), "r"}.content()}));
      (**tt).locations_.resolve_timezones();
      loaded = true;
    } catch (std::exception const& e) {
      log(nigiri::log_lvl::error, "benchmark.load",
          "can not read cached timetable image: {}", e.what());
    }
  }

  // load timetable from GTFS files
  if (!loaded) {
    log(nigiri::log_lvl::info, "benchmark.load",
        "no cached timetable found, loading from files");
    tt = std::make_unique<cista::wrapped<nigiri::timetable>>(
        cista::raw::make_unique<nigiri::timetable>(
            load(input_files, config, tt_interval)));
    // write cache file
    (**tt).write(dump_file_path);
  }

  return tt;
}

// needs sorted vector
template <typename T>
T quantile(std::vector<T> const& v, double q) {
  if (v.empty()) {
    return 0;
  }
  q = q < 0.0 ? 0.0 : q;
  q = 1.0 < q ? 1.0 : q;
  if (q == 1.0) {
    return v.back();
  }
  return v[static_cast<std::size_t>(v.size() * q)];
}

template <typename T>
void print_stats(std::vector<T> const& var, std::string const& var_name) {
  std::cout << "\n--- " << var_name << " --- (n = " << var.size() << ")"
            << "\n  25%: " << std::setw(12) << quantile(var, 0.25)
            << "\n  50%: " << std::setw(12) << quantile(var, 0.5)
            << "\n  75%: " << std::setw(12) << quantile(var, 0.75)
            << "\n  90%: " << std::setw(12) << quantile(var, 0.9)
            << "\n  99%: " << std::setw(12) << quantile(var, 0.99)
            << "\n99.9%: " << std::setw(12) << quantile(var, 0.999)
            << "\n  max: " << std::setw(12) << var.back()
            << "\n----------------------------------\n";
}

int main(int argc, char* argv[]) {
  using namespace nigiri;
  using namespace nigiri::routing;
  auto const progress_tracker = utl::activate_progress_tracker("benchmark");
  utl::get_global_progress_trackers().silent_ = false;

  if (argc != 2 && argc != 3) {
    std::cout
        << "usage: nigiri-benchmark GTFS-ZIP-FILE | DIRECTORY [seed]\nLoads a "
           "zip file containing a timetable in GTFS format, or attempts to "
           "load all zip files within a given directory.\nIf a seed is "
           "provided, it will be used to seed the random number generator\n";
    return 1;
  }

  auto tt = load_timetable({argv[1]});

  std::optional<std::uint32_t> seed;
  if (argc == 3) {
    seed = static_cast<std::uint32_t>(std::stol(argv[2]));
  }

  std::mutex queries_mutex;

  // generate queries
  auto queries = std::vector<query>{};
  {
    auto const num_queries = 10000U;
    auto const gs = query_generation::generator_settings{
        .interval_size_ = duration_t{60U},
        .start_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
        .dest_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
        .start_mode_ = query_generation::kWalk,
        .dest_mode_ = query_generation::kWalk,
        .use_start_footpaths_ = true,
        .max_transfers_ = routing::kMaxTransfers,
        .min_connection_count_ = 3U,
        .extend_interval_earlier_ = true,
        .extend_interval_later_ = true,
        .prf_idx_ = profile_idx_t{0},
        .allowed_claszes_ = clasz_mask_t{routing::all_clasz_allowed()}};
    auto qg = seed.has_value()
                  ? query_generation::generator{**tt, gs, seed.value()}
                  : query_generation::generator{**tt, gs};
    auto query_generation_timer = scoped_timer(fmt::format(
        "generation of {} queries using seed {}", num_queries, qg.seed_));
    progress_tracker->status("generating queries").in_high(queries.size());
    utl::parallel_for_run(
        num_queries,
        [&](auto const i) {
          auto const q = qg.random_pretrip_query();
          if (q.has_value()) {
            std::lock_guard<std::mutex> guard(queries_mutex);
            queries.emplace_back(q.value());
          }
        },
        progress_tracker->update_fn());
  }

  // process queries
  auto results =
      std::vector<pair<std::uint64_t, routing_result<raptor_stats>>>{};
  {
    auto query_processing_timer =
        scoped_timer(fmt::format("processing of {} queries", queries.size()));
    progress_tracker->status("processing queries").in_high(queries.size());
    utl::parallel_for_run(
        queries.size(),
        [&](auto const q_idx) {
          auto ss = search_state{};
          auto rs = raptor_state{};

          try {
            auto const result =
                routing::search<direction::kForward,
                                routing::raptor<direction::kForward, false>>{
                    **tt, nullptr, ss, rs, queries[q_idx]}
                    .execute();
            std::lock_guard<std::mutex> guard(queries_mutex);
            results.emplace_back(q_idx, result);
          } catch (const std::exception& e) {
            std::cout << e.what();
          }
        },
        progress_tracker->update_fn());
  }

  // print results
  auto routing_times = std::vector<std::chrono::milliseconds::rep>{};
  routing_times.reserve(results.size());
  auto search_iterations = std::vector<std::uint64_t>{};
  search_iterations.reserve(results.size());
  for (auto const& result : results) {
    routing_times.emplace_back(
        result.second.search_stats_.execute_time_.count());
    search_iterations.emplace_back(
        result.second.search_stats_.search_iterations_);
  }
  std::sort(begin(routing_times), end(routing_times));
  print_stats(routing_times, "routing times [ms]");
  std::sort(begin(search_iterations), end(search_iterations));
  print_stats(search_iterations, "search iterations");

  return 0;
}