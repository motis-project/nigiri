#include <algorithm>
#include <filesystem>
#include <iostream>
#include <regex>

#include "boost/program_options.hpp"

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

std::vector<std::string> tokenize(std::string const& str,
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

  auto const bbox_regex = std::regex{
      "^[-+]?[0-9]*\\.?[0-9]+,[-+]?[0-9]*\\.?[0-9]+,[-+]?[0-9]*\\.?[0-9]+,[-+]?"
      "[0-9]*\\.?[0-9]+$"};
  if (!std::regex_match(begin(str), end(str), bbox_regex)) {
    return std::nullopt;
  }
  auto const tokens = tokenize(str, ',', 4U);
  return box{latlng{std::stod(tokens[0]), std::stod(tokens[1])},
             latlng{std::stod(tokens[2]), std::stod(tokens[3])}};
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
  if (var.empty()) {
    std::cout << "Info: prints_stats for " << var_name
              << ": Input empty, no statistics to display\n";
    return;
  }
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
  namespace bpo = boost::program_options;
  auto const progress_tracker = utl::activate_progress_tracker("benchmark");
  utl::get_global_progress_trackers().silent_ = false;

  std::uint32_t num_queries;
  auto gs = query_generation::generator_settings{};

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", "produce this help message")
    ("tt_path,p", bpo::value<std::string>(),
            "path to a binary file containing a serialized nigiri timetable, "
            "can be created using nigiri-importer")
    ("seed,s", bpo::value<std::uint32_t>(),
            "value to seed the RNG of the query generator with, "
            "omit for random seed")
    ("num_queries,n",
            bpo::value<std::uint32_t>(&num_queries)->default_value(10000U),
            "number of queries to generate/process")
    ("interval_size,i", bpo::value<std::uint32_t>()->default_value(60U),
            "the initial size of the search interval in minutes")
    ("bounding_box,b", bpo::value<std::string>(),
            "limit randomized locations to a bounding box, "
            "format: lat_min,lon_min,lat_max,lon_max\ne.g., 36.0,-11.0,72.0,32.0\n"
            "(available via \"-b europe\")")
    ("start_mode", bpo::value<std::string>()->default_value("intermodal"),
            "intermodal | station")
    ("dest_mode", bpo::value<std::string>()->default_value("intermodal"),
            "intermodal | station")
    ("intermodal_start", bpo::value<std::string>()->default_value("walk"),
            "walk | bicycle | car")
    ("intermodal_dest", bpo::value<std::string>()->default_value("walk"),
            "walk | bicycle | car")
    ("use_start_footpaths",
            bpo::value<bool>(&gs.use_start_footpaths_)->default_value(true), "")
    ("max_transfers,t",
            bpo::value<std::uint8_t>(&gs.max_transfers_)->default_value(7U),
            "maximum number of transfers during routing")
    ("min_connection_count,m",
            bpo::value<std::uint32_t>(&gs.min_connection_count_)->default_value(3U),
            "the minimum number of connections to find with each query")
    ("extend_interval_earlier,e",
            bpo::value<bool>(&gs.extend_interval_earlier_)->default_value(true),
            "allows extension of the search interval into the past")
    ("extend_interval_later,l",
            bpo::value<bool>(&gs.extend_interval_later_)->default_value(true),
            "allows extension of the search interval into the future")
    ("prf_idx", bpo::value<profile_idx_t>(&gs.prf_idx_)->default_value(0U), "")
    ("allowed_claszes",
            bpo::value<clasz_mask_t>(&gs.allowed_claszes_)->default_value(routing::all_clasz_allowed()),
            "")
  ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);  // sets default values

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  std::unique_ptr<cista::wrapped<nigiri::timetable>> tt;
  if (vm.count("tt_path")) {
    try {
      auto const tt_path =
          std::filesystem::path{vm["tt_path"].as<std::string>()};
      auto load_timetable_timer = scoped_timer(
          fmt::format("loading timetable from {}", tt_path.string()));
      tt = std::make_unique<cista::wrapped<nigiri::timetable>>(
          nigiri::timetable::read(cista::memory_holder{
              cista::file{tt_path.c_str(), "r"}.content()}));
      (**tt).locations_.resolve_timezones();
    } catch (std::exception const& e) {
      log(nigiri::log_lvl::error, "benchmark.load",
          "can not read timetable file: {}", e.what());
      return 1;
    }
  } else {
    std::cout << "Error: path to timetable missing\n";
    return 1;
  }

  gs.interval_size_ = duration_t{vm["interval_size"].as<std::uint32_t>()};

  if (vm.count("bounding_box")) {
    gs.bbox_ = parse_bbox(vm["bounding_box"].as<std::string>());
    if (!gs.bbox_.has_value()) {
      std::cout << "Error: malformed bounding box input\n";
      return 1;
    }
  }

  if (vm["start_mode"].as<std::string>() == "intermodal") {
    gs.start_match_mode_ = location_match_mode::kIntermodal;
    if (vm.count("intermodal_start")) {
      if (vm["intermodal_start"].as<std::string>() == "walk") {
        gs.start_mode_ = query_generation::kWalk;
      } else if (vm["intermodal_start"].as<std::string>() == "bicycle") {
        gs.start_mode_ = query_generation::kBicycle;
      } else if (vm["intermodal_start"].as<std::string>() == "car") {
        gs.start_mode_ = query_generation::kCar;
      } else {
        std::cout << "Error: Unknown intermodal start mode\n";
        return 1;
      }
    }
  } else if (vm["start_mode"].as<std::string>() == "station") {
    gs.start_match_mode_ = location_match_mode::kExact;
  } else {
    std::cout << "Error: Invalid start mode\n";
    return 1;
  }

  if (vm["dest_mode"].as<std::string>() == "intermodal") {
    gs.dest_match_mode_ = location_match_mode::kIntermodal;
    if (vm.count("intermodal_dest")) {
      if (vm["intermodal_dest"].as<std::string>() == "walk") {
        gs.dest_mode_ = query_generation::kWalk;
      } else if (vm["intermodal_dest"].as<std::string>() == "bicycle") {
        gs.dest_mode_ = query_generation::kBicycle;
      } else if (vm["intermodal_dest"].as<std::string>() == "car") {
        gs.dest_mode_ = query_generation::kCar;
      } else {
        std::cout << "Error: Unknown intermodal start mode\n";
        return 1;
      }
    }
  } else if (vm["dest_mode"].as<std::string>() == "station") {
    gs.dest_match_mode_ = location_match_mode::kExact;
  } else {
    std::cout << "Error: Invalid destination mode\n";
    return 1;
  }

  std::mutex mutex;

  // generate queries
  auto queries = std::vector<query>{};
  {
    auto qg = vm.count("seed")
                  ? query_generation::generator{**tt, gs,
                                                vm["seed"].as<std::uint32_t>()}
                  : query_generation::generator{**tt, gs};

    auto query_generation_timer = scoped_timer(fmt::format(
        "generation of {} queries using seed {}", num_queries, qg.seed_));
    std::cout << "Query generator settings:\n" << gs << "\n";
    for (auto i = 0U; i != num_queries; ++i) {
      auto const q = qg.random_pretrip_query();
      if (q.has_value()) {
        queries.emplace_back(q.value());
      }
    }
  }

  std::cout << queries.size() << " queries generated successfully\n";

  // process queries
  auto results =
      std::vector<pair<std::uint64_t, routing_result<raptor_stats>>>{};
  {
    auto query_processing_timer =
        scoped_timer(fmt::format("processing of {} queries", queries.size()));
    progress_tracker->status("processing queries").in_high(queries.size());
    struct query_state {
      search_state ss_;
      raptor_state rs_;
    };
    utl::parallel_for_run_threadlocal<query_state>(
        queries.size(),
        [&](auto& query_state, auto const q_idx) {
          try {
            auto const result =
                routing::search<direction::kForward,
                                routing::raptor<direction::kForward, false>>{
                    **tt, nullptr, query_state.ss_, query_state.rs_,
                    queries[q_idx]}
                    .execute();
            std::lock_guard<std::mutex> guard(mutex);
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