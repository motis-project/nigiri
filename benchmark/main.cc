#include <algorithm>
#include <filesystem>
#include <iostream>
#include <regex>

#include "boost/program_options.hpp"

#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"

#include "nigiri/logging.h"
#include "nigiri/query_generator/generator.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

using namespace nigiri;
using namespace nigiri::routing;

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

std::optional<geo::latlng> parse_coord(std::string const& str) {
  using namespace geo;

  auto const coord_regex =
      std::regex{"^[-+]?[0-9]*\\.?[0-9]+,[-+]?[0-9]*\\.?[0-9]+"};
  if (!std::regex_match(begin(str), end(str), coord_regex)) {
    return std::nullopt;
  }
  auto const tokens = tokenize(str, ',', 2U);
  return latlng{std::stod(tokens[0]), std::stod(tokens[1])};
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

struct benchmark_result {
  friend std::ostream& operator<<(std::ostream& out,
                                  benchmark_result const& br) {
    out << "(t_total: " << std::fixed << std::setprecision(3) << std::setw(8)
        << std::chrono::duration_cast<
               std::chrono::duration<double, std::ratio<1>>>(br.total_time_)
               .count()
        << ", t_exec: " << std::setw(8)
        << std::chrono::duration_cast<
               std::chrono::duration<double, std::ratio<1>>>(
               br.routing_result_.search_stats_.execute_time_)
               .count()
        << ", intvl_ext: " << std::setw(2)
        << br.routing_result_.search_stats_.interval_extensions_
        << ", intvl_size: " << std::setw(3) << std::defaultfloat
        << std::chrono::duration_cast<
               std::chrono::duration<std::uint32_t, std::ratio<3600>>>(
               br.routing_result_.interval_.size())
               .count()
        << ", #jrny: " << std::setfill(' ') << std::setw(2)
        << br.journeys_.size() << ")";
    return out;
  }

  std::uint64_t q_idx_;
  routing_result<raptor_stats> routing_result_;
  pareto_set<journey> journeys_;
  std::chrono::milliseconds total_time_;
};

void print_results(std::vector<benchmark_result> const& var,
                   std::string const& var_name) {
  std::cout << "\n--- " << var_name << " --- (n = " << var.size() << ")"
            << "\n  25%: " << quantile(var, 0.25)
            << "\n  50%: " << quantile(var, 0.5)
            << "\n  75%: " << quantile(var, 0.75)
            << "\n  90%: " << quantile(var, 0.9)
            << "\n  99%: " << quantile(var, 0.99)
            << "\n99.9%: " << quantile(var, 0.999) << "\n  max: " << var.back()
            << "\n----------------------------------\n";
}

int main(int argc, char* argv[]) {
  namespace bpo = boost::program_options;
  auto const progress_tracker = utl::activate_progress_tracker("benchmark");
  utl::get_global_progress_trackers().silent_ = false;

  auto tt_path = std::filesystem::path{};
  auto num_queries = std::uint32_t{10000U};
  auto gs = query_generation::generator_settings{};

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", "produce this help message")
    ("tt_path,p", bpo::value(&tt_path),
            "path to a binary file containing a serialized nigiri timetable")
    ("seed,s", bpo::value<std::uint32_t>(),
            "value to seed the RNG of the query generator with, "
            "omit for random seed")
    ("num_queries,n",
            bpo::value(&num_queries)->default_value(num_queries),
            "number of queries to generate/process")
    ("interval_size,i", bpo::value<std::uint32_t>()->default_value(60U),
            "the initial size of the search interval in minutes, set to 0 for ontrip queries")
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
    ("start_coord", bpo::value<std::string>(),
            "start coordinate for random queries")
    ("dest_coord", bpo::value<std::string>(),
            "destination coordinate for random queries")
    ("start_loc", bpo::value<location_idx_t::value_t>(),
            "start location for random queries")
    ("dest_loc", bpo::value<location_idx_t::value_t>(),
        "destination location for random queries")
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
    gs.start_match_mode_ = location_match_mode::kEquivalent;
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
    gs.dest_match_mode_ = location_match_mode::kEquivalent;
  } else {
    std::cout << "Error: Invalid destination mode\n";
    return 1;
  }

  if (vm.count("start_coord")) {
    gs.start_match_mode_ = location_match_mode::kIntermodal;
    auto const start_coord = parse_coord(vm["start_coord"].as<std::string>());
    if (start_coord.has_value()) {
      gs.start_ = start_coord.value();
    } else {
      std::cout << "Error: Invalid start coordinate\n";
      return 1;
    }
  }

  if (vm.count("dest_coord")) {
    gs.dest_match_mode_ = location_match_mode::kIntermodal;
    auto const dest_coord = parse_coord(vm["dest_coord"].as<std::string>());
    if (dest_coord.has_value()) {
      gs.dest_ = dest_coord.value();
    } else {
      std::cout << "Error: Invalid destination coordinate\n";
      return 1;
    }
  }

  if (vm.count("start_loc")) {
    gs.start_match_mode_ = location_match_mode::kEquivalent;
    gs.start_ = vm["start_loc"].as<location_idx_t>();
  }

  if (vm.count("dest_loc")) {
    gs.dest_match_mode_ = location_match_mode::kEquivalent;
    gs.dest_ = vm["dest_loc"].as<location_idx_t>();
  }

  std::mutex mutex;

  // generate queries
  auto queries = std::vector<nigiri::query_generation::start_dest_query>{};
  {
    auto qg = vm.count("seed")
                  ? query_generation::generator{**tt, gs,
                                                vm["seed"].as<std::uint32_t>()}
                  : query_generation::generator{**tt, gs};

    auto query_generation_timer = scoped_timer(fmt::format(
        "generation of {} queries using seed {}", num_queries, qg.seed_));
    std::cout << "--- Query generator settings ---\n"
              << gs << "\n--- --- ---\n";
    for (auto i = 0U; i != num_queries; ++i) {
      auto const sdq = qg.random_query();
      if (sdq.has_value()) {
        queries.emplace_back(sdq.value());
      }
    }
  }

  std::cout << queries.size() << " queries generated successfully\n";

  // process queries
  auto results = std::vector<benchmark_result>{};
  results.reserve(queries.size());
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
            auto const total_time_start = std::chrono::steady_clock::now();
            auto const result =
                routing::search<direction::kForward,
                                routing::raptor<direction::kForward, false>>{
                    **tt, nullptr, query_state.ss_, query_state.rs_,
                    queries[q_idx].q_}
                    .execute();
            auto const total_time_stop = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> guard(mutex);
            results.emplace_back(
                q_idx, result, *result.journeys_,
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    total_time_stop - total_time_start));
          } catch (const std::exception& e) {
            std::cout << e.what();
          }
        },
        progress_tracker->update_fn());
  }

  // print results

  std::sort(begin(results), end(results), [](auto const& a, auto const& b) {
    return a.total_time_ < b.total_time_;
  });
  print_results(results, "total_time [s]");

  auto const print_slow_result = [&](auto const& br) {
    auto const visit_loc_idx = [&](location_idx_t const loc_idx) {
      std::stringstream ss;
      ss << "(loc_idx: " << loc_idx.v_ << ", name: "
         << std::string_view{begin((**tt).locations_.names_[loc_idx]),
                             end((**tt).locations_.names_[loc_idx])}
         << ", coord: (" << (**tt).locations_.coordinates_[loc_idx].lat() << ","
         << (**tt).locations_.coordinates_[loc_idx].lng() << "))";
      return ss.str();
    };
    auto const visit_coord = [](geo::latlng const& coord) {
      std::stringstream ss;
      ss << "(" << coord.lat() << "," << coord.lng() << ")";
      return ss.str();
    };

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
    std::cout << "\n--- " << i + 1 << " ---\n";
    print_slow_result(rbegin(results)[i]);
  }
  std::cout << "\n";

  std::sort(begin(results), end(results), [](auto const& a, auto const& b) {
    return a.routing_result_.search_stats_.execute_time_ <
           b.routing_result_.search_stats_.execute_time_;
  });
  print_results(results, "execute_time [s]");

  std::sort(begin(results), end(results), [](auto const& a, auto const& b) {
    return a.routing_result_.search_stats_.interval_extensions_ <
           b.routing_result_.search_stats_.interval_extensions_;
  });
  print_results(results, "interval_extensions");

  std::sort(begin(results), end(results), [](auto const& a, auto const& b) {
    return a.routing_result_.interval_.size() <
           b.routing_result_.interval_.size();
  });
  print_results(results, "interval_size [h]");

  std::sort(begin(results), end(results), [](auto const& a, auto const& b) {
    return a.journeys_.size() < b.journeys_.size();
  });
  print_results(results, "#journeys");

  return 0;
}
