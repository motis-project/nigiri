#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>

#include "boost/program_options.hpp"

#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"

#include "nigiri/query_generator/generator.h"
#include "nigiri/routing/meat/csa/meat_csa_state.h"
#include "nigiri/routing/meat/decision_graph.h"
#include "nigiri/routing/meat/expanded_representation.h"
#include "nigiri/routing/meat/raptor/meat_raptor_state.h"
#include "nigiri/routing/meat_csa_search.h"
#include "nigiri/routing/meat_raptor_search.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"
// #include "nigiri/logging.h"

#ifndef _WIN32
#include <sys/resource.h>
#endif

using namespace nigiri;
using namespace nigiri::routing;

namespace m = routing::meat;
namespace mcsa = m::csa;
namespace mraptor = m::raptor;

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

template <typename MeatStats>
struct benchmark_result {
  std::uint64_t q_idx_;
  MeatStats stats_;
  m::decision_graph g_;
  std::chrono::milliseconds total_time_;
};
std::ostream& operator<<(std::ostream& out,
                         benchmark_result<mcsa::meat_csa_stats> const& br) {
  using double_seconds_t = std::chrono::duration<double, std::ratio<1>>;
  out << "(t_total: " << std::fixed << std::setprecision(3) << std::setw(9)
      << std::chrono::duration_cast<double_seconds_t>(br.total_time_).count()
      << "s, t_exec: " << std::setw(9) << br.stats_.total_duration_
      << "ms, t_esa: " << std::setw(9) << br.stats_.esa_duration_
      << "ms, t_ea: " << std::setw(9) << br.stats_.ea_duration_
      << "ms, t_meat: " << std::setw(9) << br.stats_.meat_duration_
      << "ms, t_ext_g: " << std::setw(9) << br.stats_.extract_graph_duration_
      << "ms, esa_#con: " << std::setw(9)
      << br.stats_.esa_n_connections_scanned_ << ", ea_#con: " << std::setw(9)
      << br.stats_.ea_n_connections_scanned_ << ", meat_#con: " << std::setw(9)
      << br.stats_.meat_n_connections_scanned_
      << ", #fp_to_que: " << std::setw(9) << br.stats_.meat_n_fp_added_to_que_
      << ", #fp_to_prof: " << std::setw(9)
      << br.stats_.meat_n_fp_added_to_profile_
      << ", #con_to_prof: " << std::setw(9)
      << br.stats_.meat_n_e_added_or_replaced_to_profile_
      << ", #e_in_prof: " << std::setw(9) << br.stats_.meat_n_e_in_profile_
      << ", q_idx: " << br.q_idx_ << ")";
  return out;
}
std::ostream& operator<<(
    std::ostream& out, benchmark_result<mraptor::meat_raptor_stats> const& br) {
  using double_seconds_t = std::chrono::duration<double, std::ratio<1>>;
  out << "(t_total: " << std::fixed << std::setprecision(3) << std::setw(9)
      << std::chrono::duration_cast<double_seconds_t>(br.total_time_).count()
      << "s, t_exec: " << std::setw(9) << br.stats_.total_duration_
      << "ms, t_esa: " << std::setw(9) << br.stats_.esa_duration_
      << "ms, t_ea: " << std::setw(9) << br.stats_.ea_duration_
      << "ms, t_meat: " << std::setw(9) << br.stats_.meat_duration_
      << "ms, t_ext_g: " << std::setw(9) << br.stats_.extract_graph_duration_
      << "ms, meat_#tran_it: " << std::setw(9)
      << br.stats_.meat_n_active_transports_iterated_
      << ", meat_#stops_it: " << std::setw(9)
      << br.stats_.meat_n_stops_iterated_ << ", #fp_to_prof: " << std::setw(9)
      << br.stats_.meat_n_fp_added_to_profile_
      << ", #con_to_prof: " << std::setw(9)
      << br.stats_.meat_n_e_added_to_profile_
      << ", #e_in_prof: " << std::setw(9) << br.stats_.meat_n_e_in_profile_
      << ", q_idx: " << br.q_idx_ << ")";
  return out;
}

void process_queries(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    std::vector<benchmark_result<mcsa::meat_csa_stats>>& results,
    nigiri::timetable const& tt,
    bool parallel) {
  results.reserve(queries.size());
  std::mutex mutex;
  {
    auto query_processing_timer = scoped_timer(
        fmt::format("processing of {} queries (MEAT CSA)", queries.size()));
    auto const progress_tracker =
        utl::activate_progress_tracker("benchmark MEAT CSA");
    utl::get_global_progress_trackers().silent_ = false;
    progress_tracker->status("processing queries").in_high(queries.size());
    if (parallel) {
      struct query_state {
        mcsa::meat_csa_state<mcsa::dynamic_profile_set> ms_;
      };
      utl::parallel_for_run_threadlocal<query_state>(
          queries.size(), [&](auto& query_state, auto const q_idx) {
            try {
              auto const total_time_start = std::chrono::steady_clock::now();
              auto const result = routing::meat_csa_search(tt, query_state.ms_,
                                                           queries[q_idx].q_);
              auto const total_time_stop = std::chrono::steady_clock::now();
              auto const guard = std::lock_guard{mutex};
              results.emplace_back(benchmark_result{
                  q_idx, result.stats_, result.g_,
                  std::chrono::duration_cast<std::chrono::milliseconds>(
                      total_time_stop - total_time_start)});
              progress_tracker->increment();
            } catch (const std::exception& e) {
              std::cout << e.what();
            }
          });
    } else {
      mcsa::meat_csa_state<mcsa::dynamic_profile_set> ms;
      for (auto const [q_idx, q] : utl::enumerate(queries)) {
        try {
          auto const total_time_start = std::chrono::steady_clock::now();
          auto const result = routing::meat_csa_search(tt, ms, q.q_);
          auto const total_time_stop = std::chrono::steady_clock::now();
          results.emplace_back(benchmark_result{
              q_idx, result.stats_, result.g_,
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  total_time_stop - total_time_start)});
          progress_tracker->increment();
        } catch (const std::exception& e) {
          std::cout << e.what();
        }
      }
    }
  }
}

void process_queries(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    std::vector<benchmark_result<mraptor::meat_raptor_stats>>& results,
    nigiri::timetable const& tt,
    bool parallel) {
  results.reserve(queries.size());
  std::mutex mutex;
  {
    auto query_processing_timer = scoped_timer(
        fmt::format("processing of {} queries (MEAT RAPTOR)", queries.size()));
    auto const progress_tracker =
        utl::activate_progress_tracker("benchmark MEAT RAPTOR");
    utl::get_global_progress_trackers().silent_ = false;
    progress_tracker->status("processing queries").in_high(queries.size());
    if (parallel) {
      struct query_state {
        mraptor::meat_raptor_state ms_;
      };
      utl::parallel_for_run_threadlocal<query_state>(
          queries.size(), [&](auto& query_state, auto const q_idx) {
            try {
              auto const total_time_start = std::chrono::steady_clock::now();
              auto const result = routing::meat_raptor_search(
                  tt, query_state.ms_, queries[q_idx].q_);
              auto const total_time_stop = std::chrono::steady_clock::now();
              auto const guard = std::lock_guard{mutex};
              results.emplace_back(benchmark_result{
                  q_idx, result.stats_, result.g_,
                  std::chrono::duration_cast<std::chrono::milliseconds>(
                      total_time_stop - total_time_start)});
              progress_tracker->increment();
            } catch (const std::exception& e) {
              std::cout << e.what();
            }
          });
    } else {
      mraptor::meat_raptor_state ms;
      for (auto const [q_idx, q] : utl::enumerate(queries)) {
        try {
          auto const total_time_start = std::chrono::steady_clock::now();
          auto const result = routing::meat_raptor_search(tt, ms, q.q_);
          auto const total_time_stop = std::chrono::steady_clock::now();
          results.emplace_back(benchmark_result{
              q_idx, result.stats_, result.g_,
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  total_time_stop - total_time_start)});
          progress_tracker->increment();
        } catch (const std::exception& e) {
          std::cout << e.what();
        }
      }
    }
  }
}

void process_queries(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    std::vector<benchmark_result<mcsa::meat_csa_stats>>& results_csa,
    std::vector<benchmark_result<mraptor::meat_raptor_stats>>& results_raptor,
    nigiri::timetable const& tt,
    bool parallel) {
  results_csa.reserve(queries.size());
  results_raptor.reserve(queries.size());
  std::mutex mutex;
  {
    auto query_processing_timer = scoped_timer(fmt::format(
        "processing of {} queries (MEAT CSA & RAPTOR)", queries.size()));
    auto const progress_tracker =
        utl::activate_progress_tracker("benchmark MEAT CSA & RAPTOR");
    utl::get_global_progress_trackers().silent_ = false;
    progress_tracker->status("processing queries").in_high(queries.size());
    if (parallel) {
      struct query_state {
        mcsa::meat_csa_state<mcsa::dynamic_profile_set> ms_c_;
        mraptor::meat_raptor_state ms_r_;
      };
      utl::parallel_for_run_threadlocal<query_state>(
          queries.size(), [&](auto& query_state, auto const q_idx) {
            try {
              auto const total_time_c_start = std::chrono::steady_clock::now();
              auto const result_c = routing::meat_csa_search(
                  tt, query_state.ms_c_, queries[q_idx].q_);
              auto const total_time_c_stop = std::chrono::steady_clock::now();

              auto const total_time_r_start = std::chrono::steady_clock::now();
              auto const result_r = routing::meat_raptor_search(
                  tt, query_state.ms_r_, queries[q_idx].q_);
              auto const total_time_r_stop = std::chrono::steady_clock::now();

              auto const guard = std::lock_guard{mutex};
              results_csa.emplace_back(benchmark_result{
                  q_idx, result_c.stats_, result_c.g_,
                  std::chrono::duration_cast<std::chrono::milliseconds>(
                      total_time_c_stop - total_time_c_start)});
              results_raptor.emplace_back(benchmark_result{
                  q_idx, result_r.stats_, result_r.g_,
                  std::chrono::duration_cast<std::chrono::milliseconds>(
                      total_time_r_stop - total_time_r_start)});

              progress_tracker->increment();
            } catch (const std::exception& e) {
              std::cout << e.what();
            }
          });
    } else {
      mcsa::meat_csa_state<mcsa::dynamic_profile_set> ms_c;
      mraptor::meat_raptor_state ms_r;
      for (auto const [q_idx, q] : utl::enumerate(queries)) {
        try {
          //// TODO remove
          // if (q_idx != 26) continue;
          // if (q_idx != 705) continue;
          //  if (q_idx == 63 || q_idx == 94){
          //   std::cout << q_idx << std::endl;
          // }
          //// remove end

          auto const total_time_c_start = std::chrono::steady_clock::now();
          auto const result_c = routing::meat_csa_search(tt, ms_c, q.q_);
          auto const total_time_c_stop = std::chrono::steady_clock::now();
          results_csa.emplace_back(benchmark_result{
              q_idx, result_c.stats_, result_c.g_,
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  total_time_c_stop - total_time_c_start)});

          auto const total_time_r_start = std::chrono::steady_clock::now();
          auto const result_r = routing::meat_raptor_search(tt, ms_r, q.q_);
          auto const total_time_r_stop = std::chrono::steady_clock::now();
          results_raptor.emplace_back(benchmark_result{
              q_idx, result_r.stats_, result_r.g_,
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  total_time_r_stop - total_time_r_start)});

          if (((result_c.g_.first_arc_ == dg_arc_idx_t::invalid() ||
                result_r.g_.first_arc_ == dg_arc_idx_t::invalid()) &&
               result_c.g_.first_arc_ != result_r.g_.first_arc_) ||
              (result_c.g_.first_arc_ != dg_arc_idx_t::invalid() &&
               result_r.g_.first_arc_ != dg_arc_idx_t::invalid() &&
               result_c.g_.arcs_[result_c.g_.first_arc_].meat_ !=
                   result_r.g_.arcs_[result_r.g_.first_arc_].meat_)) {
            std::cout << std::endl
                      << "!!!!!!!!!!!! csa.meat_ != raptor.meat_ !!!!!!!!!!!!"
                      << std::endl
                      << "q_idx: " << q_idx << std::endl
                      << std::endl
                      << std::endl;
          }

          //// TODO remove
          // auto er_c = m::expanded_representation{result_c.g_};
          // auto er_r = m::expanded_representation{result_r.g_};
          // auto ss_c = std::stringstream{};
          // auto ss_r = std::stringstream{};
          // m::write_dot(ss_c, tt, result_c.g_, er_c);
          // m::write_dot(ss_r, tt, result_r.g_, er_r);
          //
          // std::cout << std::endl
          //           << ss_c.str() << std::endl
          //           << ss_r.str() << std::endl;
          //
          // if (ss_c.str() != ss_r.str() /*result_c.g_ != result_r.g_*/) {
          //   std::cout << std::endl
          //             << (result_c.g_.arcs_[result_c.g_.first_arc_].meat_ ==
          //                 result_r.g_.arcs_[result_r.g_.first_arc_].meat_)
          //             << std::endl
          //             << (!semantically_equal(result_c.g_, result_r.g_))
          //             << std::endl
          //             << ss_c.str() << std::endl
          //             << ss_r.str() << std::endl
          //             << result_c.g_ << std::endl
          //             << result_r.g_ << std::endl;
          // }
          // if (!semantically_equal(result_c.g_, result_r.g_)) {
          //   std::cout << std::endl
          //             << ss_c.str() << std::endl
          //             << ss_r.str() << std::endl
          //             << result_c.g_ << std::endl
          //             << result_r.g_ << std::endl;
          // }

          progress_tracker->increment();
        } catch (const std::exception& e) {
          std::cout << e.what();
        }
      }
    }
  }
}

// needs sorted vector
template <typename T>
T const& quantile(std::vector<T> const& v, double q) {
  q = q < 0.0 ? 0.0 : q;
  q = 1.0 < q ? 1.0 : q;
  if (q == 1.0) {
    return v.back();
  }
  return v[static_cast<std::size_t>(v.size() * q)];
}

template <typename T>
void print_result(std::vector<T> const& var,
                  std::string const& var_name,
                  std::ostream& os = std::cout) {
  os << "\n--- " << var_name << " --- (n = " << var.size() << ")"
     << "\n  10%: " << quantile(var, 0.1) << "\n  20%: " << quantile(var, 0.2)
     << "\n  30%: " << quantile(var, 0.3) << "\n  40%: " << quantile(var, 0.4)
     << "\n  50%: " << quantile(var, 0.5) << "\n  60%: " << quantile(var, 0.6)
     << "\n  70%: " << quantile(var, 0.7) << "\n  80%: " << quantile(var, 0.8)
     << "\n  90%: " << quantile(var, 0.9) << "\n  99%: " << quantile(var, 0.99)
     << "\n99.9%: " << quantile(var, 0.999) << "\n  max: " << var.back()
     << "\n----------------------------------\n";
}

void print_query_rerun(
    size_t q_idx,
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    nigiri::query_generation::generator_settings const& gs,
    std::filesystem::path const& tt_path,
    std::ostream& os = std::cout) {
  auto ss = std::stringstream{};
  ss << "Re-run query " << q_idx << ":\n./nigiri-benchmark -p "
     << tt_path.string() << " -n 1 -i " << gs.interval_size_.count()
     << " --max_delay " << gs.max_delay_ << " --bound_parameter "
     << gs.bound_parameter_;
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
    ss << " --start_coord \"" << get<geo::latlng>(queries[q_idx].start_)
       << "\"";
  } else {
    ss << " --start_loc " << get<location_idx_t>(queries[q_idx].start_);
  }
  if (gs.dest_match_mode_ == location_match_mode::kIntermodal) {
    ss << " --dest_coord \"" << get<geo::latlng>(queries[q_idx].dest_) << "\"";
  } else {
    ss << " --dest_loc " << get<location_idx_t>(queries[q_idx].dest_) << "\n";
  }
  os << ss.str() << "\n";
}

template <typename MeatStats>
void print_results(
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    std::vector<benchmark_result<MeatStats>>& results,
    nigiri::timetable const& tt,
    nigiri::query_generation::generator_settings const& gs,
    std::filesystem::path const& tt_path,
    std::ostream& os = std::cout) {
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
    ss << "loc_idx: " << loc_idx.v_ << ", name: "
       << std::string_view{begin(tt.locations_.names_[loc_idx]),
                           end(tt.locations_.names_[loc_idx])}
       << ", coord: " << tt.locations_.coordinates_[loc_idx];
    return ss.str();
  };

  auto const print_slow_result = [&](auto const& br) {
    os << br << "\nstart: "
       << std::visit(utl::overloaded{visit_loc_idx, visit_coord},
                     queries[br.q_idx_].start_)
       << "\ndest: "
       << std::visit(utl::overloaded{visit_loc_idx, visit_coord},
                     queries[br.q_idx_].dest_)
       << "\n";
  };
  os << "\nSlowest Queries:\n";
  for (auto i = 0; i != results.size() && i != 10; ++i) {
    os << "\n--- " << i + 1 << " ---\nquery_idx: " << rbegin(results)[i].q_idx_
       << '\n';
    print_slow_result(rbegin(results)[i]);
  }
  os << "\n";

  auto ss = std::stringstream{};
  ss << "Re-run the slowest source-destination "
        "combination:\n./nigiri-benchmark -p "
     << tt_path.string() << " -n 1 -i " << gs.interval_size_.count()
     << " --max_delay " << gs.max_delay_ << " --bound_parameter "
     << gs.bound_parameter_;
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
  os << ss.str() << "\n";

  utl::sort(results, [](auto const& a, auto const& b) {
    return a.stats_.total_duration_ < b.stats_.total_duration_;
  });
  print_result(results, "execute_time");

  utl::sort(results, [](auto const& a, auto const& b) {
    return a.stats_.esa_duration_ < b.stats_.esa_duration_;
  });
  print_result(results, "esa_duration_");

  utl::sort(results, [](auto const& a, auto const& b) {
    return a.stats_.ea_duration_ < b.stats_.ea_duration_;
  });
  print_result(results, "ea_duration_");

  utl::sort(results, [](auto const& a, auto const& b) {
    return a.stats_.meat_duration_ < b.stats_.meat_duration_;
  });
  print_result(results, "meat_duration_");

  utl::sort(results, [](auto const& a, auto const& b) {
    return a.stats_.extract_graph_duration_ < b.stats_.extract_graph_duration_;
  });
  print_result(results, "extract_graph_duration_");
}

void print_all_results(
    std::ostream& out,
    std::vector<benchmark_result<mcsa::meat_csa_stats>>& results) {
  using double_seconds_t = std::chrono::duration<double, std::ratio<1>>;
  out << "t_total,t_exec,t_esa,t_ea,t_meat,t_ext_g,esa_num_con,"
         "ea_num_con,"
         "meat_num_con,num_fp_to_que,num_fp_to_prof,num_con_to_prof,"
         "num_e_in_prof,"
         "q_idx\n";
  for (auto const& br : results) {
    out << std::fixed << std::setprecision(3)
        << std::chrono::duration_cast<double_seconds_t>(br.total_time_).count()
        << "," << br.stats_.total_duration_ << "," << br.stats_.esa_duration_
        << "," << br.stats_.ea_duration_ << "," << br.stats_.meat_duration_
        << "," << br.stats_.extract_graph_duration_ << ","
        << br.stats_.esa_n_connections_scanned_ << ","
        << br.stats_.ea_n_connections_scanned_ << ","
        << br.stats_.meat_n_connections_scanned_ << ","
        << br.stats_.meat_n_fp_added_to_que_ << ","
        << br.stats_.meat_n_fp_added_to_profile_ << ","
        << br.stats_.meat_n_e_added_or_replaced_to_profile_ << ","
        << br.stats_.meat_n_e_in_profile_ << "," << br.q_idx_ << "\n";
  }
}

void print_all_results(
    std::ostream& out,
    std::vector<benchmark_result<mraptor::meat_raptor_stats>>& results) {
  using double_seconds_t = std::chrono::duration<double, std::ratio<1>>;
  out << "t_total,t_exec,t_esa,t_ea,t_meat,t_ext_g,meat_num_tran_it,"
         "meat_num_stops_it,num_fp_to_prof,num_con_to_prof,num_e_in_prof,"
         "q_idx\n";
  for (auto const& br : results) {
    out << std::fixed << std::setprecision(3)
        << std::chrono::duration_cast<double_seconds_t>(br.total_time_).count()
        << "," << br.stats_.total_duration_ << "," << br.stats_.esa_duration_
        << "," << br.stats_.ea_duration_ << "," << br.stats_.meat_duration_
        << "," << br.stats_.extract_graph_duration_ << ","
        << br.stats_.meat_n_active_transports_iterated_ << ","
        << br.stats_.meat_n_stops_iterated_ << ","
        << br.stats_.meat_n_fp_added_to_profile_ << ","
        << br.stats_.meat_n_e_added_to_profile_ << ","
        << br.stats_.meat_n_e_in_profile_ << "," << br.q_idx_ << "\n";
  }
}

void print_memory_usage(std::ostream& os = std::cout) {
#ifndef _WIN32
  auto r = rusage{};
  getrusage(RUSAGE_SELF, &r);
  os << "\n--- memory usage ---\nrusage.ru_maxrss: "
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
  auto prf_idx = std::uint32_t{0};
  auto start_coord_str = std::string{};
  auto dest_coord_str = std::string{};
  auto start_loc_val = location_idx_t::value_t{0U};
  auto dest_loc_val = location_idx_t::value_t{0U};
  auto seed = std::int64_t{-1};
  auto min_transfer_time = duration_t::rep{};
  auto max_delay = delta_t{30};
  auto bound_parameter = 1.0;
  auto out_file_path_prefix = std::filesystem::path{"results"};

  bpo::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce this help message")  //
      ("tt_path,p", bpo::value(&tt_path)->required(),
       "path to a binary file containing a serialized nigiri timetable")  //
      ("seed,s", bpo::value<std::int64_t>(&seed),
       "value to seed the RNG of the query generator with, "
       "omit for random seed")  //
      ("num_queries,n", bpo::value(&n_queries)->default_value(n_queries),
       "number of queries to generate/process")(
          "interval_size,i",
          bpo::value<duration_t::rep>(&interval_size)->default_value(0U, "0"),
          "the initial size of the search interval in minutes, set to 0 for "
          "ontrip queries")  //
      ("bounding_box,b", bpo::value<std::string>(&bbox_str),
       "limit randomized locations to a bounding box, "
       "format: lat_min,lon_min,lat_max,lon_max\ne.g., 36.0,-11.0,72.0,32.0\n"
       "(available via \"-b europe\")")  //
      ("start_mode",
       bpo::value<std::string>(&start_mode_str)->default_value("station"),
       "intermodal | station")  //
      ("dest_mode",
       bpo::value<std::string>(&dest_mode_str)->default_value("station"),
       "intermodal | station")  //
      ("intermodal_start",
       bpo::value<std::string>(&intermodal_start_str)->default_value("walk"),
       "walk | bicycle | car")  //
      ("intermodal_dest",
       bpo::value<std::string>(&intermodal_dest_str)->default_value("walk"),
       "walk | bicycle | car")  //
      ("use_start_footpaths",
       bpo::value<bool>(&gs.use_start_footpaths_)->default_value(true),
       "")("max_transfers,t",
           bpo::value<std::uint32_t>(&max_transfers)
               ->default_value(kMaxTransfers),
           "maximum number of transfers during routing")  //
      ("min_connection_count,m",
       bpo::value<std::uint32_t>(&gs.min_connection_count_)->default_value(3U),
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
      ("max_delay", bpo::value<delta_t>(&max_delay)->default_value(max_delay),
       "maximum delay (in minutes) of a train (for meat-csa and "
       "meat-raptor)")  //
      ("bound_parameter",
       bpo::value<double>(&bound_parameter)->default_value(bound_parameter),
       "aka Alpha value (for meat-csa and meat-raptor)")  //
      ("out_file",
       bpo::value<std::filesystem::path>(&out_file_path_prefix)
           ->default_value(out_file_path_prefix),
       "path to the output files prefix to write the results to");
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);

  // process program options - begin
  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  bpo::notify(vm);

  std::cout << "loading timetable...\n";
  auto tt = *nigiri::timetable::read(cista::memory_holder{
      cista::file{tt_path.generic_string().c_str(), "r"}.content()});
  tt.locations_.resolve_timezones();

  gs.interval_size_ = duration_t{interval_size};
  gs.max_delay_ = max_delay;
  gs.bound_parameter_ = bound_parameter;

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

  gs.max_transfers_ = max_transfers > std::numeric_limits<std::uint8_t>::max()
                          ? std::numeric_limits<std::uint8_t>::max()
                          : max_transfers;

  gs.transfer_time_settings_.min_transfer_time_ = duration_t{min_transfer_time};
  gs.transfer_time_settings_.default_ =
      min_transfer_time == 0U && gs.transfer_time_settings_.factor_ == 1.0F;

  if (vm.count("profile_idx") != 0) {
    if (prf_idx >= kMaxProfiles) {
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

  auto queries = std::vector<nigiri::query_generation::start_dest_query>{};
  generate_queries(queries, n_queries, tt, gs, seed);

  // print_query_rerun(4, queries,gs,tt_path);

  auto results_csa = std::vector<benchmark_result<mcsa::meat_csa_stats>>{};
  auto results_raptor =
      std::vector<benchmark_result<mraptor::meat_raptor_stats>>{};
  process_queries(queries, results_csa, results_raptor, tt, false);

  std::cout << "\n ----------- \nNumber of CSA connections: "
            << tt.fwd_connections_.size();
  std::cout << "\nNumber of transports: " << tt.transport_route_.size()
            << "\n ----------- \n";

  std::cout << std::endl << "Results for the MEAT CSA:" << std::endl;
  print_results(queries, results_csa, tt, gs, tt_path);
  std::cout << std::endl << "Results for the MEAT RAPTOR:" << std::endl;
  print_results(queries, results_raptor, tt, gs, tt_path);

  print_memory_usage();

  {
    auto file_path_csa = out_file_path_prefix;
    file_path_csa += "csa.csv";
    std::ofstream out_file_csa;
    out_file_csa.open(file_path_csa);
    if (out_file_csa.is_open()) {
      print_all_results(out_file_csa, results_csa);
      std::cout << "Result file meat_csa: " << file_path_csa << "\n";
    } else {
      std::cout << "Unable to open file " << file_path_csa << "\n";
    }
    out_file_csa.close();
  }
  {
    auto file_path_raptor = out_file_path_prefix;
    file_path_raptor += "raptor.csv";
    std::ofstream out_file_raptor;
    out_file_raptor.open(file_path_raptor);
    if (out_file_raptor.is_open()) {
      print_all_results(out_file_raptor, results_raptor);
      std::cout << "Result file meat_raptor: " << file_path_raptor << "\n";
    } else {
      std::cout << "Unable to open file " << file_path_raptor << "\n";
    }
    out_file_raptor.close();
  }

  return 0;
}