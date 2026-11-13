#include <cmath>
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <span>
#include <thread>
#include <tuple>

#include "boost/program_options.hpp"

#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
#include "utl/parallel_for.h"
#include "utl/parser/cstr.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"
#include "utl/zip.h"

#include "nigiri/logging.h"
#include "nigiri/qa/qa.h"
#include "nigiri/query_generator/generator.h"
#include "nigiri/routing/interval_estimate.h"
#include "nigiri/routing/raptor/bmrap_common.h"
#include "nigiri/routing/raptor/bmraptor.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/mcraptor.h"
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

struct result_set {
  std::string label_;
  std::vector<pareto_set<routing::journey>> res_;
  // the interval each search reports as covered (routing_result::interval_)
  std::vector<interval<unixtime_t>> intervals_;
  std::vector<double> latencies_;
};

// bmrapp optimizes every criterion the device mcraptor supports as well
using bmrapp_criteria_t =
    routing::arr_non_transit_mode_filter_mode_switches_criteria;

// algorithms optimizing more than (departure, arrival, transfers)
bool is_multicriteria(std::string const& label) {
  return label.find("-bmrapp-") != std::string::npos ||
         label.find("-mcraptor-all-") != std::string::npos ||
         label.find("-mcraptor-restricted-") != std::string::npos;
}

// Problems with a journey's legs, empty = valid: rides as in the timetable,
// connected in place and time (footpaths as in the timetable, same-stop
// transfers at least the transfer time), and the journey's transfers and - if
// `mc` - extra criteria recomputed from the legs as the mc engines count them.
std::vector<std::string> leg_problems(timetable const& tt,
                                      query const& q,
                                      journey const& j,
                                      bool const mc) {
  auto problems = std::vector<std::string>{};
  auto const fail = [&]<typename... Args>(fmt::format_string<Args...> f,
                                          Args&&... args) {
    problems.push_back(fmt::format(f, std::forward<Args>(args)...));
  };
  if (j.legs_.empty()) {
    fail("no legs");
    return problems;
  }
  if (j.legs_.front().dep_time_ != j.departure_time() ||
      j.legs_.back().arr_time_ != j.arrival_time()) {
    fail("legs span {}..{}, journey {}..{}", j.legs_.front().dep_time_,
         j.legs_.back().arr_time_, j.departure_time(), j.arrival_time());
  }
  auto const transfer = [&](location_idx_t const l) {
    return duration_t{adjusted_transfer_time(
        q.transfer_time_settings_, tt.locations_.transfer_time_[l].count())};
  };

  auto n_rides = 0U;
  auto units = 0U;
  auto avoided = false;
  auto group = routing::kNoModeGroup;
  auto switches = 0U;
  auto prev_ride_end = std::optional<std::pair<location_idx_t, unixtime_t>>{};
  for (auto i = 0U; i != j.legs_.size(); ++i) {
    auto const& l = j.legs_[i];
    if (l.arr_time_ < l.dep_time_) {
      fail("leg {} ends before it starts", i);
    }
    if (i != 0U) {
      auto const& p = j.legs_[i - 1U];
      if (p.to_ != l.from_ || p.arr_time_ > l.dep_time_) {
        fail("leg {} does not continue leg {} ({} at {} -> {} at {})", i, i - 1,
             tt.locations_.ids_[p.to_].view(), p.arr_time_,
             tt.locations_.ids_[l.from_].view(), l.dep_time_);
      }
    }
    std::visit(
        utl::overloaded{
            [&](journey::run_enter_exit const& ree) {
              auto fr = rt::frun{tt, nullptr, ree.r_};
              fr.stop_range_ = ree.stop_range_;
              auto const first = fr[0];
              auto const last =
                  fr[static_cast<stop_idx_t>(fr.stop_range_.size() - 1U)];
              if (first.get_location_idx() != l.from_ ||
                  last.get_location_idx() != l.to_ ||
                  first.time(event_type::kDep) != l.dep_time_ ||
                  last.time(event_type::kArr) != l.arr_time_) {
                fail("ride {} not as in the timetable", i);
              }
              if (!first.in_allowed() || !last.out_allowed()) {
                fail("ride {} boards or alights where it may not", i);
              }
              if (prev_ride_end.has_value() &&
                  prev_ride_end->first == l.from_ &&
                  l.dep_time_ - prev_ride_end->second < transfer(l.from_)) {
                fail("ride {}: transfer at {} shorter than {}", i,
                     tt.locations_.ids_[l.from_].view(), transfer(l.from_));
              }
              prev_ride_end = {l.to_, l.arr_time_};
              auto const c = tt.route_clasz_[tt.transport_route_[fr.t_.t_idx_]];
              avoided |=
                  is_allowed(routing::mode_filter_dim::avoided_mask(), c);
              auto const g = routing::mode_group(c);
              switches = routing::switches_after_ride(group, switches, g);
              group = g;
              ++n_rides;
            },
            [&](footpath const& fp) {
              if (l.from_ == l.to_) {
                return;  // a same-stop transfer: no walk
              }
              auto const fps =
                  tt.locations_.footpaths_out_[q.prf_idx_][l.from_];
              auto const it = utl::find_if(
                  fps, [&](footpath const& x) { return x.target() == l.to_; });
              auto const need =
                  it == end(fps)
                      ? fp.duration()
                      : duration_t{adjusted_transfer_time(
                            q.transfer_time_settings_, it->duration().count())};
              if (it == end(fps) || l.arr_time_ - l.dep_time_ < need) {
                fail("footpath {} {} -> {} not in the timetable or too short",
                     i, tt.locations_.ids_[l.from_].view(),
                     tt.locations_.ids_[l.to_].view());
              }
              units = routing::non_transit_add(
                  units, static_cast<std::uint32_t>(
                             (l.arr_time_ - l.dep_time_).count()));
            },
            [&](offset const& o) {
              units = routing::non_transit_add(
                  units, static_cast<std::uint32_t>(o.duration().count()));
            }},
        l.uses_);
  }
  if (n_rides != 0U && j.transfers_ != n_rides - 1U) {
    fail("transfers {} with {} rides", j.transfers_, n_rides);
  }
  if (mc) {
    if (j.criteria_cost_ != units * routing::kNonTransitBin ||
        j.criteria_mode_filter_ != avoided ||
        j.criteria_mode_switches_ != switches) {
      fail("criteria {}/{}/{} recomputed from the legs {}/{}/{}",
           j.criteria_cost_, j.criteria_mode_filter_, j.criteria_mode_switches_,
           units * routing::kNonTransitBin, avoided, switches);
    }
  }
  return problems;
}

// Range McRAPTOR over every criterion, cut down by the restricted-set
// definition (bmrap_common.h): the anchors are its own journeys optimal on
// (departure, arrival, transfers) alone, and a journey survives if it is an
// anchor or stays within the trip budget and deadline of its anchors. This is
// what bmrapp computes, the slow way. See compare_restricted() for where the
// two can be compared.
std::vector<routing::bmrap_detail::anchor> anchors_of(
    pareto_set<journey> const& res) {
  auto anchors = std::vector<routing::bmrap_detail::anchor>{};
  for (auto const& j : res) {
    if (utl::none_of(res, [&](journey const& o) {
          return o.tuple_dominates(j) && !j.tuple_dominates(o);
        })) {
      anchors.push_back({j.start_time_, j.dest_time_,
                         static_cast<std::uint8_t>(j.transfers_ + 1U)});
    }
  }
  return anchors;
}

template <direction SearchDir>
routing::routing_result restricted_range_search(
    timetable const& tt,
    search_state& ss,
    routing::mcraptor_non_transit_mode_filter_mode_switches_state& state,
    routing::query q) {
  auto r =
      routing::raptor_search(tt, nullptr, ss, state, std::move(q), SearchDir);
  auto& res = ss.results_;
  auto const anchors = anchors_of(res);
  utl::erase_if(res.els_, [&](journey const& j) {
    return routing::bmrap_detail::outside_restriction<SearchDir>(
        anchors, j.start_time_, j.dest_time_,
        static_cast<unsigned>(j.transfers_) + 1U);
  });
  return r;
}

// bmrapp against the restricted range McRAPTOR: the same set, criteria and all,
// for every journey both can decide. The restriction of a journey depends on
// the anchors after it, so near the end of its window the reference may lack
// one. It can decide a journey anchored at d with n trips iff one of its
// anchors y is available at d (anchored no earlier), has at most n trips, and
// arrives (found_) within its covered interval: any anchor it never saw is
// anchored after that, so it arrives after y and changes nothing. Legs differ
// where equal journeys have several paths; those are only counted.
struct restricted_diff {
  std::uint64_t mismatches_{0U}, compared_{0U}, other_legs_{0U};
};
restricted_diff compare_restricted(result_set const& bm,
                                   result_set const& ref,
                                   direction const search_dir) {
  auto const fwd = search_dir == direction::kForward;
  auto const is_better = [&](unixtime_t const a, unixtime_t const b) {
    return fwd ? a < b : a > b;
  };
  auto const criteria = [](journey const& j) {
    return std::tuple{j.start_time_,
                      j.dest_time_,
                      j.transfers_,
                      j.criteria_cost_,
                      j.criteria_mode_filter_,
                      j.criteria_mode_switches_};
  };
  auto d = restricted_diff{};
  for (auto i = 0U; i != bm.res_.size(); ++i) {
    auto const& itv = ref.intervals_[i];
    auto const window =
        interval<unixtime_t>{std::max(itv.from_, bm.intervals_[i].from_),
                             std::min(itv.to_, bm.intervals_[i].to_)};
    auto const edge = fwd ? itv.to_ : itv.from_;
    auto const anchors = anchors_of(ref.res_[i]);
    auto const decidable = [&](journey const& j) {
      return window.contains(j.start_time_) &&
             j.travel_time() < kCheckedMaxTravelTime &&
             utl::any_of(anchors, [&](routing::bmrap_detail::anchor const& y) {
               return !is_better(y.anchored_, j.start_time_) &&
                      y.trips_ <= j.transfers_ + 1U &&
                      !is_better(edge, y.found_);
             });
    };
    auto const set = [&](pareto_set<journey> const& res) {
      auto v = std::vector<journey const*>{};
      for (auto const& j : res) {
        if (decidable(j)) {
          v.push_back(&j);
        }
      }
      utl::sort(v, [&](journey const* a, journey const* b) {
        return criteria(*a) < criteria(*b);
      });
      return v;
    };
    auto const b = set(bm.res_[i]);
    auto const r = set(ref.res_[i]);
    auto problems = std::vector<std::string>{};
    auto const key = [](journey const& j) {
      return fmt::format("dep={} arr={} tr={} walk={} avoided={} switches={}",
                         j.departure_time(), j.arrival_time(), j.transfers_,
                         j.criteria_cost_, j.criteria_mode_filter_,
                         j.criteria_mode_switches_);
    };
    d.compared_ += b.size();
    for (auto const* x : b) {
      auto const it = utl::find_if(
          r, [&](journey const* y) { return criteria(*x) == criteria(*y); });
      if (it == end(r)) {
        problems.push_back(fmt::format("only {}: {}", bm.label_, key(*x)));
      } else if ((*it)->legs_ != x->legs_) {
        ++d.other_legs_;
      }
    }
    for (auto const* y : r) {
      if (utl::none_of(b, [&](journey const* x) {
            return criteria(*x) == criteria(*y);
          })) {
        problems.push_back(fmt::format("only {}: {}", ref.label_, key(*y)));
      }
    }
    if (!problems.empty()) {
      fmt::println("query #{} {} vs {}: {} restricted mismatches", i, bm.label_,
                   ref.label_, problems.size());
      for (auto const& p : problems) {
        fmt::println("  === MISMATCH: {} ===", p);
      }
    }
    d.mismatches_ += problems.size();
  }
  return d;
}

// every journey of a cell through leg_problems(); returns the invalid ones
std::uint64_t validate_legs(
    timetable const& tt,
    result_set const& cell,
    std::vector<nigiri::query_generation::start_dest_query> const& queries) {
  auto invalid = std::uint64_t{0U};
  for (auto i = 0U; i != cell.res_.size(); ++i) {
    for (auto const& j : cell.res_[i]) {
      auto const problems =
          leg_problems(tt, queries[i].q_, j, is_multicriteria(cell.label_));
      if (!problems.empty()) {
        if (invalid < 10U) {
          fmt::println("query #{} {}: invalid legs: {}", i, cell.label_,
                       fmt::join(problems, "; "));
        }
        ++invalid;
      }
    }
  }
  return invalid;
}

std::uint64_t compare_results(
    timetable const& tt,
    result_set const& ref,
    result_set const& cmp,
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    direction const search_dir,
    unsigned const min_connection_count) {
  auto const& ref_name = ref.label_;
  auto const& cmp_name = cmp.label_;
  auto mismatches = std::uint64_t{0U};

  auto const criteria = [](journey const& j) {
    return std::tuple{j.start_time_,
                      j.transfers_,
                      j.dest_time_,
                      j.criteria_cost_,
                      j.criteria_mode_filter_,
                      j.criteria_mode_switches_};
  };
  auto const front_criteria = [](journey const& j) {
    return std::tuple{j.start_time_, j.transfers_, j.dest_time_};
  };
  auto const front_dominates = [](journey const& a, journey const& b) {
    return a.departure_time() >= b.departure_time() &&
           a.arrival_time() <= b.arrival_time() && a.transfers_ <= b.transfers_;
  };
  auto const strictly_dominates = [&](journey const& a, journey const& b) {
    return front_dominates(a, b) && !front_dominates(b, a);
  };

  // labels end in the engine, so this compares everything but the engine
  auto const same_algo = ref_name.substr(0U, ref_name.rfind('-')) ==
                         cmp_name.substr(0U, cmp_name.rfind('-'));

  auto const key = [](journey const& j) {
    return fmt::format(
        "dep={} arr={} transfers={} cost={} mode_filter={} mode_switches={}",
        j.departure_time(), j.arrival_time(), j.transfers_, j.criteria_cost_,
        j.criteria_mode_filter_, j.criteria_mode_switches_);
  };

  auto const max_window = [&](query const& q) {
    return search_dir == direction::kForward
               ? interval_estimator<direction::kForward>{tt, q}.max_interval()
               : interval_estimator<direction::kBackward>{tt, q}.max_interval();
  };

  auto const filtered = [&](pareto_set<routing::journey> const& set,
                            interval<unixtime_t> const& window) {
    auto v = std::vector<journey const*>{};
    for (auto const& j : set) {
      if (j.travel_time() < kCheckedMaxTravelTime /* slack for pong */ &&
          window.contains(j.start_time_) /* range raptor search limit */) {
        v.push_back(&j);
      }
    }
    // the search leaves ties on the last criteria in no particular order
    std::sort(begin(v), end(v), [&](journey const* a, journey const* b) {
      return criteria(*a) < criteria(*b);
    });
    if (search_dir == direction::kBackward) {
      std::reverse(begin(v), end(v));
    }
    return v;
  };

  // (departure, arrival, transfers) pareto front of a set, one per tie
  auto const front = [&](std::vector<journey const*> const& v) {
    auto out = std::vector<journey const*>{};
    for (auto const* a : v) {
      if (utl::none_of(v, [&](journey const* b) {
            return front_dominates(*b, *a) &&
                   (!front_dominates(*a, *b) || b < a);
          })) {
        out.push_back(a);
      }
    }
    return out;
  };

  auto const print_set = [&](std::string const& name, auto const& journeys) {
    fmt::print("  {}: ", name);
    for (auto const* j : journeys) {
      fmt::print("[{}] ", key(*j));
    }
    fmt::println("");
  };

  for (auto i = std::size_t{0U}; i != ref.res_.size(); ++i) {
    // Engines running the same algorithm have to find the same set within
    // the widest window. Different algorithms scan different windows (their
    // own stop criteria), so they are compared where both say they searched.
    auto window = max_window(queries[i].q_);
    if (!same_algo) {
      for (auto const& itv : {ref.intervals_[i], cmp.intervals_[i]}) {
        window = {std::max(window.from_, itv.from_),
                  std::min(window.to_, itv.to_)};
      }
      if (window.from_ >= window.to_) {
        continue;  // nothing both covered
      }
    }
    auto const r = filtered(ref.res_[i], window);
    auto const c = filtered(cmp.res_[i], window);

    auto misses = std::uint64_t{0U};
    auto problems = std::vector<std::string>{};
    if (same_algo) {
      auto const n = std::min(r.size(), c.size());
      misses += std::max(r.size(), c.size()) - n;
      for (auto k = std::size_t{0U}; k != n; ++k) {
        if (criteria(*r[k]) != criteria(*c[k])) {
          ++misses;
          problems.push_back(fmt::format("{} [{}] vs {} [{}]", ref_name,
                                         key(*r[k]), cmp_name, key(*c[k])));
        }
      }
    } else {
      // Count under-delivering results: >= min_connection_count reached by
      // one but not the other. A multicriteria side reaches the count with
      // criteria variants of fewer tuples, so it never makes a two-criteria
      // side under-deliver.
      auto const raw_r = ref.res_[i].size();
      auto const raw_c = cmp.res_[i].size();
      auto const ref_mc = is_multicriteria(ref_name);
      auto const cmp_mc = is_multicriteria(cmp_name);
      if (r.size() >= min_connection_count && raw_c < min_connection_count &&
          (cmp_mc || !ref_mc)) {
        misses += min_connection_count - raw_c;
      }
      if (c.size() >= min_connection_count && raw_r < min_connection_count &&
          (ref_mc || !cmp_mc)) {
        misses += min_connection_count - raw_r;
      }

      // A two-criteria side's front is exact: the other side must contain
      // each of its tuples and may beat none. A multicriteria side's own
      // front is no such reference - it keeps journeys whose two-criteria
      // dominator departs outside the window.
      auto const check = [&](std::string const& x_name,
                             std::vector<journey const*> const& x,
                             std::string const& y_name,
                             std::vector<journey const*> const& y) {
        for (auto const* a : front(x)) {
          if (utl::none_of(y, [&](journey const* b) {
                return front_criteria(*a) == front_criteria(*b);
              })) {
            ++misses;
            problems.push_back(
                fmt::format("{} lacks {} [{}]", y_name, x_name, key(*a)));
          }
          for (auto const* b : y) {
            if (strictly_dominates(*b, *a)) {
              ++misses;
              problems.push_back(fmt::format("{} [{}] beats {} [{}]", y_name,
                                             key(*b), x_name, key(*a)));
            }
          }
        }
      };
      if (!is_multicriteria(ref_name)) {
        check(ref_name, r, cmp_name, c);
      }
      if (!is_multicriteria(cmp_name)) {
        check(cmp_name, c, ref_name, r);
      }
    }

    if (misses != 0U) {
      fmt::println(
          "query #{} mismatches={} ({} n={} itv={}, {} n={} itv={}, "
          "compared in {})",
          i, misses, ref_name, r.size(), ref.intervals_[i], cmp_name, c.size(),
          cmp.intervals_[i], window);
      print_set(ref_name, r);
      print_set(cmp_name, c);
      for (auto const& p : problems) {
        fmt::println("  === MISMATCH: {} ===", p);
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
  auto const q = [&](double const p) {  // nearest rank
    return l.empty() ? 0.0
                     : l[static_cast<std::size_t>(
                             std::max(std::ceil(p * l.size()), 1.0)) -
                         1U];
  };
  auto const avg =
      l.empty() ? 0.0 : std::accumulate(begin(l), end(l), 0.0) / l.size();
  fmt::print(
      "| {:<36} | {:>6.1f} | {:>6.0f} | {:>6.0f} | {:>6.0f} | "
      "{:>6.0f} | {:>6.0f} |\n",
      tag, qps, avg, q(0.50), q(0.90), q(0.99), l.empty() ? 0.0 : l.back());
  return lat;
}

// bmrapp: the average time per phase (bmrapp_ms_* stats) over the answered
// queries, and what the phases leave of the average latency
void print_phases(
    std::string const& tag,
    std::vector<double> const& latencies,
    std::vector<std::map<std::string, std::uint64_t>> const& stats) {
  constexpr auto kPrefix = std::string_view{"bmrapp_ms_"};
  auto sum = std::map<std::string, double>{};
  auto latency = 0.0;
  auto n = 0U;
  for (auto i = 0U; i != stats.size(); ++i) {
    if (latencies[i] < 0.0 || !stats[i].contains("bmrapp_steps")) {
      continue;
    }
    ++n;
    latency += latencies[i];
    for (auto const& [k, v] : stats[i]) {
      if (k.starts_with(kPrefix)) {
        sum[k.substr(kPrefix.size())] += static_cast<double>(v);
      }
    }
  }
  if (n == 0U) {
    return;
  }
  auto line = fmt::format("  phases {} (avg ms, n={}):", tag, n);
  auto phases = 0.0;
  for (auto const& [k, v] : sum) {
    line += fmt::format(" {}={:.0f}", k, v / n);
    phases += v;
  }
  fmt::println("{} other={:.0f}", line, (latency - phases) / n);
}

// --stats_path: one JSON line per (cell, query) - latency, result count, the
// query window and covered interval (unix minutes) and the algo stats
auto stats_out = std::ofstream{};

void dump_stats(
    std::string const& tag,
    std::vector<nigiri::query_generation::start_dest_query> const& queries,
    result_set const& out,
    std::vector<std::map<std::string, std::uint64_t>> const& stats) {
  if (!stats_out.is_open()) {
    return;
  }
  auto const minutes = [](unixtime_t const t) {
    return t.time_since_epoch().count();
  };
  for (auto i = 0U; i != queries.size(); ++i) {
    auto const window = std::visit(
        utl::overloaded{
            [](interval<unixtime_t> const x) { return x; },
            [](unixtime_t const t) { return interval<unixtime_t>{t, t}; }},
        queries[i].q_.start_time_);
    auto line = fmt::format(
        R"({{"cell":"{}","i":{},"ms":{:.1f},"n":{},"q":[{},{}],"itv":[{},{}])",
        tag, i, out.latencies_[i], out.res_[i].size(), minutes(window.from_),
        minutes(window.to_), minutes(out.intervals_[i].from_),
        minutes(out.intervals_[i].to_));
    for (auto const& [k, v] : stats[i]) {
      line += fmt::format(R"(,"{}":{})", k, v);
    }
    stats_out << line << "}\n";
  }
  stats_out.flush();
}

// worker state per search state type
template <typename RS>
struct cpu_ws_t {
  search_state ss_;
  RS rs_;
};

#if defined(NIGIRI_CUDA)
template <typename RS>
struct gpu_ws_t {
  explicit gpu_ws_t(routing::gpu::gpu_timetable const& gtt)
      : rs_{std::make_unique<RS>(gtt)} {}
  search_state ss_;
  std::unique_ptr<RS> rs_;
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
  out.intervals_.resize(queries.size());

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
    auto stats =
        std::vector<std::map<std::string, std::uint64_t>>(queries.size());
    out.latencies_ =
        run_load<WS>(queries, label + "-" + std::to_string(n), pool,
                     [&](WS& w, routing::query q, std::size_t const i) {
                       auto const r = search(w, std::move(q));
                       out.res_[i] = *r.journeys_;
                       out.intervals_[i] = r.interval_;
                       stats[i] = r.algo_stats_;
                     });
    print_phases(label + "-" + std::to_string(n), out.latencies_, stats);
    dump_stats(label + "-" + std::to_string(n), queries, out, stats);
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
  auto stats_path = std::filesystem::path{};
  auto engines = std::vector<std::string>{"cpu", "gpu"};
  auto algos = std::vector<std::string>{"range", "pong"};
  auto modes = std::vector<std::string>{"s2s", "c2c"};
  auto dirs = std::vector<std::string>{"fwd"};
  auto threads_v =
      std::vector<unsigned>{std::max(std::thread::hardware_concurrency(), 1U)};
  auto gpu_states_v = std::vector<unsigned>{2U};
  auto bmrap_gpu_mc_v = std::vector<int>{1};

  bpo::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce this help message")  //
      ("tt_path,p", bpo::value(&tt_path)->required(),
       "path to a binary file containing a serialized nigiri timetable")  //
      ("engines", bpo::value(&engines)->multitoken(),
       "engines to benchmark (default: cpu gpu); every axis is a vector -- "
       "the run is the full cross product of engines x algos x modes x dirs "
       "(x threads/states within an engine), all against the once-loaded "
       "timetable, with one PROFILE throughput/latency line per point; "
       "whenever more than one (engine, algo) cell ran within a (mode, dir), "
       "their pareto sets are pairwise cross-checked per query and the "
       "process exits non-zero on any divergence")  //
      ("algo,a", bpo::value(&algos)->multitoken(),
       "algorithms: range | pong | mcraptor | mcraptor-all | bmrapp (default: "
       "range pong); mcraptor-all optimizes non_transit, mode_filter and "
       "mode_switches; bmrapp is the BM-RAPTOR profile search over the same "
       "criteria; mcraptor-restricted (cpu) is mcraptor-all cut down by the "
       "restricted-set definition, which bmrapp must equal criteria and all "
       "(compare_restricted()); every cell's legs are validated "
       "(leg_problems()); against another algorithm, a two-criteria side's "
       "(departure, arrival, transfers) front must be contained in the other "
       "side and not be beaten by it, within the intervals both report; "
       "every ran "
       "cell (any engine/algo combination) within a (mode, "
       "dir) is checked pairwise against every other for agreement on the "
       "intersection of the final search intervals")  //
      ("modes", bpo::value(&modes)->multitoken(),
       "<start>2<dest> query modes with s = station, c = coordinate: "
       "s2s | s2c | c2s | c2c (default: s2s c2c); c = intermodal offsets "
       "(walk)")  //
      ("dirs", bpo::value(&dirs)->multitoken(),
       "search directions: fwd | bwd | fwd-opp | bwd-opp (default: fwd); bwd "
       "flips the generated queries (start/dest swapped, vias reversed) and "
       "searches backward = arriveBy with the interval as arrival window; the "
       "interval extension flags are forced to the search direction (fwd: "
       "later only, bwd: earlier only), overriding -e/-l; -opp extends against "
       "it instead (fwd: earlier, bwd: later), as a later page does")  //
      ("threads", bpo::value(&threads_v)->multitoken(),
       "CPU worker thread counts to sweep (default: hardware "
       "concurrency)")  //
      ("gpu_states", bpo::value(&gpu_states_v)->multitoken(),
       "concurrent GPU pipeline counts to sweep (default: 2)")  //
      ("bmrap_gpu_mc", bpo::value(&bmrap_gpu_mc_v)->multitoken(),
       "bmrapp on the GPU: where the multicriteria phases run, one cell "
       "each (0 = CPU, 1 = device; default: 1)")  //
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
      ("stats_path", bpo::value(&stats_path),
       "write per-query latency, interval and algo stats as JSON lines")  //
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
  if (!stats_path.empty()) {
    stats_out.open(stats_path);
    utl::verify(stats_out.is_open(), "cannot open {}", stats_path.string());
  }

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

  // ---- benchmark matrix: engines x algos x modes x dirs (x threads/states)
  for (auto const& d : dirs) {
    if (d != "fwd" && d != "bwd" && d != "fwd-opp" && d != "bwd-opp") {
      std::cerr << "invalid dir \"" << d
                << "\", expected fwd | bwd | fwd-opp | bwd-opp\n";
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
    if (a != "range" && a != "pong" && a != "mcraptor" && a != "mcraptor-all" &&
        a != "mcraptor-restricted" && a != "bmrapp") {
      std::cerr << "invalid algo \"" << a
                << "\", expected range | pong | mcraptor | mcraptor-all | "
                   "mcraptor-restricted | bmrapp\n";
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
  fmt::print("| {:<36} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} | {:>6} |\n",
             "config", "q/s", "avg ms", "median", "q90", "q99", "max");
  fmt::print(
      "| {0:-<36} | {0:->5}: | {0:->5}: | {0:->5}: | {0:->5}: | "
      "{0:->5}: | {0:->5}: |\n",
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
    }

    // (mode, dir) are the incomparable dimensions -- within one (mode, dir),
    // every (engine, algo) combination has to agree
    for (auto const& dir_str : dirs) {
      auto const dir = dir_str.starts_with("fwd") ? direction::kForward
                                                  : direction::kBackward;
      // extend against the search direction (bmrapp's opposed path)
      auto const opposed = dir_str.ends_with("-opp");

      auto qs = fwd_qs;
      for (auto& sdq : qs) {
        if (dir == direction::kBackward) {
          sdq.q_.flip_dir();
          sdq.q_.use_start_footpaths_ =
              gs.use_start_footpaths_ &&
              sdq.q_.start_match_mode_ != location_match_mode::kIntermodal;
        }
        sdq.q_.extend_interval_earlier_ =
            (dir == direction::kBackward) != opposed;
        sdq.q_.extend_interval_later_ = (dir == direction::kForward) != opposed;
      }

      auto cells = std::vector<result_set>{};
      for (auto const& algo : algos) {
        // pong and bmrapp run on the raptor state, the rest are range searches
        auto const use_pong = algo == "pong";
        auto const use_bmrap = algo == "bmrapp";
        auto const use_restricted = algo == "mcraptor-restricted";
        auto const algo_part = use_pong || use_bmrap ? std::string{"range"}
                               : use_restricted ? std::string{"mcraptor-all"}
                                                : algo;
        auto const label = mode + "-" + dir_str + "-" + algo;

        // pong only exists for the plain raptor state
        auto const run_cpu_cell = [&]<typename RS>() {
          return run_cell<cpu_ws_t<RS>>(
              qs, label + "-cpu", threads_v,
              [&](cpu_ws_t<RS>& w, routing::query q) {
                if constexpr (std::is_same_v<RS, routing::raptor_state>) {
                  return use_bmrap
                             ? routing::bmrap_profile_search<bmrapp_criteria_t>(
                                   tt, nullptr, w.ss_, w.rs_, std::move(q), dir)
                         : use_pong
                             ? routing::pong_search(tt, nullptr, w.ss_, w.rs_,
                                                    std::move(q), dir)
                             : routing::raptor_search(tt, nullptr, w.ss_, w.rs_,
                                                      std::move(q), dir);
                } else {
                  utl::verify(!use_pong, "pong only supports raptor_state");
                  if constexpr (
                      std::is_same_v<
                          RS,
                          routing::
                              mcraptor_non_transit_mode_filter_mode_switches_state>) {
                    if (use_restricted) {
                      return dir == direction::kForward
                                 ? restricted_range_search<direction::kForward>(
                                       tt, w.ss_, w.rs_, std::move(q))
                                 : restricted_range_search<
                                       direction::kBackward>(tt, w.ss_, w.rs_,
                                                             std::move(q));
                    }
                  }
                  return routing::raptor_search(tt, nullptr, w.ss_, w.rs_,
                                                std::move(q), dir);
                }
              });
        };
#if defined(NIGIRI_CUDA)
        auto const run_gpu_cell = [&]<typename RS>(
                                      std::string const& engine = "gpu",
                                      bool const gpu_mc = true) {
          return run_cell<gpu_ws_t<RS>>(
              qs, label + "-" + engine, gpu_states_v,
              [&](gpu_ws_t<RS>& w, routing::query q) {
                if constexpr (std::is_same_v<RS,
                                             routing::gpu::gpu_raptor_state>) {
                  return use_bmrap
                             ? routing::bmrap_profile_search<bmrapp_criteria_t>(
                                   tt, nullptr, w.ss_, *w.rs_, std::move(q),
                                   dir, std::nullopt, gpu_mc)
                         : use_pong
                             ? routing::pong_search(tt, nullptr, w.ss_, *w.rs_,
                                                    std::move(q), dir)
                             : routing::raptor_search(tt, nullptr, w.ss_,
                                                      *w.rs_, std::move(q),
                                                      dir);
                } else {
                  utl::verify(!use_pong, "pong only supports gpu_raptor_state");
                  return routing::raptor_search(tt, nullptr, w.ss_, *w.rs_,
                                                std::move(q), dir);
                }
              },
              *gpu_tt);
        };
#endif

        try {
          if (run_cpu) {
            cells.push_back(
                algo_part == "mcraptor"
                    ? run_cpu_cell
                          .template operator()<routing::mcraptor_state>()
                : algo_part == "mcraptor-all"
                    ? run_cpu_cell.template operator()<
                          routing::
                              mcraptor_non_transit_mode_filter_mode_switches_state>()
                    : run_cpu_cell
                          .template operator()<routing::raptor_state>());
            ++qa_n_cpu_cells;
            if (vm.count("qa_path")) {
              qa_cell = cells.back();
            }
          }

#if defined(NIGIRI_CUDA)
          if (run_gpu && use_bmrap) {
            // one cell per mc placement ("gpu.mc0": mc phases on the CPU,
            // "gpu.mc1": on the device); keeping the engine the last label
            // token checks all of them as one algorithm
            for (auto const m : bmrap_gpu_mc_v) {
              cells.push_back(
                  run_gpu_cell
                      .template operator()<routing::gpu::gpu_raptor_state>(
                          "gpu.mc" + std::to_string(m != 0 ? 1 : 0), m != 0));
            }
          } else if (run_gpu && !use_restricted) {  // a cpu reference
            cells.push_back(
                algo_part == "mcraptor"
                    ? run_gpu_cell.template
                      operator()<routing::gpu::gpu_mcraptor_state>()
                : algo_part == "mcraptor-all"
                    ? run_gpu_cell.template operator()<
                          routing::gpu::
                              gpu_mcraptor_non_transit_mode_filter_mode_switches_state>()
                    : run_gpu_cell.template
                      operator()<routing::gpu::gpu_raptor_state>());
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
      auto const is_restricted = [](result_set const& c) {
        return c.label_.find("-mcraptor-restricted-") != std::string::npos;
      };
      auto const is_bmrapp = [](result_set const& c) {
        return c.label_.find("-bmrapp-") != std::string::npos;
      };
      for (auto const& c : cells) {
        auto const invalid = validate_legs(tt, c, qs);
        summary.push_back(fmt::format("{:<24} legs: invalid={:<4} {}", c.label_,
                                      invalid,
                                      invalid == 0U ? "PASS" : "FAIL"));
        total += invalid;
      }
      for (auto a = std::size_t{0U}; a < cells.size(); ++a) {
        for (auto b = a + 1U; b < cells.size(); ++b) {
          if ((is_restricted(cells[a]) && is_bmrapp(cells[b])) ||
              (is_bmrapp(cells[a]) && is_restricted(cells[b]))) {
            auto const& bm = is_bmrapp(cells[a]) ? cells[a] : cells[b];
            auto const& ref = is_bmrapp(cells[a]) ? cells[b] : cells[a];
            auto const d = compare_restricted(bm, ref, dir);
            summary.push_back(fmt::format(
                "{:<24} vs {:<24} n={:<6} mismatches={:<4} {} ({} journeys "
                "compared, {} with other legs)",
                bm.label_, ref.label_, qs.size(), d.mismatches_,
                d.mismatches_ == 0U ? "PASS" : "FAIL", d.compared_,
                d.other_legs_));
            total += d.mismatches_;
            continue;
          }
          auto const mismatches = compare_results(
              tt, cells[a], cells[b], qs, dir, gs.min_connection_count_);
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
