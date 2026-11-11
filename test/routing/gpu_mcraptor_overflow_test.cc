#if defined(NIGIRI_CUDA)

#include "gtest/gtest.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "fmt/format.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/load.h"
#include "nigiri/routing/gpu/mcraptor.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace std::chrono_literals;
using routing::gpu::gpu_mcraptor_limits;

namespace {

// Route R boards at S0 .. S<B-1> and ends at D. per_stop[b] labels reach S<b>
// in round 1, pareto over (arrival, non_transit): label g walks 5 * (N - g)
// min (N - g non_transit units) to its own access stop A<g>, whose trip Q<g>
// drops it at S<b> 5 min before R's trip g departs there. Each label catches a
// different trip of R, so all N = sum(per_stop) journeys to D stay
// pareto-optimal and R's route bag holds N entries.
loader::mem_dir make_files(std::vector<unsigned> const& per_stop) {
  auto const n = std::accumulate(begin(per_stop), end(per_stop), 0U);
  auto const n_board = static_cast<unsigned>(per_stop.size());
  auto const hhmm = [](unsigned const min) {
    return fmt::format("{:02}:{:02}:00", min / 60U, min % 60U);
  };
  // R's trip g at S<b>
  auto const dep = [](unsigned const g, unsigned const b) {
    return 9U * 60U + 10U + 10U * g + 2U * b;
  };

  auto stops = std::string{};
  auto routes = std::string{"R,DB,R,,,2\n"};
  auto trips = std::string{};
  auto stop_times = std::string{};
  auto lat = 0U;
  auto const add_stop = [&](std::string const& id) {
    stops += fmt::format("{},{},,{}.0,1.0,,\n", id, id, lat++);
  };
  for (auto b = 0U; b != n_board; ++b) {
    add_stop(fmt::format("S{}", b));
  }
  add_stop("D");
  for (auto g = 0U, b = 0U, in_b = 0U; g != n; ++g, ++in_b) {
    while (in_b == per_stop[b]) {
      ++b;
      in_b = 0U;
    }
    add_stop(fmt::format("A{}", g));
    routes += fmt::format("Q{},DB,Q{},,,2\n", g, g);
    trips += fmt::format("Q{},MON,QT{},,\n", g, g);
    stop_times += fmt::format("QT{},{},{},A{},1,0,0\n", g,
                              hhmm(dep(g, b) - 10U), hhmm(dep(g, b) - 10U), g);
    stop_times += fmt::format("QT{},{},{},S{},2,0,0\n", g, hhmm(dep(g, b) - 5U),
                              hhmm(dep(g, b) - 5U), b);

    trips += fmt::format("R,MON,RT{},,\n", g);
    for (auto s = 0U; s != n_board; ++s) {
      stop_times += fmt::format("RT{},{},{},S{},{},0,0\n", g, hhmm(dep(g, s)),
                                hhmm(dep(g, s)), s, s + 1U);
    }
    auto const arr = dep(g, n_board) + 30U;
    stop_times += fmt::format("RT{},{},{},D,{},0,0\n", g, hhmm(arr), hhmm(arr),
                              n_board + 1U);
  }

  return loader::mem_dir::read(fmt::format(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
{}
# calendar_dates.txt
service_id,date,exception_type
MON,20240617,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
{}
# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
{}
# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
{})",
                                           stops, routes, trips, stop_times));
}

struct fixture {
  explicit fixture(std::vector<unsigned> const& per_stop)
      : n_{std::accumulate(begin(per_stop), end(per_stop), 0U)},
        tt_{loader::load(make_files(per_stop),
                         {2024_y / June / 16, 2024_y / June / 19})} {}

  // from the first m access stops: m pareto journeys
  routing::query query(unsigned const m) const {
    auto starts = std::vector<routing::offset>{};
    for (auto g = 0U; g != m; ++g) {
      starts.emplace_back(
          tt_.find(location_id{fmt::format("A{}", g), source_idx_t{0}}).value(),
          duration_t{5U * (n_ - g)}, 0U);
    }
    return {
        .start_time_ = unixtime_t{sys_days{2024_y / June / 17} + 3h},
        .start_match_mode_ = routing::location_match_mode::kIntermodal,
        .dest_match_mode_ = routing::location_match_mode::kIntermodal,
        .use_start_footpaths_ = false,
        .start_ = starts,
        .destination_ = {
            {tt_.find(location_id{"D", source_idx_t{0}}).value(), 0min, 0U}}};
  }

  unsigned n_;
  timetable tt_;
};

using criteria_t =
    std::tuple<unixtime_t, unixtime_t, std::uint8_t, std::uint16_t>;

template <typename State>
std::vector<criteria_t> run(timetable const& tt,
                            routing::query const& q,
                            State& state) {
  auto ss = routing::search_state{};
  auto v = std::vector<criteria_t>{};
  for (auto const& j :
       *routing::raptor_search(tt, nullptr, ss, state, q, direction::kForward)
            .journeys_) {
    v.emplace_back(j.start_time_, j.dest_time_, j.transfers_, j.criteria_cost_);
  }
  std::sort(begin(v), end(v));
  return v;
}

std::vector<criteria_t> run_cpu(timetable const& tt, routing::query const& q) {
  auto state = routing::mcraptor_non_transit_state{};
  return run(tt, q, state);
}

using gpu_state = routing::gpu::gpu_mcraptor_non_transit_state;

}  // namespace

// The device route scan holds a route's boardings in fixed-size bags (stop
// bags and prefix-scan route bags of kMcPrefixK = 4 entries, one stop bag per
// et task slot); each overflow falls back to a wider scan and must lose
// nothing.
TEST(gpu_mcraptor, stop_bag_overflows_match_cpu) {
  struct overflow_case {
    std::vector<unsigned> per_stop_;
    char const* what_;
  };
  for (auto const& c : std::vector<overflow_case>{
           {{2U}, "route bag over K = 1: rescanned with K = 4"},
           {{5U},
            "stop bag over 4 boardings: the sequential scan looks the stop "
            "up"},
           {{3U, 3U},
            "stop bags fit, the route bag (6) does not: the sequential scan "
            "reads the stop bags"}}) {
    auto const f = fixture{c.per_stop_};
    auto const q = f.query(f.n_);
    auto const cpu = run_cpu(f.tt_, q);
    ASSERT_EQ(f.n_, cpu.size()) << c.what_ << ": every trip of R is pareto";

    auto const gtt = routing::gpu::gpu_timetable{f.tt_};
    auto state = gpu_state{gtt};
    EXPECT_EQ(cpu, run(f.tt_, q, state)) << c.what_;
    // a single et task slot: every other boardable stop has no stop bag
    auto one_slot = gpu_state{gtt, {.et_tasks_ = 1U}};
    EXPECT_EQ(cpu, run(f.tt_, q, one_slot)) << c.what_ << ", one task slot";
  }
}

// A capacity that would make the search lossy fails it (naming the capacity)
// rather than returning fewer journeys, and leaves the state usable: a search
// that fits on the same state still matches the CPU.
TEST(gpu_mcraptor, capacity_overflows_throw) {
  struct capacity_case {
    std::vector<unsigned> per_stop_;
    gpu_mcraptor_limits limits_;
    std::string_view flag_;  // in the message
    unsigned fits_;  // access stops of a search within the capacity
    char const* what_;
  };
  for (auto const& c : std::vector<capacity_case>{
           {{9U},
            {.bag_pool_blocks_ = 0U},
            "bag=true",
            2U,
            "9 labels at S, 8 inline slots, no pool blocks"},
           {{4U},
            {.breadcrumbs_ = 8U},
            "arena=true",
            1U,
            "3 breadcrumbs per journey (2 rides + egress), arena of 8"},
           {{65U},
            {},
            "route_bag=true",
            4U,
            "65 boardings at S, sequential route bag of 64"}}) {
    auto const f = fixture{c.per_stop_};
    ASSERT_EQ(f.n_, run_cpu(f.tt_, f.query(f.n_)).size()) << c.what_;

    auto const gtt = routing::gpu::gpu_timetable{f.tt_};
    auto state = gpu_state{gtt, c.limits_};
    try {
      run(f.tt_, f.query(f.n_), state);
      ADD_FAILURE() << c.what_ << ": no capacity overflow";
    } catch (std::exception const& e) {
      EXPECT_NE(std::string_view{e.what()}.find(c.flag_),
                std::string_view::npos)
          << c.what_ << ": " << e.what();
    }

    auto const fits = f.query(c.fits_);
    auto const cpu = run_cpu(f.tt_, fits);
    ASSERT_EQ(c.fits_, cpu.size()) << c.what_;
    EXPECT_EQ(cpu, run(f.tt_, fits, state)) << c.what_ << ", after overflow";
  }
}

#endif
