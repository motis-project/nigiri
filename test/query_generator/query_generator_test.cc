#include "gtest/gtest.h"

#include "utl/visit.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/query_generator/generator.h"
#include "nigiri/query_generator/generator_settings.h"
#include "nigiri/query_generator/transport_mode.h"
#include "nigiri/routing/query.h"

#include "geo/box.h"
#include "geo/latlng.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"

#include "../loader/hrd/hrd_timetable.h"
#include "../transfer_rules_util.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;
using namespace nigiri::query_generation;

TEST(query_generation, pretrip_station) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  generator_settings gs;
  gs.start_match_mode_ = routing::location_match_mode::kEquivalent;
  gs.dest_match_mode_ = routing::location_match_mode::kEquivalent;

  auto qg = generator{tt, gs};

  auto const sdq = qg.random_query();
  ASSERT_TRUE(sdq.has_value());
}

TEST(query_generation, pretrip_intermodal) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  generator_settings gs;
  gs.start_mode_ = kCar;

  auto qg = generator{tt, gs};

  auto const sdq = qg.random_query();
  ASSERT_TRUE(sdq.has_value());
}

TEST(query_generation, reproducibility) {
  constexpr auto const src = source_idx_t{0U};
  timetable tt;
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  generator_settings const gs;
  auto const seed = 2342;
  auto const num_queries = 3U;

  auto qg0 = generator{tt, gs, seed};
  auto result_qg0 =
      std::vector<std::optional<query_generation::start_dest_query>>{};
  result_qg0.reserve(num_queries);
  for (auto i = 0U; i < num_queries; ++i) {
    result_qg0.emplace_back(qg0.random_query());
  }

  auto qg1 = generator{tt, gs, seed};
  for (auto i = 0U; i < num_queries; ++i) {
    auto const result_qg1 = qg1.random_query();
    ASSERT_EQ(result_qg0[i].has_value(), result_qg1.has_value());
    if (result_qg0[i].has_value()) {
      EXPECT_EQ(result_qg0[i].value().q_, result_qg1.value().q_);
    }
  }
}

// The location r-tree only holds the non-virtual locations and yields
// positions in that pool. Taken as location indices, they would point the
// intermodal offsets at unrelated, far away stops.
TEST(query_generation, intermodal_offsets_skip_virtual_locations) {
  // Feed 0: a route-qualified same-stop rule next to the unqualified pair
  // default gives the trip stops at BY virtual locations. Feed 1 is loaded
  // after feed 0, so its stops come after feed 0's virtual locations. They are
  // far enough apart that the generator does not discard every query for
  // having a short direct walk.
  auto const tt = test::load_feeds(
      {test::feed({{"BA", 52.50, 13.30},
                   {"BY", 52.50, 13.40},
                   {"BC", 52.50, 13.50},
                   {"BD", 52.50, 13.60}},
                  {{"U1", "R10", {{"BA", "12:00"}, {"BY", "12:30"}}},
                   {"U2", "R11", {{"BY", "12:31"}, {"BC", "13:00"}}},
                   {"U3", "R12", {{"BY", "12:31"}, {"BD", "13:00"}}}},
                  "BY,BY,2,120,,,,\n"
                  "BY,BY,2,0,R10,R11,,\n"),
       test::feed(
           {{"PA", 48.85, 2.35}, {"PB", 48.85, 2.42}, {"PC", 48.85, 2.49}},
           {{"P1", "RP", {{"PA", "09:00"}, {"PB", "09:10"}, {"PC", "09:20"}}},
            {"P2", "RP", {{"PA", "10:00"}, {"PB", "10:10"}, {"PC", "10:20"}}}},
           "")});

  // Pool positions and location indices only differ if virtual locations
  // exist and real locations follow them.
  auto first_virt = std::optional<location_idx_t>{};
  auto last_real = location_idx_t{0U};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    if (tt.locations_.types_[l] == location_type::kVirt) {
      if (!first_virt.has_value()) {
        first_virt = l;
      }
    } else {
      last_real = l;
    }
  }
  ASSERT_TRUE(first_virt.has_value())
      << "precondition: feed 0 has virtual locations";
  ASSERT_GT(last_real, *first_virt)
      << "precondition: a real location follows a virtual one, so pool "
         "positions and location indices differ";

  auto gs = generator_settings{};
  gs.bbox_ =
      geo::make_box({geo::latlng{48.83, 2.33}, geo::latlng{48.87, 2.51}});

  auto qg = generator{tt, gs, 42U};

  auto checked = 0U;
  for (auto i = 0U; i != 50U; ++i) {
    auto const sdq = qg.random_query();
    if (!sdq.has_value()) {
      continue;
    }
    auto const check = [&](std::variant<location_idx_t, geo::latlng> const& p,
                           std::vector<routing::offset> const& offsets,
                           transport_mode const& mode) {
      utl::visit(p, [&](geo::latlng const& pos) {
        for (auto const& o : offsets) {
          EXPECT_NE(tt.locations_.types_[o.target()], location_type::kVirt);
          EXPECT_LE(geo::distance(pos, tt.locations_.coordinates_[o.target()]),
                    static_cast<double>(mode.range()) + 1.0)
              << "offset points outside the search radius";
          ++checked;
        }
      });
    };
    check(sdq->start_, sdq->q_.start_, gs.start_mode_);
    check(sdq->dest_, sdq->q_.destination_, gs.dest_mode_);
  }
  EXPECT_GT(checked, 0U) << "no intermodal offset was generated";
}

// Stops are drawn by the number of their events, the ones at their virtual
// locations included: all trips at X stop at virtual locations of X (the rule
// R1 -> R2), yet X is a start and a destination. Z has no events and virtual
// locations are no stops: neither is ever drawn.
TEST(query_generation, draws_stops_by_events) {
  auto const tt = test::load_feeds(
      {test::feed({{"XA", 52.50, 13.30},
                   {"X", 52.50, 13.40},
                   {"XB", 52.50, 13.50},
                   {"Z", 52.50, 13.60}},
                  {{"T1", "R1", {{"XA", "12:00"}, {"X", "12:30"}}},
                   {"T2", "R2", {{"X", "12:31"}, {"XB", "13:00"}}}},
                  "X,X,2,120,,,,\n"
                  "X,X,2,0,R1,R2,,\n")});
  auto const x = tt.locations_.location_id_to_idx_.at({"X", source_idx_t{0}});
  auto const z = tt.locations_.location_id_to_idx_.at({"Z", source_idx_t{0}});
  ASSERT_TRUE(tt.location_routes_[x].empty())
      << "precondition: X is served at its virtual locations only";

  auto gs = generator_settings{};
  gs.start_match_mode_ = routing::location_match_mode::kEquivalent;
  gs.dest_match_mode_ = routing::location_match_mode::kEquivalent;
  auto qg = generator{tt, gs, 42U};

  auto has_x_start = false;
  auto has_x_dest = false;
  for (auto i = 0U; i != 100U; ++i) {
    auto const sdq = qg.random_query();
    ASSERT_TRUE(sdq.has_value());
    for (auto const& o :
         {sdq->q_.start_.front(), sdq->q_.destination_.front()}) {
      EXPECT_NE(location_type::kVirt, tt.locations_.types_[o.target()]);
      EXPECT_NE(z, o.target());
    }
    has_x_start |= sdq->q_.start_.front().target() == x;
    has_x_dest |= sdq->q_.destination_.front().target() == x;
  }
  EXPECT_TRUE(has_x_start);
  EXPECT_TRUE(has_x_dest);
}
