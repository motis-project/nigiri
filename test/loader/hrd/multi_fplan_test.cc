#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "./hrd_timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::hrd;
using namespace nigiri::test_data::hrd_timetable;

// route_ids_ is indexed by source, so a feed split over several fplan files
// must still end up with exactly one entry - otherwise create_rt_timetable,
// which sizes its per-source alert lists by n_sources(), writes past the end.
TEST(hrd, multiple_fplan_files_one_route_ids_entry) {
  // a second fplan file, disjoint from services.101 so both are really read
  constexpr auto const second_fplan = R"(
*Z 09001 80____       048 030                             %
*A VE 0000008 0000009 000005                              %
*G RE  0000008 0000009                                    %
0000008 H                            00230                %
0000009 I                     00330                       %
)";

  auto d = files();
  auto const& f = hrd_5_20_26.fplan_;
  d.dir_[(f / "services.102").string()] = second_fplan;

  auto tt = timetable{};
  tt.date_range_ = full_period();
  register_special_stations(tt);
  load_timetable(source_idx_t{0U}, hrd_5_20_26, d, tt);
  finalize(tt);

  EXPECT_EQ(1U, tt.n_sources());
  ASSERT_EQ(1U, tt.route_ids_.size())
      << "one route_ids entry per source, not per fplan file";

  // used to segfault: n_sources()=1 sized list, indexed once per route_ids_
  auto const rtt = rt::create_rt_timetable(tt, full_period().from_);

  // the alert lookup indexes route_ids_ with the trip's route id
  // (motis looks up alerts for every leg), so HRD trips need one
  ASSERT_NE(0U, tt.n_trips());
  for (auto t = trip_idx_t{0U}; t != trip_idx_t{tt.n_trips()}; ++t) {
    ASSERT_NE(route_id_idx_t::invalid(), tt.trip_route_id_[t]);
    EXPECT_TRUE(rtt.alerts_
                    .get_alerts(tt, source_idx_t{0U}, t,
                                rt_transport_idx_t::invalid(),
                                location_idx_t::invalid(), false)
                    .empty());
  }
}
