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

namespace {

// a second fplan file, disjoint from services.101 so both are really read
constexpr auto const second_fplan = R"(
*Z 09001 80____       048 030                             %
*A VE 0000008 0000009 000005                              %
*G RE  0000008 0000009                                    %
0000008 H                            00230                %
0000009 I                     00330                       %
)";

}  // namespace

// route_ids_ is indexed by source, so a feed split over several fplan files
// must still end up with exactly one entry - otherwise create_rt_timetable,
// which sizes its per-source alert lists by n_sources(), writes past the end.
TEST(hrd, multiple_fplan_files_one_route_ids_entry) {
  auto d = files();
  auto const& f = hrd_5_20_26.fplan_;
  d.dir_[(f / "services.102").string()] = second_fplan;

  auto tt = timetable{};
  tt.date_range_ = full_period();
  register_special_stations(tt);
  tt.n_sources_ = 1U;  // what loader::load does for a one dataset import
  load_timetable(source_idx_t{0U}, hrd_5_20_26, d, tt);
  finalize(tt);

  ASSERT_EQ(1U, tt.route_ids_.size())
      << "one route_ids entry per source, not per fplan file";

  // used to segfault: n_sources()=1 sized list, indexed once per route_ids_
  EXPECT_NO_FATAL_FAILURE(
      { auto const rtt = rt::create_rt_timetable(tt, full_period().from_); });
}
