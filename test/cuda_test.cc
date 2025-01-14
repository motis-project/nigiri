#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/gpu/raptor.h"
#include "nigiri/routing/raptor_search.h"

#include "./loader/hrd/hrd_timetable.h"

namespace ngpu = nigiri::routing::gpu;
namespace nr = nigiri::routing;

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;

constexpr auto const fwd_journeys = R"(
[2020-03-30 05:00, 2020-03-30 07:15]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:00]
       TO: (C, 0000003) [2020-03-30 07:15]
leg 0: (A, 0000001) [2020-03-30 05:00] -> (B, 0000002) [2020-03-30 06:00]
   0: 0000001 A...............................................                               d: 30.03 05:00 [30.03 07:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/300/0000002/360/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:00 [30.03 08:00]
leg 1: (B, 0000002) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 06:02]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:15] -> (C, 0000003) [2020-03-30 07:15]
   0: 0000002 B...............................................                               d: 30.03 06:15 [30.03 08:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/375/0000003/435/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:15 [30.03 09:15]


[2020-03-30 05:30, 2020-03-30 07:45]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:30]
       TO: (C, 0000003) [2020-03-30 07:45]
leg 0: (A, 0000001) [2020-03-30 05:30] -> (B, 0000002) [2020-03-30 06:30]
   0: 0000001 A...............................................                               d: 30.03 05:30 [30.03 07:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/330/0000002/390/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:30 [30.03 08:30]
leg 1: (B, 0000002) [2020-03-30 06:30] -> (B, 0000002) [2020-03-30 06:32]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:45] -> (C, 0000003) [2020-03-30 07:45]
   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/405/0000003/465/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]


)";

TEST(nigiri_cuda, test) {
  constexpr auto const src = source_idx_t{0U};

  auto tt = timetable{};
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  std::cout << "HOST: #locations=" << tt.n_locations() << "\n";

  auto const gpu_tt = ngpu::gpu_timetable{tt};
  auto algo_state = ngpu::gpu_raptor_state{gpu_tt};
  auto search_state = nr::search_state{};

  auto q = routing::query{
      .start_time_ =
          interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
                   unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},
      .start_ = {{tt.locations_.location_id_to_idx_.at({"0000001", src}),
                  0_minutes, 0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({"0000003", src}),
                        0_minutes, 0U}},
      .via_stops_ = {}};
  auto const results =
      *(routing::raptor_search(tt, nullptr, search_state, algo_state,
                               std::move(q), direction::kForward)
            .journeys_);

  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n\n";
  }
  EXPECT_EQ(std::string_view{fwd_journeys}, ss.str());
}