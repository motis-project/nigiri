#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/lookup/get_transport.h"
#include "nigiri/routing/ontrip_train.h"
#include "nigiri/timetable.h"

#include "../loader/hrd/hrd_timetable.h"
#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;
using namespace nigiri::test_data::hrd_timetable;
using nigiri::test::raptor_search;

TEST(routing, ontrip_train) {
  using namespace date;
  timetable tt;
  tt.date_range_ = full_period();
  constexpr auto const src = source_idx_t{0U};
  load_timetable(src, loader::hrd::hrd_5_20_26, files(), tt);
  finalize(tt);

  auto const t = get_ref_transport(
      tt, {"3374/0000008/1350/0000006/2950/", source_idx_t{0}},
      March / 29 / 2020, false);
  ASSERT_TRUE(t.has_value());

  auto q = routing::query{
      .start_time_ = {},
      .start_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .start_ = {},
      .destination_ = {{tt.locations_.location_id_to_idx_.at(
                            {.id_ = "0000004", .src_ = src}),
                        10_minutes, 77U}}};
  generate_ontrip_train_query(tt, t->first, 1, q);

  auto const results = raptor_search(tt, nullptr, std::move(q));

  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    std::cout << "result\n";
    x.print(std::cout, tt);
    ss << "\n\n";
  }
  std::cout << "results: " << results.size() << "\n";
}
