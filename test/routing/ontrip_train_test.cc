#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/lookup/get_transport.h"
#include "nigiri/print_transport.h"
#include "nigiri/routing/ontrip_train.h"
#include "nigiri/routing/raptor.h"
#include "nigiri/routing/search_state.h"
#include "nigiri/timetable.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace nigiri;
using namespace nigiri::routing;
using namespace nigiri::test_data::hrd_timetable;
using namespace date;

TEST(routing, ontrip_train) {
  using namespace date;
  timetable tt;
  tt.date_range_ = full_period();
  constexpr auto const src = source_idx_t{0U};
  load_timetable(src, loader::hrd::hrd_5_20_26, files(), tt);

  for (auto const& [trip_id, trip_idx] : tt.trip_id_to_idx_) {
    fmt::print("trip_id=\"{}\" -> trip_idx={}\n",
               tt.trip_id_strings_[trip_id].view(), trip_idx);
  }

  auto q = routing::query{
      .start_time_ = {},
      .start_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
      .use_start_footpaths_ = true,
      .start_ = {/* filled in by generate_ontrip_train_query() */},
      .destinations_ = {{{tt.locations_.location_id_to_idx_.at(
                              {.id_ = "0000004", .src_ = src}),
                          10_minutes, 77U}}},
      .via_destinations_ = {},
      .allowed_classes_ = bitset<kNumClasses>::max(),
      .max_transfers_ = 6U,
      .min_connection_count_ = 0,
      .extend_interval_earlier_ = false,
      .extend_interval_later_ = false};
  auto const t =
      get_transport(tt, "3374/0000008/1410/0000006/2950/", March / 30 / 2020);
  ASSERT_TRUE(t.has_value());
  generate_ontrip_train_query(tt, *t, 1, q);
  auto state = routing::search_state{};
  auto fwd_r = routing::raptor<direction::kForward, true>{tt, state, q};
  fwd_r.route();

  std::stringstream ss;
  ss << "\n";
  for (auto const& x : state.results_.at(0)) {
    std::cout << "result\n";
    x.print(std::cout, tt);
    ss << "\n\n";
  }
  std::cout << "results: " << state.results_.at(0).size() << "\n";
}
