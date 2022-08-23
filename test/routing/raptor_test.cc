#include "doctest/doctest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/raptor.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace nigiri;

TEST_CASE("raptor, simple_search") {
  using namespace date;
  auto tt = std::make_shared<timetable>();
  auto const src = source_idx_t{0U};
  load_timetable(src, loader::hrd::hrd_5_20_26,
                 test_data::hrd_timetable::files_simple(), *tt);
  auto state = routing::search_state{};
  auto const q = routing::query{
      .search_dir_ = direction::kForward,
      .interval_ = {unixtime_t{sys_days{2020_y / March / 30}} + 1_hours,
                    unixtime_t{sys_days{2020_y / March / 30}} + 2_hours},
      .start_ = {nigiri::routing::offset{
          .location_ = tt->locations_.location_id_to_idx_.at(
              {.id_ = "0000001", .src_ = src}),
          .offset_ = 0_minutes,
          .type_ = 0U}},
      .destinations_ = {{nigiri::routing::offset{
          .location_ = tt->locations_.location_id_to_idx_.at(
              {.id_ = "0000002", .src_ = src}),
          .offset_ = 0_minutes,
          .type_ = 0U}}},
      .via_destinations_ = {},
      .allowed_classes_ = bitset<kNumClasses>::max(),
      .max_transfers_ = nigiri::routing::kMaxTransfers,
      .min_connection_count_ = 0U,
      .extend_interval_earlier_ = false,
      .extend_interval_later_ = false};
  auto r = routing::raptor<direction::kForward>{tt, state, q};
  r.route();
  std::cerr << "num results: " << r.state_.results_.size() << "\n";
  for (auto const& x : r.state_.results_) {
    std::cerr << static_cast<int>(x.transfers_) << " " << x.dest_time_ << "\n";
  }
  std::cerr << "\n\n";
};
