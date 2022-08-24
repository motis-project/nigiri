#include "doctest/doctest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/routing/raptor.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace nigiri;

constexpr auto const services = R"(
*Z 01337 80____       048 030                             %
*A VE 0000001 0000002 000005                              %
*G RE  0000001 0000002                                    %
0000001 A                            00230                %
0000002 B                     00330                       %
*Z 07331 80____       092 015                             %
*A VE 0000002 0000003 000005                              %
*G RE  0000002 0000003                                    %
0000002 B                            00230                %
0000003 C                     00330                       %
)";

TEST_CASE("raptor, simple_search") {
  using namespace date;
  auto tt = std::make_shared<timetable>();
  auto const src = source_idx_t{0U};
  load_timetable(
      src, loader::hrd::hrd_5_20_26,
      test_data::hrd_timetable::base().add(
          {loader::hrd::hrd_5_20_26.fplan_ / "services.101", services}),
      *tt);
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
    std::cerr << "transfers=" << static_cast<int>(x.transfers_) << ", "
              << x.start_time_ << " -> " << x.dest_time_ << "\n";
  }
  std::cerr << "\n\n";
};
