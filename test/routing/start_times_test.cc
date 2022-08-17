#include "doctest/doctest.h"

#include <iostream>

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/routing/start_times.h"
#include "../loader/hrd/hrd_timetable.h"

using namespace nigiri;
using namespace nigiri::routing;

constexpr auto const service = R"(
*Z 01337 80____       012 030                             %
*A VE 0000001 0000002 000005                              %
*G ICE 0000001 0000006                                    %
0000001 A                            00230                %
0000002 B                     00330                       %
*Z 07331 80____       024 015                             %
*A VE 0000001 0000002 000005                              %
*G ICE 0000001 0000006                                    %
0000001 A                            00230                %
0000002 B                     00330                       %
)";

//   +---15--- 30freq --> A
//  /
// O-----30--  15freq --> B
//
// A:         [15-30],          [45-60],          [75-90], ...
// B: [0-30], [15-45], [30-60], [45-75], [60-90], [75-115], ...
TEST_CASE("routing start times") {
  //  auto const tt = nigiri::loader::hrd::load_timetable(
  //      nigiri::loader::hrd::hrd_5_20_26,
  //      nigiri::test_data::hrd_timetable::files(),
  //      {nigiri::test_data::hrd_timetable::service_file_content});
  //
  //  using namespace date;
  //  auto const A = tt->locations_.location_id_to_idx_.at(
  //      location_id{.id_ = "0000001", .src_ = source_idx_t{0}});
  //  auto const B = tt->locations_.location_id_to_idx_.at(
  //      location_id{.id_ = "0000002", .src_ = source_idx_t{0}});
  //  auto const starts = get_starts<nigiri::direction::kForward>(
  //      *tt,
  //      interval{sys_days{2020_y / March / 30}, sys_days{2020_y / March /
  //      31}}, {offset{.location_ = A, .offset_ = 15_minutes, .type_ = 0},
  //       offset{.location_ = B, .offset_ = 30_minutes, .type_ = 0}});
  //  for (auto const& s : starts) {
  //    std::cout << s << "\n";
  //  }
}