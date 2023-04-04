#include "doctest/doctest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/routing/ontrip_train.h"
#include "nigiri/timetable.h"

#include "../loader/hrd/hrd_timetable.h"

using namespace nigiri;
using namespace nigiri::routing;
using namespace nigiri::test_data::hrd_timetable;
using namespace date;

TEST_CASE("routing.ontrip_train") {
  using namespace date;
  timetable tt;
  tt.date_range_ = full_period();
  constexpr auto const src = source_idx_t{0U};
  load_timetable(src, loader::hrd::hrd_5_20_26, files(), tt);

  auto const get_transport_idx =
      [&](std::string_view trip_idx) -> std::optional<transport_idx_t> {
    auto const lb = std::lower_bound(
        begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), trip_idx,
        [&](pair<trip_id_idx_t, trip_idx_t> const& a, std::string_view b) {
          return tt.trip_id_strings_[a.first].view() < b;
        });
    if (lb == end(tt.trip_id_to_idx_) ||
        tt.trip_id_strings_[lb->first].view() != trip_idx) {
      return std::nullopt;
    }
    return tt.trip_ref_transport_[lb->second].first;
  };
  for (auto const& [trip_id, trip_idx] : tt.trip_id_to_idx_) {
    fmt::print("trip_id=\"{}\" -> trip_idx={}\n",
               tt.trip_id_strings_[trip_id].view(), trip_idx);
  }

  auto const q = generate_ontrip_train_query(
      tt,
      transport{get_transport_idx("").value(), tt.day_idx(April / 4 / 2023)}, 2,
      routing::query{
          .start_time_ = {},
          .start_match_mode_ =
              nigiri::routing::location_match_mode::kIntermodal,
          .dest_match_mode_ = nigiri::routing::location_match_mode::kIntermodal,
          .use_start_footpaths_ = true,
          .start_ = {nigiri::routing::offset{
              tt.locations_.location_id_to_idx_.at(
                  {.id_ = "0000003", .src_ = src}),
              15_minutes, 99U}},
          .destinations_ = {{{tt.locations_.location_id_to_idx_.at(
                                  {.id_ = "0000001", .src_ = src}),
                              10_minutes, 77U}}},
          .via_destinations_ = {},
          .allowed_classes_ = bitset<kNumClasses>::max(),
          .max_transfers_ = 6U,
          .min_connection_count_ = 3U,
          .extend_interval_earlier_ = true,
          .extend_interval_later_ = true});
  /*
  auto state = routing::search_state{};
  auto fwd_r = routing::raptor<direction::kForward, true>{tt, state, q};
  fwd_r.route();
  */
}