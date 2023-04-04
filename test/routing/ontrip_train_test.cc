#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/routing/ontrip_train.h"
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

  auto const get_transport =
      [&](std::string_view trip_idx,
          date::year_month_day const day) -> std::optional<transport> {
    auto const lb = std::lower_bound(
        begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), trip_idx,
        [&](pair<trip_id_idx_t, trip_idx_t> const& a, std::string_view b) {
          return tt.trip_id_strings_[a.first].view() < b;
        });
    if (lb == end(tt.trip_id_to_idx_) ||
        tt.trip_id_strings_[lb->first].view() != trip_idx) {
      return std::nullopt;
    }

    auto const day_idx = tt.day_idx(day);
    for (auto it = lb; it != end(tt.trip_id_to_idx_); ++it) {
      auto const t = tt.trip_ref_transport_[it->second].first;
      auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];
      if (traffic_days.test(to_idx(day_idx))) {
        return transport{.t_idx_ = t, .day_ = day_idx};
      }
    }
    return std::nullopt;
  };
  for (auto const& [trip_id, trip_idx] : tt.trip_id_to_idx_) {
    fmt::print("trip_id=\"{}\" -> trip_idx={}\n",
               tt.trip_id_strings_[trip_id].view(), trip_idx);
  }

  auto const q = generate_ontrip_train_query(
      tt,
      get_transport("1337/0000001/1260/0000004/2586/", March / 28 / 2020)
          .value(),
      2,
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
