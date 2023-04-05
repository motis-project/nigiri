#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/routing/ontrip_train.h"
#include "nigiri/routing/raptor.h"
#include "nigiri/routing/search_state.h"
#include "nigiri/timetable.h"

#include "nigiri/print_transport.h"
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
      fmt::print("  no trip with id=\"{}\" found\n", trip_idx);
      return std::nullopt;
    }

    auto const day_idx = tt.day_idx(day);
    for (auto it = lb; it != end(tt.trip_id_to_idx_); ++it) {
      auto const t = tt.trip_ref_transport_[it->second].first;
      auto const& traffic_days = tt.bitfields_[tt.transport_traffic_days_[t]];
      if (traffic_days.test(to_idx(day_idx))) {
        return transport{.t_idx_ = t, .day_ = day_idx};
      }
      auto const reverse = [](std::string s) {
        std::reverse(s.begin(), s.end());
        return s;
      };
      fmt::print("  trip={} not active on day\n", t,
                 tt.to_unixtime(day_idx, 0_minutes));
      auto const range = tt.internal_interval_days();
      std::cout << "TRAFFIC_DAYS="
                << reverse(traffic_days.to_string().substr(
                       traffic_days.size() -
                       static_cast<size_t>((range.size() + 2_days) / 1_days)))
                << "\n";
      for (auto d = range.from_; d != range.to_; d += std::chrono::days{1}) {
        auto const x = day_idx_t{
            static_cast<day_idx_t::value_t>((d - range.from_) / 1_days)};
        if (traffic_days.test(to_idx(x))) {
          date::to_stream(std::cout, "%F", d);
          std::cout << " (day_idx=" << x << ")\n";
          print_transport(tt, std::cout, {t, x});
        }
      }
    }
    return std::nullopt;
  };
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
      get_transport("3374/0000008/1410/0000006/2950/", March / 30 / 2020);
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
