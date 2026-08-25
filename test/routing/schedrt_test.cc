#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor/schedrt_criterion.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/util.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_literals;

namespace {

// A -> B, three direct trips (Europe/Berlin, June = UTC+2):
//   TRIP_C 10:00 -> 11:00 local (08:00Z -> 09:00Z), cancelled in rt
//   TRIP_D 12:00 -> 13:00 local (10:00Z -> 11:00Z), delayed +30min in rt
//   TRIP_N 14:00 -> 15:00 local (12:00Z -> 13:00Z), unchanged
mem_dir test_files() {
  return mem_dir::read(R"(
     "(
# agency.txt
agency_name,agency_url,agency_timezone,agency_lang,agency_phone,agency_id
test,https://test.com,Europe/Berlin,DE,0800123456,AGENCY_1

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,1.0,1.0
B,B,2.0,2.0

# calendar_dates.txt
service_id,date,exception_type
SERVICE_1,20240610,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
ROUTE_1,AGENCY_1,Route 1,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,
ROUTE_1,SERVICE_1,TRIP_C,B,,
ROUTE_1,SERVICE_1,TRIP_D,B,,
ROUTE_1,SERVICE_1,TRIP_N,B,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TRIP_C,10:00:00,10:00:00,A,1,0,0
TRIP_C,11:00:00,11:00:00,B,2,0,0
TRIP_D,12:00:00,12:00:00,A,1,0,0
TRIP_D,13:00:00,13:00:00,B,2,0,0
TRIP_N,14:00:00,14:00:00,A,1,0,0
TRIP_N,15:00:00,15:00:00,B,2,0,0
)");
}

auto const kUpdate =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1718000000"
 },
 "entity": [
  {
    "id": "1",
    "isDeleted": false,
    "tripUpdate": {
     "trip": {
      "tripId": "TRIP_C",
      "startTime": "10:00:00",
      "startDate": "20240610",
      "scheduleRelationship": "CANCELED"
     }
    }
  },
  {
    "id": "2",
    "isDeleted": false,
    "tripUpdate": {
     "trip": {
      "tripId": "TRIP_D",
      "startTime": "12:00:00",
      "startDate": "20240610"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "departure": {
        "delay": 1800
       }
      }
     ]
    }
  }
 ]
})"s;

template <direction Dir>
struct srt_test_algo
    : routing::basic_raptor<Dir,
                            true,
                            routing::schedrt_criterion<Dir>,
                            routing::search_mode::kOneToOne> {
  using base_t = routing::basic_raptor<Dir,
                                       true,
                                       routing::schedrt_criterion<Dir>,
                                       routing::search_mode::kOneToOne>;
  using base_t::base_t;
  static constexpr auto const kBothWorldStarts = true;
  static constexpr auto const kNResultSlots = std::uint8_t{2U};
};

template <direction Dir, bool Rt>
struct plain_test_algo
    : routing::raptor<Dir, Rt, 0U, routing::search_mode::kOneToOne> {
  using base_t = routing::raptor<Dir, Rt, 0U, routing::search_mode::kOneToOne>;
  using base_t::base_t;
};

template <direction Dir>
struct cod_test_algo
    : routing::basic_raptor<Dir,
                            true,
                            routing::schedrt_cod_criterion<Dir>,
                            routing::search_mode::kOneToOne> {
  using base_t = routing::basic_raptor<Dir,
                                       true,
                                       routing::schedrt_cod_criterion<Dir>,
                                       routing::search_mode::kOneToOne>;
  using base_t::base_t;
  static constexpr auto const kBothWorldStarts = true;
  static constexpr auto const kNResultSlots = std::uint8_t{2U};
};

pareto_set<routing::journey> run_cod(timetable const& tt,
                                     rt_timetable& rtt,
                                     routing::query const& q) {
  auto ss = routing::search_state{};
  auto rs = routing::raptor_state{};
  return *routing::search<direction::kForward,
                          cod_test_algo<direction::kForward>>{tt, &rtt, ss, rs,
                                                              q}
              .execute()
              .journeys_;
}

using jkey_t = std::tuple<std::uint8_t, unixtime_t, unixtime_t, std::uint8_t>;

std::vector<jkey_t> keys(pareto_set<routing::journey> const& js,
                        std::optional<std::uint8_t> const slot_override =
                            std::nullopt) {
  auto v = std::vector<jkey_t>{};
  for (auto const& j : js) {
    v.emplace_back(slot_override.value_or(j.slot_), j.start_time_,
                   j.dest_time_, j.transfers_);
  }
  std::sort(begin(v), end(v));
  return v;
}

routing::query make_query(timetable const& tt) {
  auto const src = source_idx_t{0};
  auto const t0 = unixtime_t{sys_days{2024_y / June / 10}} + 6h;
  auto const t1 = unixtime_t{sys_days{2024_y / June / 10}} + 14h;
  return routing::query{
      .start_time_ = interval<unixtime_t>{t0, t1},
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"A", src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({"B", src}),
                        0_minutes, 0U}}};
}

timetable make_tt() {
  auto tt = timetable{};
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / June / 8},
                    date::sys_days{2024_y / June / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  return tt;
}

}  // namespace

TEST(routing, schedrt_pong) {
  auto tt = make_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / June / 10});
  auto const stats = rt::gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "",
                                           rt::json_to_protobuf(kUpdate));
  ASSERT_EQ(2U, stats.total_entities_success_);

  auto const q = make_query(tt);

  auto ss = routing::search_state{};
  auto rs = routing::raptor_state{};
  auto const combined = *routing::pong_search_srt(tt, &rtt, ss, rs, q,
                                                  direction::kForward)
                             .journeys_;
  auto const off =
      *routing::pong_search(tt, nullptr, ss, rs, q, direction::kForward)
           .journeys_;
  auto const on = *routing::pong_search(tt, &rtt, ss, rs, q,
                                        direction::kForward)
                       .journeys_;

  auto combined_sched = std::vector<jkey_t>{};
  auto combined_rt = std::vector<jkey_t>{};
  for (auto const& k : keys(combined)) {
    (std::get<0>(k) == 0U ? combined_sched : combined_rt).push_back(k);
  }

  EXPECT_EQ(keys(off, std::uint8_t{0U}), combined_sched);
  EXPECT_EQ(keys(on, std::uint8_t{1U}), combined_rt);

  // copy-on-diverge pong: identical to the 2-slot combined pong
  auto ss2 = routing::search_state{};
  auto rs2 = routing::raptor_state{};
  auto const cod = *routing::pong_search_srt(tt, &rtt, ss2, rs2, q,
                                             direction::kForward, std::nullopt,
                                             /*copy_on_diverge=*/true)
                        .journeys_;
  EXPECT_EQ(keys(combined), keys(cod));
}

TEST(routing, schedrt_intermodal_extension_empty_rtt) {
  auto tt = make_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / June / 10});

  auto const src = source_idx_t{0};
  auto const day = unixtime_t{sys_days{2024_y / June / 10}};
  auto const q = routing::query{
      .start_time_ = interval<unixtime_t>{day + 6h, day + 7h},
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .use_start_footpaths_ = false,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"A", src}), 5_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({"B", src}),
                        7_minutes, 0U}},
      .min_connection_count_ = 2U,
      .extend_interval_later_ = true};

  auto ss = routing::search_state{};
  auto rs = routing::raptor_state{};
  auto const combined =
      *routing::search<direction::kForward, srt_test_algo<direction::kForward>>{
           tt, &rtt, ss, rs, q}
           .execute()
           .journeys_;
  auto const off =
      *routing::search<direction::kForward,
                       plain_test_algo<direction::kForward, false>>{
           tt, nullptr, ss, rs, q}
           .execute()
           .journeys_;

  auto combined_sched = std::vector<jkey_t>{};
  auto combined_rt = std::vector<jkey_t>{};
  for (auto const& k : keys(combined)) {
    (std::get<0>(k) == 0U ? combined_sched : combined_rt).push_back(k);
  }
  EXPECT_EQ(keys(off, std::uint8_t{0U}), combined_sched);
  EXPECT_EQ(keys(off, std::uint8_t{1U}), combined_rt);

  auto const cod = run_cod(tt, rtt, q);
  EXPECT_EQ(keys(combined), keys(cod));
}

TEST(routing, schedrt_intermodal_extension) {
  auto tt = make_tt();
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / June / 10});
  auto const stats = rt::gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "",
                                           rt::json_to_protobuf(kUpdate));
  ASSERT_EQ(2U, stats.total_entities_success_);

  auto const src = source_idx_t{0};
  auto const day = unixtime_t{sys_days{2024_y / June / 10}};
  auto const q = routing::query{
      .start_time_ = interval<unixtime_t>{day + 6h, day + 7h},
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .use_start_footpaths_ = false,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"A", src}), 5_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({"B", src}),
                        7_minutes, 0U}},
      .min_connection_count_ = 2U,
      .extend_interval_later_ = true};

  auto ss = routing::search_state{};
  auto rs = routing::raptor_state{};
  auto const combined =
      *routing::search<direction::kForward, srt_test_algo<direction::kForward>>{
           tt, &rtt, ss, rs, q}
           .execute()
           .journeys_;
  auto const off =
      *routing::search<direction::kForward,
                       plain_test_algo<direction::kForward, false>>{
           tt, nullptr, ss, rs, q}
           .execute()
           .journeys_;
  auto const on = *routing::search<direction::kForward,
                                   plain_test_algo<direction::kForward, true>>{
                       tt, &rtt, ss, rs, q}
                      .execute()
                      .journeys_;

  auto combined_sched = std::vector<jkey_t>{};
  auto combined_rt = std::vector<jkey_t>{};
  for (auto const& k : keys(combined)) {
    (std::get<0>(k) == 0U ? combined_sched : combined_rt).push_back(k);
  }

  // interval extension continues until BOTH worlds have min_connection_count
  // connections -> the laggard world (rt here: one trip cancelled) matches its
  // standalone run exactly, the early world may collect extra journeys in the
  // wider interval (superset)
  EXPECT_EQ(keys(on, std::uint8_t{1U}), combined_rt);
  auto const off_keys = keys(off, std::uint8_t{0U});
  EXPECT_TRUE(std::includes(begin(combined_sched), end(combined_sched),
                            begin(off_keys), end(off_keys)))
      << "standalone scheduled results must be contained in the combined "
         "scheduled slot";

  auto const cod = run_cod(tt, rtt, q);
  EXPECT_EQ(keys(combined), keys(cod));
}

TEST(routing, schedrt_combined_search) {
  auto tt = timetable{};
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / June / 8},
                    date::sys_days{2024_y / June / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / June / 10});
  auto const q = make_query(tt);

  {  // empty rtt: both slots must equal the scheduled-only search
    auto ss0 = routing::search_state{};
    auto rs0 = routing::raptor_state{};
    auto const c0 = *routing::search<direction::kForward,
                                     srt_test_algo<direction::kForward>>{
                         tt, &rtt, ss0, rs0, q}
                         .execute()
                         .journeys_;
    auto const off0 = *routing::search<direction::kForward,
                                       plain_test_algo<direction::kForward,
                                                       false>>{
                           tt, nullptr, ss0, rs0, q}
                           .execute()
                           .journeys_;
    auto s0 = std::vector<jkey_t>{};
    auto r0 = std::vector<jkey_t>{};
    for (auto const& k : keys(c0)) {
      (std::get<0>(k) == 0U ? s0 : r0).push_back(k);
    }
    EXPECT_EQ(keys(off0, std::uint8_t{0U}), s0);
    EXPECT_EQ(keys(off0, std::uint8_t{1U}), r0);
  }

  auto const stats = rt::gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "",
                                           rt::json_to_protobuf(kUpdate));
  ASSERT_EQ(2U, stats.total_entities_success_);

  // combined scheduled+rt search: one sweep, two slots
  auto ss = routing::search_state{};
  auto rs = routing::raptor_state{};
  auto const combined =
      *routing::search<direction::kForward, srt_test_algo<direction::kForward>>{
           tt, &rtt, ss, rs, q}
           .execute()
           .journeys_;

  // reference: two separate searches
  auto const off =
      *routing::search<direction::kForward,
                       plain_test_algo<direction::kForward, false>>{
           tt, nullptr, ss, rs, q}
           .execute()
           .journeys_;
  auto const on = *routing::search<direction::kForward,
                                   plain_test_algo<direction::kForward, true>>{
                       tt, &rtt, ss, rs, q}
                       .execute()
                       .journeys_;

  auto const day = unixtime_t{sys_days{2024_y / June / 10}};
  auto const expected_sched = std::vector<jkey_t>{
      {0U, day + 8h, day + 9h, 0U},  // TRIP_C: cancelled, scheduled world only
      {0U, day + 10h, day + 11h, 0U},  // TRIP_D at scheduled times
      {0U, day + 12h, day + 13h, 0U},  // TRIP_N
  };
  auto const expected_rt = std::vector<jkey_t>{
      {1U, day + 10h + 30min, day + 11h + 30min, 0U},  // TRIP_D delayed
      {1U, day + 12h, day + 13h, 0U},  // TRIP_N
  };

  auto combined_sched = std::vector<jkey_t>{};
  auto combined_rt = std::vector<jkey_t>{};
  for (auto const& k : keys(combined)) {
    (std::get<0>(k) == 0U ? combined_sched : combined_rt).push_back(k);
  }

  EXPECT_EQ(expected_sched, combined_sched);
  EXPECT_EQ(expected_rt, combined_rt);

  // combined slots == the two separate searches
  EXPECT_EQ(keys(off, std::uint8_t{0U}), combined_sched);
  EXPECT_EQ(keys(on, std::uint8_t{1U}), combined_rt);

  // reconstruct: every combined journey's legs must match the legs of the
  // corresponding standalone journey of its world
  auto const find_ref = [](pareto_set<routing::journey> const& ref,
                           routing::journey const& j)
      -> routing::journey const* {
    for (auto const& r : ref) {
      if (r.start_time_ == j.start_time_ && r.dest_time_ == j.dest_time_ &&
          r.transfers_ == j.transfers_) {
        return &r;
      }
    }
    return nullptr;
  };
  for (auto const& j : combined) {
    EXPECT_FALSE(j.legs_.empty());
    auto const* ref = find_ref(j.slot_ == 0U ? off : on, j);
    ASSERT_NE(nullptr, ref);
    EXPECT_EQ(ref->legs_, j.legs_);
  }

  // copy-on-diverge: same criteria and legs as the 2-slot combined search
  auto const cod = run_cod(tt, rtt, q);
  EXPECT_EQ(keys(combined), keys(cod));
  for (auto const& j : cod) {
    EXPECT_FALSE(j.legs_.empty());
    auto const* ref = find_ref(j.slot_ == 0U ? off : on, j);
    ASSERT_NE(nullptr, ref);
    EXPECT_EQ(ref->legs_, j.legs_);
  }
}
