#include "nigiri/routing/tb/tb_a_star/a_star.h"

#include "nigiri/routing/tb/preprocess.h"

#include <nigiri/routing/search.h>
#include <nigiri/special_stations.h>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

namespace nigiri::routing::tb::a_star {

using namespace date;
using namespace std::chrono_literals;

timetable load_gtfs(auto const& files) {
  timetable tt;
  tt.date_range_ = {sys_days{2021_y / March / 1}, sys_days{2021_y / March / 8}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0}, files(), tt);
  loader::finalize(tt);
  return tt;
}

void check_neighbours(tb_data const& tbd) {
  for (segment_idx_t i(0); i < tbd.segment_transports_.size(); ++i) {
    auto [neighbours, next_exists] = get_neighbours(i, tbd, day_idx_t{0});
    auto end = next_exists ? neighbours.size() - 1 : neighbours.size();
    std::vector seen(tbd.segment_transports_.size(), false);
    for (int j = 0; j < end; ++j) {
      auto neighbour = neighbours[j];
      EXPECT_TRUE(utl::any_of(tbd.segment_transfers_[i],
                              [neighbour](transfer const& t) {
                                return t.to_segment_ == neighbour;
                              }));
      EXPECT_FALSE(seen[neighbour.v_]);
      seen[neighbour.v_] = true;
    }
    if (next_exists) {
      EXPECT_EQ(i + 1, neighbours.back());
      EXPECT_EQ(tbd.segment_transports_[i], tbd.segment_transports_[i + 1]);
    }
  }
}

std::vector<segment_idx_t> get_segments(tb_a_star const& a_star_exec) {
  std::vector<segment_idx_t> result;
  auto p = a_star_exec.end_segment_;
  auto invalid_segment = segment_idx_t::invalid();
  while (p != invalid_segment) {
    result.push_back(p);
    p = a_star_exec.pred_[p];
  }
  std::ranges::reverse(result);
  return result;
}

loader::mem_dir no_transfer_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,
S5,S5,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S4",2
R1,DTA,R1,R1,"S2 -> S5 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,02:00:00,02:00:00,S2,2,0,0
R0_MON,03:00:00,03:00:00,S3,3,0,0
R0_MON,04:00:00,04:00:00,S4,4,0,0
R1_MON,02:00:10,02:10:00,S2,0,0,0
R1_MON,02:30:00,02:30:00,S5,1,0,0
R1_MON,02:50:00,02:50:00,S3,2,0,0
)");
}

TEST(tb_a_star, a_star_no_transfer) {
  auto const tt = load_gtfs(no_transfer_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  auto start_segment = tbd.transport_first_segment_[transport_idx_t(0)];
  auto end_segment = tbd.transport_first_segment_[transport_idx_t(1)] - 1;
  auto day_idx = tt.day_idx(year_month_day{year{2021}, March, day{1}});
  unixtime_t start_time = sys_days{year{2021} / March / day{1}};
  tb_a_star a_star_exec(tbd.segment_transfers_.size());
  a_star_exec.state_.tbd_ = tbd;
  a_star_exec.tt_ = tt;
  a_star_exec.travel_time_lower_bound_ =
      std::vector<uint16_t>(tt.n_locations(), 0);
  a_star_exec.queue_.push(std::pair(start_segment, duration_t(0)));
  a_star_exec.state_.end_reachable_.resize(tbd.segment_transports_.size());
  a_star_exec.state_.end_reachable_.set(end_segment);
  a_star_exec.transfers_[start_segment] = 0;
  a_star_exec.day_idx_[start_segment] =
      day_idx;  // get_arrival_day(start_time,start_segment,tt,tbd);
  a_star_exec.state_.dist_to_dest_ =
      hash_map<segment_idx_t, duration_t>(tbd.segment_transports_.size());

  auto const it = a_star_exec.state_.dist_to_dest_.find(end_segment);
  if (it == end(a_star_exec.state_.dist_to_dest_)) {
    a_star_exec.state_.dist_to_dest_.emplace_hint(it, end_segment,
                                                  duration_t::zero());
  } else {
    it->second = duration_t::zero();
  }

  pareto_set<journey> journeys;
  a_star_exec.execute(start_time, std::numeric_limits<std::uint8_t>::max(),
                      unixtime_t::max(), profile_idx_t{0}, journeys);
  auto result = get_segments(a_star_exec);
  std::vector<segment_idx_t> expected;
  expected.emplace_back(0);
  expected.emplace_back(1);
  expected.emplace_back(2);
  expected.emplace_back(3);
  EXPECT_EQ(result, expected);
  EXPECT_EQ(a_star_exec.end_segment_, end_segment);

  for (auto const& segment : result) {
    EXPECT_EQ(a_star_exec.day_idx_[segment], day_idx);
    EXPECT_EQ(a_star_exec.transfers_[segment], 0);
  }

  EXPECT_EQ(journeys.els_.size(), 1);
  journey journey = journeys.els_[0];
  EXPECT_TRUE(journey.legs_.empty());
  EXPECT_EQ(journey.start_time_, start_time);
  EXPECT_EQ(journey.transfers_, 0);
  EXPECT_EQ(journey.dest_, location_idx_t::invalid());
  EXPECT_EQ(journey.dest_time_, start_time + 4h);

  a_star_exec.reconstruct(
      query{.start_time_ = start_time,
            .start_match_mode_ = location_match_mode::kEquivalent,
            .dest_match_mode_ = location_match_mode::kEquivalent,
            .start_ = {{tt.locations_.location_id_to_idx_.at(
                            {"S0", source_idx_t{0}}),
                        0_minutes, 0U}},
            .destination_ = {{tt.locations_.location_id_to_idx_.at(
                                  {"S4", source_idx_t{0}}),
                              0_minutes, 0U}}},
      journey);
  EXPECT_EQ(journey.legs_.size(), 2);
  EXPECT_FALSE(journey.error_);
  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .back()}
          .location_idx();
  EXPECT_EQ(journey.dest_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(journey.legs_.front().from_, start);

  for (int i = 0; i < journey.legs_.size() - 1; ++i) {
    EXPECT_EQ(journey.legs_[i].to_, journey.legs_[i + 1].from_);
  }
  EXPECT_EQ(journey.legs_.back().to_, journey.dest_);
  EXPECT_EQ(journey.transfers_, 0);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 4h);
  EXPECT_EQ(journey.legs_.back().dep_time_, start_time + 4h);
  EXPECT_EQ(journey.legs_.back().arr_time_, start_time + 4h);

  check_neighbours(tbd);
}

loader::mem_dir with_transfer_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S2",2
R1,DTA,R1,R1,"S1 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,02:00:00,02:00:00,S2,2,0,0
R1_MON,01:10:00,01:10:00,S1,0,0,0
R1_MON,02:00:00,02:00:00,S3,1,0,0
)");
}

TEST(tb_a_star, a_star_with_transfer) {
  auto const tt = load_gtfs(with_transfer_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  auto start_segment = tbd.transport_first_segment_[transport_idx_t(0)];
  auto end_segment = segment_idx_t(tbd.segment_transports_.size() - 1);
  auto day_idx = tt.day_idx(year_month_day{year{2021}, March, day{1}});
  unixtime_t start_time = sys_days{year{2021} / March / day{1}};
  tb_a_star a_star_exec(tbd.segment_transfers_.size());
  a_star_exec.state_.tbd_ = tbd;
  a_star_exec.tt_ = tt;
  a_star_exec.travel_time_lower_bound_ =
      std::vector<uint16_t>(tt.n_locations(), 0);
  a_star_exec.queue_.push(std::pair(start_segment, duration_t(0)));
  a_star_exec.state_.end_reachable_.resize(tbd.segment_transports_.size());
  a_star_exec.state_.end_reachable_.set(end_segment);
  a_star_exec.transfers_[start_segment] = 0;
  a_star_exec.day_idx_[start_segment] =
      day_idx;  // get_arrival_day(start_time,start_segment,tt,tbd);
  a_star_exec.state_.dist_to_dest_ =
      hash_map<segment_idx_t, duration_t>(tbd.segment_transports_.size());

  auto const it = a_star_exec.state_.dist_to_dest_.find(end_segment);
  if (it == end(a_star_exec.state_.dist_to_dest_)) {
    a_star_exec.state_.dist_to_dest_.emplace_hint(it, end_segment,
                                                  duration_t::zero());
  } else {
    it->second = duration_t::zero();
  }

  pareto_set<journey> journeys;
  a_star_exec.execute(start_time, std::numeric_limits<std::uint8_t>::max(),
                      unixtime_t::max(), profile_idx_t{0}, journeys);
  auto result = get_segments(a_star_exec);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], start_segment);
  EXPECT_EQ(result[1], end_segment);

  for (auto const& segment : result) {
    EXPECT_EQ(a_star_exec.day_idx_[segment], day_idx);
  }
  EXPECT_EQ(a_star_exec.transfers_[start_segment], 0);
  EXPECT_EQ(a_star_exec.transfers_[end_segment], 1);
  EXPECT_EQ(a_star_exec.end_segment_, end_segment);

  EXPECT_EQ(journeys.els_.size(), 1);
  journey journey = journeys.els_[0];
  EXPECT_TRUE(journey.legs_.empty());
  EXPECT_EQ(journey.start_time_, start_time);
  EXPECT_EQ(journey.transfers_, 1);
  EXPECT_EQ(journey.dest_, location_idx_t::invalid());
  EXPECT_EQ(journey.dest_time_, start_time + 2h);

  a_star_exec.reconstruct(
      query{.start_time_ = start_time,
            .start_match_mode_ = location_match_mode::kEquivalent,
            .dest_match_mode_ = location_match_mode::kEquivalent,
            .start_ = {{tt.locations_.location_id_to_idx_.at(
                            {"S0", source_idx_t{0}}),
                        0_minutes, 0U}},
            .destination_ = {{tt.locations_.location_id_to_idx_.at(
                                  {"S3", source_idx_t{0}}),
                              0_minutes, 0U}}},
      journey);
  EXPECT_EQ(journey.legs_.size(), 4);
  EXPECT_FALSE(journey.error_);
  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{1}]]
               .back()}
          .location_idx();
  EXPECT_EQ(journey.dest_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(journey.legs_.front().from_, start);
  for (int i = 0; i < journey.legs_.size() - 1; ++i) {
    EXPECT_EQ(journey.legs_[i].to_, journey.legs_[i + 1].from_);
  }
  EXPECT_EQ(journey.legs_.back().to_, journey.dest_);
  EXPECT_EQ(journey.transfers_, 1);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 1h);
  EXPECT_EQ(journey.legs_[1].dep_time_, start_time + 1h);
  EXPECT_EQ(journey.legs_[1].arr_time_, start_time + 1h + 2min);
  EXPECT_EQ(journey.legs_[2].dep_time_, start_time + 1h + 10min);
  EXPECT_EQ(journey.legs_[2].arr_time_, start_time + 2h);
  EXPECT_EQ(journey.legs_[3].dep_time_, start_time + 2h);
  EXPECT_EQ(journey.legs_[3].arr_time_, start_time + 2h);

  check_neighbours(tbd);
}

loader::mem_dir longer_trips_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,
S4,S4,,,,,,
S5,S5,,,,,,
S6,S6,,,,,,
S7,S7,,,,,,
S8,S8,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S4",2
R1,DTA,R1,R1,"S3 -> S6 -> S8",2
R2,DTA,R2,R2,"S6 -> S7",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,1
R2,MON,R2_MON,R2_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,02:00:00,02:00:00,S2,2,0,0
R0_MON,03:00:00,03:00:00,S3,3,0,0
R0_MON,04:00:00,04:00:00,S4,4,0,0
R1_MON,03:10:00,03:10:00,S3,0,0,0
R1_MON,04:00:00,04:00:00,S5,1,0,0
R1_MON,05:00:00,05:00:00,S6,2,0,0
R1_MON,06:00:00,06:00:00,S8,3,0,0
R2_MON,05:10:00,05:10:00,S6,0,0,0
R2_MON,06:00:00,06:00:00,S7,1,0,0
)");
}

TEST(tb_a_star, a_star_longer_routes) {
  auto const tt = load_gtfs(longer_trips_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  auto start_segment = tbd.transport_first_segment_[transport_idx_t(0)];
  auto end_segment = segment_idx_t(tbd.segment_transports_.size() - 1);
  auto day_idx = tt.day_idx(year_month_day{year{2021}, March, day{1}});
  unixtime_t start_time = sys_days{year{2021} / March / day{1}};
  tb_a_star a_star_exec(tbd.segment_transfers_.size());
  a_star_exec.state_.tbd_ = tbd;
  a_star_exec.tt_ = tt;
  a_star_exec.travel_time_lower_bound_ =
      std::vector<uint16_t>(tt.n_locations(), 0);
  a_star_exec.queue_.push(std::pair(start_segment, duration_t(0)));
  a_star_exec.state_.end_reachable_.resize(tbd.segment_transports_.size());
  a_star_exec.state_.end_reachable_.set(end_segment);
  a_star_exec.transfers_[start_segment] = 0;
  a_star_exec.day_idx_[start_segment] =
      day_idx;  // get_arrival_day(start_time,start_segment,tt,tbd);
  a_star_exec.state_.dist_to_dest_ =
      hash_map<segment_idx_t, duration_t>(tbd.segment_transports_.size());

  auto const it = a_star_exec.state_.dist_to_dest_.find(end_segment);
  if (it == end(a_star_exec.state_.dist_to_dest_)) {
    a_star_exec.state_.dist_to_dest_.emplace_hint(it, end_segment,
                                                  duration_t::zero());
  } else {
    it->second = duration_t::zero();
  }

  pareto_set<journey> journeys;
  a_star_exec.execute(start_time, std::numeric_limits<std::uint8_t>::max(),
                      unixtime_t::max(), profile_idx_t{0}, journeys);
  auto result = get_segments(a_star_exec);
  std::vector<segment_idx_t> expected(6);
  expected[0] = start_segment;  // S0 to S1
  expected[1] = start_segment + 1;  // S1 to S2
  expected[2] = start_segment + 2;  // S2 to S3
  expected[3] = tbd.transport_first_segment_[transport_idx_t(
      1)];  // S3 to S5 in the next transport
  expected[4] = expected[3] + 1;  // S5 to S6
  expected[5] = end_segment;  // S6 to S7
  EXPECT_EQ(expected, result);
  EXPECT_EQ(a_star_exec.end_segment_, end_segment);

  for (auto const& segment : result) {
    EXPECT_EQ(a_star_exec.day_idx_[segment], day_idx);
  }
  EXPECT_EQ(a_star_exec.transfers_[start_segment], 0);
  EXPECT_EQ(a_star_exec.transfers_[start_segment + 1], 0);
  EXPECT_EQ(a_star_exec.transfers_[start_segment + 2], 0);
  EXPECT_EQ(a_star_exec.transfers_[expected[3]], 1);
  EXPECT_EQ(a_star_exec.transfers_[expected[4]], 1);
  EXPECT_EQ(a_star_exec.transfers_[end_segment], 2);

  EXPECT_EQ(journeys.els_.size(), 1);
  journey journey = journeys.els_[0];
  EXPECT_TRUE(journey.legs_.empty());
  EXPECT_EQ(journey.start_time_, start_time);
  EXPECT_EQ(journey.transfers_, 2);
  EXPECT_EQ(journey.dest_, location_idx_t::invalid());
  EXPECT_EQ(journey.dest_time_, start_time + 6h);

  a_star_exec.reconstruct(
      query{.start_time_ = start_time,
            .start_match_mode_ = location_match_mode::kEquivalent,
            .dest_match_mode_ = location_match_mode::kEquivalent,
            .start_ = {{tt.locations_.location_id_to_idx_.at(
                            {"S0", source_idx_t{0}}),
                        0_minutes, 0U}},
            .destination_ = {{tt.locations_.location_id_to_idx_.at(
                                  {"S7", source_idx_t{0}}),
                              0_minutes, 0U}}},
      journey);
  EXPECT_EQ(journey.legs_.size(), 6);
  EXPECT_FALSE(journey.error_);
  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{2}]]
               .back()}
          .location_idx();
  EXPECT_EQ(journey.dest_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(journey.legs_.front().from_, start);
  for (int i = 0; i < journey.legs_.size() - 1; ++i) {
    EXPECT_EQ(journey.legs_[i].to_, journey.legs_[i + 1].from_);
  }
  EXPECT_EQ(journey.legs_.back().to_, journey.dest_);
  EXPECT_EQ(journey.transfers_, 2);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 3h);
  EXPECT_EQ(journey.legs_[1].dep_time_, start_time + 3h);
  EXPECT_EQ(journey.legs_[1].arr_time_, start_time + 3h + 2min);
  EXPECT_EQ(journey.legs_[2].dep_time_, start_time + 3h + 10min);
  EXPECT_EQ(journey.legs_[2].arr_time_, start_time + 5h);
  EXPECT_EQ(journey.legs_[3].dep_time_, start_time + 5h);
  EXPECT_EQ(journey.legs_[3].arr_time_, start_time + 5h + 2min);
  EXPECT_EQ(journey.legs_[4].dep_time_, start_time + 5h + 10min);
  EXPECT_EQ(journey.legs_[4].arr_time_, start_time + 6h);
  EXPECT_EQ(journey.legs_[5].dep_time_, start_time + 6h);
  EXPECT_EQ(journey.legs_[5].arr_time_, start_time + 6h);

  check_neighbours(tbd);
}

loader::mem_dir overnight_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307
TUE,0,1,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,TUE,R1_TUE,R1_TUE,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,10:00:00,10:00:00,S0,0,0,0
R0_MON,12:00:00,12:00:00,S1,1,0,0
R1_TUE,00:00:00,00:00:00,S1,0,0,0
R1_TUE,08:00:00,08:00:00,S2,1,0,0
)");
}

TEST(tb_a_star, overnight_travel) {
  auto const tt = load_gtfs(overnight_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  auto start_segment = tbd.transport_first_segment_[transport_idx_t(0)];
  auto end_segment = segment_idx_t(tbd.segment_transports_.size() - 1);
  auto day_idx = tt.day_idx(year_month_day{year{2021}, March, day{1}});
  unixtime_t start_time = sys_days{year{2021} / March / day{1}} + 10h;
  tb_a_star a_star_exec(tbd.segment_transfers_.size());
  a_star_exec.state_.tbd_ = tbd;
  a_star_exec.tt_ = tt;
  a_star_exec.travel_time_lower_bound_ =
      std::vector<uint16_t>(tt.n_locations(), 0);
  a_star_exec.queue_.push(std::pair(start_segment, duration_t(0)));
  a_star_exec.state_.end_reachable_.resize(tbd.segment_transports_.size());
  a_star_exec.state_.end_reachable_.set(end_segment);
  a_star_exec.transfers_[start_segment] = 0;
  a_star_exec.day_idx_[start_segment] = day_idx;
  a_star_exec.state_.dist_to_dest_ =
      hash_map<segment_idx_t, duration_t>(tbd.segment_transports_.size());

  auto const it = a_star_exec.state_.dist_to_dest_.find(end_segment);
  if (it == end(a_star_exec.state_.dist_to_dest_)) {
    a_star_exec.state_.dist_to_dest_.emplace_hint(it, end_segment,
                                                  duration_t::zero());
  } else {
    it->second = duration_t::zero();
  }

  pareto_set<journey> journeys;
  a_star_exec.execute(start_time, std::numeric_limits<std::uint8_t>::max(),
                      unixtime_t::max(), profile_idx_t{0}, journeys);
  auto result = get_segments(a_star_exec);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], start_segment);
  EXPECT_EQ(result[1], end_segment);

  EXPECT_EQ(a_star_exec.day_idx_[start_segment], day_idx);
  EXPECT_EQ(a_star_exec.day_idx_[end_segment], day_idx + 1);

  EXPECT_EQ(a_star_exec.transfers_[start_segment], 0);
  EXPECT_EQ(a_star_exec.transfers_[end_segment], 1);
  EXPECT_EQ(a_star_exec.end_segment_, end_segment);

  EXPECT_EQ(journeys.els_.size(), 1);
  journey journey = journeys.els_[0];
  EXPECT_TRUE(journey.legs_.empty());
  EXPECT_EQ(journey.start_time_, start_time);
  EXPECT_EQ(journey.transfers_, 1);
  EXPECT_EQ(journey.dest_, location_idx_t::invalid());
  EXPECT_EQ(journey.dest_time_, start_time + 22h);

  a_star_exec.reconstruct(
      query{.start_time_ = start_time,
            .start_match_mode_ = location_match_mode::kEquivalent,
            .dest_match_mode_ = location_match_mode::kEquivalent,
            .start_ = {{tt.locations_.location_id_to_idx_.at(
                            {"S0", source_idx_t{0}}),
                        0_minutes, 0U}},
            .destination_ = {{tt.locations_.location_id_to_idx_.at(
                                  {"S2", source_idx_t{0}}),
                              0_minutes, 0U}}},
      journey);
  EXPECT_EQ(journey.legs_.size(), 4);
  EXPECT_FALSE(journey.error_);
  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{1}]]
               .back()}
          .location_idx();
  EXPECT_EQ(journey.dest_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(journey.legs_.front().from_, start);
  for (int i = 0; i < journey.legs_.size() - 1; ++i) {
    EXPECT_EQ(journey.legs_[i].to_, journey.legs_[i + 1].from_);
  }
  EXPECT_EQ(journey.legs_.back().to_, journey.dest_);
  EXPECT_EQ(journey.transfers_, 1);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 2h);
  EXPECT_EQ(journey.legs_[1].dep_time_, start_time + 2h);
  EXPECT_EQ(journey.legs_[1].arr_time_, start_time + 2h + 2min);
  EXPECT_EQ(journey.legs_[2].dep_time_, start_time + 14h);
  EXPECT_EQ(journey.legs_[2].arr_time_, start_time + 22h);
  EXPECT_EQ(journey.legs_[3].dep_time_, start_time + 22h);
  EXPECT_EQ(journey.legs_[3].arr_time_, start_time + 22h);

  check_neighbours(tbd);
}

loader::mem_dir overnight_files_without_transfer() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,10:00:00,10:00:00,S0,0,0,0
R0_MON,12:00:00,12:00:00,S1,1,0,0
R0_MON,24:00:00,24:00:00,S2,2,0,0
)");
}
TEST(tb_a_star, overnight_travel_without_transfer) {
  auto const tt = load_gtfs(overnight_files_without_transfer);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  auto start_segment = tbd.transport_first_segment_[transport_idx_t(0)];
  auto end_segment = segment_idx_t(tbd.segment_transports_.size() - 1);
  auto day_idx = tt.day_idx(year_month_day{year{2021}, March, day{1}});
  unixtime_t start_time = sys_days{year{2021} / March / day{1}} + 10h;
  tb_a_star a_star_exec(tbd.segment_transfers_.size());
  a_star_exec.state_.tbd_ = tbd;
  a_star_exec.tt_ = tt;
  a_star_exec.travel_time_lower_bound_ =
      std::vector<uint16_t>(tt.n_locations(), 0);
  a_star_exec.queue_.push(std::pair(start_segment, duration_t(0)));
  a_star_exec.state_.end_reachable_.resize(tbd.segment_transports_.size());
  a_star_exec.state_.end_reachable_.set(end_segment);
  a_star_exec.transfers_[start_segment] = 0;
  a_star_exec.day_idx_[start_segment] = day_idx;
  a_star_exec.state_.dist_to_dest_ =
      hash_map<segment_idx_t, duration_t>(tbd.segment_transports_.size());

  auto const it = a_star_exec.state_.dist_to_dest_.find(end_segment);
  if (it == end(a_star_exec.state_.dist_to_dest_)) {
    a_star_exec.state_.dist_to_dest_.emplace_hint(it, end_segment,
                                                  duration_t::zero());
  } else {
    it->second = duration_t::zero();
  }

  pareto_set<journey> journeys;
  a_star_exec.execute(start_time, std::numeric_limits<std::uint8_t>::max(),
                      unixtime_t::max(), profile_idx_t{0}, journeys);
  auto result = get_segments(a_star_exec);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], start_segment);
  EXPECT_EQ(result[1], end_segment);

  EXPECT_EQ(a_star_exec.day_idx_[start_segment], day_idx);
  EXPECT_EQ(a_star_exec.day_idx_[end_segment], day_idx);

  EXPECT_EQ(a_star_exec.transfers_[start_segment], 0);
  EXPECT_EQ(a_star_exec.transfers_[end_segment], 0);
  EXPECT_EQ(a_star_exec.end_segment_, end_segment);

  EXPECT_EQ(journeys.els_.size(), 1);
  journey journey = journeys.els_[0];
  EXPECT_TRUE(journey.legs_.empty());
  EXPECT_EQ(journey.start_time_, start_time);
  EXPECT_EQ(journey.transfers_, 0);
  EXPECT_EQ(journey.dest_, location_idx_t::invalid());
  EXPECT_EQ(journey.dest_time_, start_time + 14h);

  a_star_exec.reconstruct(
      query{.start_time_ = start_time,
            .start_match_mode_ = location_match_mode::kEquivalent,
            .dest_match_mode_ = location_match_mode::kEquivalent,
            .start_ = {{tt.locations_.location_id_to_idx_.at(
                            {"S0", source_idx_t{0}}),
                        0_minutes, 0U}},
            .destination_ = {{tt.locations_.location_id_to_idx_.at(
                                  {"S2", source_idx_t{0}}),
                              0_minutes, 0U}}},
      journey);
  EXPECT_EQ(journey.legs_.size(), 2);
  EXPECT_FALSE(journey.error_);
  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .back()}
          .location_idx();
  EXPECT_EQ(journey.dest_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(journey.legs_.front().from_, start);
  for (int i = 0; i < journey.legs_.size() - 1; ++i) {
    EXPECT_EQ(journey.legs_[i].to_, journey.legs_[i + 1].from_);
  }
  EXPECT_EQ(journey.legs_.back().to_, journey.dest_);
  EXPECT_EQ(journey.transfers_, 0);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 14h);
  EXPECT_EQ(journey.legs_.back().arr_time_, start_time + 14h);
  EXPECT_EQ(journey.legs_.back().arr_time_, start_time + 14h);

  check_neighbours(tbd);
}

TEST(tb_a_star, add_starts_one_segment) {
  auto const tt = load_gtfs(overnight_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto start_segment = tbd.transport_first_segment_[transport_idx_t(0)];
  unixtime_t start_time =
      date::sys_days{date::year{2021} / date::March / date::day{1}} + 10h;
  auto stop_value = tt.route_location_seq_[route_idx_t(0)][0];
  stop start_stop(stop_value);
  tb_a_star a_star_exec(tbd.segment_transfers_.size());
  a_star_exec.state_.tbd_ = tbd;
  a_star_exec.tt_ = tt;

  a_star_exec.add_start(start_stop.location_idx(), start_time);
  EXPECT_TRUE(a_star_exec.is_start_segment_.test(start_segment));
  for (int i = 0; i < tbd.segment_transfers_.size(); ++i) {
    segment_idx_t segment{i};
    if (segment != start_segment) {
      EXPECT_FALSE(a_star_exec.is_start_segment_.test(segment));
    }
  }
  EXPECT_EQ(a_star_exec.transfers_[segment_idx_t(0)], 0);
}

loader::mem_dir two_start_segment_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S0 -> S3",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,10:00:00,10:00:00,S0,0,0,0
R0_MON,12:00:00,12:00:00,S1,1,0,0
R1_MON,11:00:00,11:00:00,S0,0,0,0
R1_MON,12:00:00,12:00:00,S2,1,0,0
R1_MON,13:00:00,13:00:00,S3,2,0,0
)");
}

TEST(tb_a_star, add_starts_two_segment) {
  auto const tt = load_gtfs(two_start_segment_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  unixtime_t start_time =
      date::sys_days{date::year{2021} / date::March / date::day{1}} + 10h;
  auto stop_value = tt.route_location_seq_[route_idx_t(0)][0];
  stop start_stop(stop_value);
  tb_a_star a_star_exec(tbd.segment_transfers_.size());
  a_star_exec.state_.tbd_ = tbd;
  a_star_exec.tt_ = tt;

  a_star_exec.add_start(start_stop.location_idx(), start_time);

  EXPECT_TRUE(a_star_exec.is_start_segment_.test(segment_idx_t{0}));
  EXPECT_TRUE(a_star_exec.is_start_segment_.test(segment_idx_t{1}));
  EXPECT_FALSE(a_star_exec.is_start_segment_.test(segment_idx_t{2}));

  EXPECT_EQ(a_star_exec.transfers_[segment_idx_t{0}], 0);
  EXPECT_EQ(a_star_exec.transfers_[segment_idx_t{1}], 0);
}

TEST(tb_a_star, a_star_two_start_segments) {
  auto const tt = load_gtfs(two_start_segment_files);
  auto const tbd = tb::preprocess(tt, profile_idx_t{0});
  auto end_segment = segment_idx_t(0);
  unixtime_t start_time =
      date::sys_days{date::year{2021} / date::March / date::day{1}} + 10h;
  auto day_idx = tt.day_idx(
      date::year_month_day{date::year{2021}, date::month{3}, date::day{1}});
  tb_a_star a_star_exec(tbd.segment_transfers_.size());
  a_star_exec.state_.tbd_ = tbd;
  a_star_exec.tt_ = tt;
  a_star_exec.travel_time_lower_bound_ =
      std::vector<uint16_t>(tt.n_locations(), 0);
  a_star_exec.state_.end_reachable_.resize(tbd.segment_transports_.size());
  a_star_exec.state_.end_reachable_.set(end_segment);
  a_star_exec.state_.end_reachable_.set(end_segment + 1);
  a_star_exec.state_.dist_to_dest_ =
      hash_map<segment_idx_t, duration_t>(tbd.segment_transports_.size());
  a_star_exec.state_.dist_to_dest_[segment_idx_t(1)] = duration_t::zero();
  a_star_exec.state_.dist_to_dest_[segment_idx_t(0)] = duration_t(1);
  auto stop_value = tt.route_location_seq_[route_idx_t(0)][0];
  stop start_stop(stop_value);
  a_star_exec.add_start(start_stop.location_idx(), start_time);
  pareto_set<journey> journeys;
  a_star_exec.execute(start_time, std::numeric_limits<std::uint8_t>::max(),
                      unixtime_t::max(), profile_idx_t{0}, journeys);
  auto result = get_segments(a_star_exec);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result.front(), segment_idx_t(1));
  EXPECT_EQ(a_star_exec.day_idx_[segment_idx_t(1)], day_idx);
  EXPECT_EQ(a_star_exec.day_idx_[segment_idx_t(0)], day_idx);
  for (int i = 0; i < tbd.segment_transfers_.size(); ++i) {
    EXPECT_EQ(a_star_exec.transfers_[segment_idx_t{i}], 0);
  }
  EXPECT_EQ(a_star_exec.end_segment_, segment_idx_t(1));

  EXPECT_EQ(journeys.els_.size(), 1);
  journey journey = journeys.els_[0];
  EXPECT_TRUE(journey.legs_.empty());
  EXPECT_EQ(journey.start_time_, start_time);
  EXPECT_EQ(journey.transfers_, 0);
  EXPECT_EQ(journey.dest_, location_idx_t::invalid());
  EXPECT_EQ(journey.dest_time_, start_time + 2h);

  a_star_exec.reconstruct(
      query{.start_time_ = start_time,
            .start_match_mode_ = location_match_mode::kEquivalent,
            .dest_match_mode_ = location_match_mode::kEquivalent,
            .start_ = {{tt.locations_.location_id_to_idx_.at(
                            {"S0", source_idx_t{0}}),
                        0_minutes, 0U}},
            .destination_ = {{tt.locations_.location_id_to_idx_.at(
                                  {"S2", source_idx_t{0}}),
                              0_minutes, 0U}}},
      journey);
  EXPECT_EQ(journey.legs_.size(), 2);
  EXPECT_FALSE(journey.error_);
  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{1}]][1]}
          .location_idx();
  EXPECT_EQ(journey.dest_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(journey.legs_.front().from_, start);
  for (int i = 0; i < journey.legs_.size() - 1; ++i) {
    EXPECT_EQ(journey.legs_[i].to_, journey.legs_[i + 1].from_);
  }
  EXPECT_EQ(journey.legs_.back().to_, journey.dest_);
  EXPECT_EQ(journey.transfers_, 0);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time + 1h);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 2h);
  EXPECT_EQ(journey.legs_.back().dep_time_, start_time + 2h);
  EXPECT_EQ(journey.legs_.back().arr_time_, start_time + 2h);

  check_neighbours(tbd);
}

TEST(tb_a_star, search_execute) {
  auto const tt = load_gtfs(longer_trips_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};
  unixtime_t start_time = sys_days{year{2021} / March / day{1}};

  auto constexpr src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = start_time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"S0", src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({"S7", src}), 0_minutes, 0U}}};

  routing::search<direction::kForward, tb_a_star> searcher{
      tt, nullptr, search_state, algo_state, std::move(q)};
  auto result = *searcher.execute().journeys_;
  EXPECT_EQ(result.els_.size(), 1);
  journey journey = result.els_[0];

  EXPECT_EQ(journey.legs_.size(), 6);
  EXPECT_FALSE(journey.error_);
  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{2}]]
               .back()}
          .location_idx();
  EXPECT_EQ(journey.dest_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(journey.legs_.front().from_, start);
  for (int i = 0; i < journey.legs_.size() - 1; ++i) {
    EXPECT_EQ(journey.legs_[i].to_, journey.legs_[i + 1].from_);
  }
  EXPECT_EQ(journey.legs_.back().to_, journey.dest_);

  auto s3 =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]][3]}
          .location_idx();
  auto s6 =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{1}]][2]}
          .location_idx();
  EXPECT_EQ(journey.legs_.front().to_, s3);
  EXPECT_EQ(journey.legs_[1].to_, s3);
  EXPECT_EQ(journey.legs_[2].to_, s6);
  EXPECT_EQ(journey.legs_[3].to_, s6);

  EXPECT_EQ(journey.transfers_, 2);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 3h);
  EXPECT_EQ(journey.legs_[1].dep_time_, start_time + 3h);
  EXPECT_EQ(journey.legs_[1].arr_time_, start_time + 3h + 2min);
  EXPECT_EQ(journey.legs_[2].dep_time_, start_time + 3h + 10min);
  EXPECT_EQ(journey.legs_[2].arr_time_, start_time + 5h);
  EXPECT_EQ(journey.legs_[3].dep_time_, start_time + 5h);
  EXPECT_EQ(journey.legs_[3].arr_time_, start_time + 5h + 2min);
  EXPECT_EQ(journey.legs_[4].dep_time_, start_time + 5h + 10min);
  EXPECT_EQ(journey.legs_[4].arr_time_, start_time + 6h);
  EXPECT_EQ(journey.legs_[5].dep_time_, start_time + 6h);
  EXPECT_EQ(journey.legs_[5].arr_time_, start_time + 6h);
}

TEST(tb_a_star, search_overnight) {
  auto const tt = load_gtfs(overnight_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};
  unixtime_t start_time = sys_days{year{2021} / March / day{1}} + 10h;

  auto constexpr src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = start_time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"S0", src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({"S2", src}), 0_minutes, 0U}}};

  routing::search<direction::kForward, tb_a_star> searcher{
      tt, nullptr, search_state, algo_state, std::move(q)};

  auto result = *searcher.execute().journeys_;
  EXPECT_EQ(result.els_.size(), 1);
  journey journey = result.els_[0];

  EXPECT_EQ(journey.legs_.size(), 4);
  EXPECT_FALSE(journey.error_);
  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{1}]]
               .back()}
          .location_idx();
  EXPECT_EQ(journey.dest_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(journey.legs_.front().from_, start);
  for (int i = 0; i < journey.legs_.size() - 1; ++i) {
    EXPECT_EQ(journey.legs_[i].to_, journey.legs_[i + 1].from_);
  }
  EXPECT_EQ(journey.legs_.back().to_, journey.dest_);
  auto s1 = stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
                     .back()}
                .location_idx();
  EXPECT_EQ(journey.legs_.front().to_, s1);
  EXPECT_EQ(journey.legs_[1].to_, s1);

  EXPECT_EQ(journey.transfers_, 1);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 2h);
  EXPECT_EQ(journey.legs_[1].dep_time_, start_time + 2h);
  EXPECT_EQ(journey.legs_[1].arr_time_, start_time + 2h + 2min);
  EXPECT_EQ(journey.legs_[2].dep_time_, start_time + 14h);
  EXPECT_EQ(journey.legs_[2].arr_time_, start_time + 22h);
  EXPECT_EQ(journey.legs_[3].dep_time_, start_time + 22h);
  EXPECT_EQ(journey.legs_[3].arr_time_, start_time + 22h);
}
TEST(tb_a_star, search_intermodal) {
  auto const tt = load_gtfs(longer_trips_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};
  unixtime_t start_time = sys_days{year{2021} / March / day{1}};

  auto constexpr src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = start_time,
      .start_match_mode_ = location_match_mode::kIntermodal,
      .dest_match_mode_ = location_match_mode::kIntermodal,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"S0", src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({"S7", src}), 0_minutes, 0U}}};

  routing::search<direction::kForward, tb_a_star> searcher{
      tt, nullptr, search_state, algo_state, std::move(q)};
  auto result = *searcher.execute().journeys_;
  EXPECT_EQ(result.els_.size(), 1);
  journey journey = result.els_[0];

  EXPECT_EQ(journey.legs_.size(), 7);
  EXPECT_FALSE(journey.error_);
  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{2}]]
               .back()}
          .location_idx();
  EXPECT_EQ(journey.dest_, dest);
  auto start = get_special_station(special_station::kStart);
  EXPECT_EQ(journey.legs_.front().from_, start);
  for (int i = 0; i < journey.legs_.size() - 1; ++i) {
    EXPECT_EQ(journey.legs_[i].to_, journey.legs_[i + 1].from_);
  }
  EXPECT_EQ(journey.legs_.back().to_,
            get_special_station(special_station::kEnd));
  auto s0 = stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
                     .front()}
                .location_idx();
  auto s3 =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]][3]}
          .location_idx();
  auto s6 =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{1}]][2]}
          .location_idx();
  EXPECT_EQ(journey.legs_.front().to_, s0);
  EXPECT_EQ(journey.legs_[1].to_, s3);
  EXPECT_EQ(journey.legs_[2].to_, s3);
  EXPECT_EQ(journey.legs_[3].to_, s6);
  EXPECT_EQ(journey.legs_[4].to_, s6);
  EXPECT_EQ(journey.legs_[5].to_, dest);

  EXPECT_EQ(journey.transfers_, 2);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time);
  EXPECT_EQ(journey.legs_[1].dep_time_, start_time);
  EXPECT_EQ(journey.legs_[1].arr_time_, start_time + 3h);
  EXPECT_EQ(journey.legs_[2].dep_time_, start_time + 3h);
  EXPECT_EQ(journey.legs_[2].arr_time_, start_time + 3h + 2min);
  EXPECT_EQ(journey.legs_[3].dep_time_, start_time + 3h + 10min);
  EXPECT_EQ(journey.legs_[3].arr_time_, start_time + 5h);
  EXPECT_EQ(journey.legs_[4].dep_time_, start_time + 5h);
  EXPECT_EQ(journey.legs_[4].arr_time_, start_time + 5h + 2min);
  EXPECT_EQ(journey.legs_[5].dep_time_, start_time + 5h + 10min);
  EXPECT_EQ(journey.legs_[5].arr_time_, start_time + 6h);
  EXPECT_EQ(journey.legs_[6].dep_time_, start_time + 6h);
  EXPECT_EQ(journey.legs_[6].arr_time_, start_time + 6h);
}

loader::mem_dir no_journey_segment_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
)");
}
TEST(tb_a_star, search_no_solution) {
  auto const tt = load_gtfs(no_journey_segment_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};
  unixtime_t start_time = sys_days{year{2021} / March / day{1}};

  auto constexpr src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = start_time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"S0", src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({"S2", src}), 0_minutes, 0U}}};

  routing::search<direction::kForward, tb_a_star> searcher{
      tt, nullptr, search_state, algo_state, std::move(q)};
  auto result = *searcher.execute().journeys_;
  EXPECT_TRUE(result.els_.empty());
}

loader::mem_dir different_termination_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S0 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R1_MON,00:00:00,00:00:00,S0,0,0,0
R1_MON,02:00:00,02:00:00,S2,1,0,0
)");
}
TEST(tb_a_star, search_different_termination) {
  auto const tt = load_gtfs(different_termination_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};
  unixtime_t start_time = sys_days{year{2021} / March / day{1}};

  auto constexpr src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = start_time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"S0", src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({"S1", src}), 0_minutes, 0U}}};

  routing::search<direction::kForward, tb_a_star> searcher{
      tt, nullptr, search_state, algo_state, std::move(q)};
  auto result = *searcher.execute().journeys_;
  EXPECT_EQ(result.els_.size(), 1);
  EXPECT_EQ(result.els_.front().legs_.size(), 2);

  EXPECT_EQ(result.els_.front().legs_.front().dep_time_, start_time);
  EXPECT_EQ(result.els_.front().legs_.front().arr_time_, start_time + 1h);
  EXPECT_EQ(result.els_.front().legs_.back().dep_time_, start_time + 1h);
  EXPECT_EQ(result.els_.front().legs_.back().arr_time_, start_time + 1h);

  EXPECT_EQ(result.els_.front().transfers_, 0);
  EXPECT_FALSE(result.els_.front().error_);
  EXPECT_EQ(result.els_.front().legs_.front().to_,
            result.els_.front().legs_.back().from_);

  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .back()}
          .location_idx();
  EXPECT_EQ(result.els_.front().dest_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().to_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().from_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(result.els_.front().legs_.front().from_, start);
  EXPECT_EQ(result.els_.front().legs_.front().to_, dest);
}

loader::mem_dir max_cost_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,
S3,S3,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S3",2
R1,DTA,R1,R1,"S1 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,01:00:00,01:00:00,S1,1,0,0
R0_MON,02:00:00,02:00:00,S3,2,0,0
R1_MON,01:10:00,01:10:00,S1,0,0,0
R1_MON,24:00:00,24:00:00,S2,1,0,0
)");
}
TEST(tb_a_star, search_max_cost) {
  auto const tt = load_gtfs(max_cost_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};
  unixtime_t start_time = sys_days{year{2021} / March / day{1}};

  auto constexpr src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = start_time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"S0", src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({"S2", src}), 0_minutes, 0U}}};

  routing::search<direction::kForward, tb_a_star> searcher{
      tt, nullptr, search_state, algo_state, std::move(q)};
  auto result = *searcher.execute().journeys_;
  EXPECT_EQ(result.els_.size(), 1);
  EXPECT_EQ(result.els_.front().legs_.size(), 4);

  EXPECT_EQ(result.els_.front().legs_.front().dep_time_, start_time);
  EXPECT_EQ(result.els_.front().legs_.front().arr_time_, start_time + 1h);
  EXPECT_EQ(result.els_.front().legs_[1].dep_time_, start_time + 1h);
  EXPECT_EQ(result.els_.front().legs_[1].arr_time_, start_time + 1h + 2min);
  EXPECT_EQ(result.els_.front().legs_[2].dep_time_, start_time + 1h + 10min);
  EXPECT_EQ(result.els_.front().legs_[2].arr_time_, start_time + 24h);
  EXPECT_EQ(result.els_.front().legs_.back().dep_time_, start_time + 24h);
  EXPECT_EQ(result.els_.front().legs_.back().arr_time_, start_time + 24h);

  EXPECT_EQ(result.els_.front().transfers_, 1);
  EXPECT_FALSE(result.els_.front().error_);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(result.els_.front().legs_[i].to_,
              result.els_.front().legs_[i + 1].from_);
  }

  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{1}]]
               .back()}
          .location_idx();
  EXPECT_EQ(result.els_.front().dest_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().to_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().from_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(result.els_.front().legs_.front().from_, start);
  auto s1 =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]][1]}
          .location_idx();
  EXPECT_EQ(result.els_.front().legs_.front().to_, s1);
  EXPECT_EQ(result.els_.front().legs_[1].to_, s1);
}

loader::mem_dir over_24h_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S1",2
R1,DTA,R1,R1,"S0 -> S2 -> S1",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1
R1,MON,R1_MON,R1_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,00:00:00,00:00:00,S0,0,0,0
R0_MON,25:00:00,25:00:00,S1,1,0,0
R1_MON,00:00:00,00:00:00,S0,0,0,0
R1_MON,00:10:00,00:10:00,S2,1,0,0
R1_MON,00:20:00,00:20:00,S1,2,0,0
)");
}

TEST(tb_a_star, longer_trip_then_24h) {
  auto const tt = load_gtfs(over_24h_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};
  unixtime_t start_time = sys_days{year{2021} / March / day{1}};

  auto constexpr src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = start_time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"S0", src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({"S1", src}),
                        0_minutes, 0U}},
  };

  routing::search<direction::kForward, tb_a_star> searcher{
      tt, nullptr, search_state, algo_state, std::move(q)};
  auto result = *searcher.execute().journeys_;
  EXPECT_EQ(result.size(), 1);
  auto journey = result.els_.front();
  EXPECT_EQ(journey.legs_.size(), 2);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 20min);
  EXPECT_EQ(journey.legs_.back().arr_time_, start_time + 20min);
  EXPECT_EQ(journey.legs_.back().dep_time_, start_time + 20min);

  EXPECT_EQ(journey.legs_.front().to_, journey.legs_.back().from_);

  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{1}]]
               .back()}
          .location_idx();
  EXPECT_EQ(result.els_.front().dest_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().to_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().from_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(result.els_.front().legs_.front().from_, start);
  EXPECT_EQ(result.els_.front().legs_.front().to_,
            result.els_.front().legs_.back().from_);
}

TEST(tb_a_star, start_time_different_date_to_first_segment) {
  auto const tt = load_gtfs(over_24h_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};
  unixtime_t start_time = sys_days{year{2021} / February / day{28}} + 23h;

  auto constexpr src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = start_time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"S0", src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({"S1", src}),
                        0_minutes, 0U}},
  };

  routing::search<direction::kForward, tb_a_star> searcher{
      tt, nullptr, search_state, algo_state, std::move(q)};
  auto result = *searcher.execute().journeys_;
  EXPECT_EQ(result.size(), 1);
  auto journey = result.els_.front();
  EXPECT_EQ(journey.legs_.size(), 2);

  EXPECT_EQ(journey.legs_.front().dep_time_,
            sys_days{year{2021} / March / day{1}});
  EXPECT_EQ(journey.legs_.front().arr_time_,
            sys_days{year{2021} / March / day{1}} + 20min);
  EXPECT_EQ(journey.legs_.back().arr_time_,
            sys_days{year{2021} / March / day{1}} + 20min);
  EXPECT_EQ(journey.legs_.back().dep_time_,
            sys_days{year{2021} / March / day{1}} + 20min);

  EXPECT_EQ(journey.legs_.front().to_, journey.legs_.back().from_);

  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{1}]]
               .back()}
          .location_idx();
  EXPECT_EQ(result.els_.front().dest_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().to_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().from_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(result.els_.front().legs_.front().from_, start);
  EXPECT_EQ(result.els_.front().legs_.front().to_,
            result.els_.front().legs_.back().from_);
}

loader::mem_dir segment_passing_midnight_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,23:30:00,23:30:00,S0,0,0,0
R0_MON,24:30:00,24:30:00,S1,1,0,0
R0_MON,25:00:00,25:00:00,S2,2,0,0
)");
}
TEST(tb_a_star, segment_passing_midnight) {
  auto const tt = load_gtfs(segment_passing_midnight_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};
  unixtime_t start_time = sys_days{year{2021} / March / day{1}} + 23h + 30min;

  auto constexpr src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = start_time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"S0", src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({"S2", src}),
                        0_minutes, 0U}},
      .max_travel_time_ = duration_t{24h}};

  routing::search<direction::kForward, tb_a_star> searcher{
      tt, nullptr, search_state, algo_state, std::move(q)};
  auto result = *searcher.execute().journeys_;
  EXPECT_EQ(result.size(), 1);
  auto journey = result.els_.front();
  EXPECT_EQ(journey.legs_.size(), 2);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 1h + 30min);
  EXPECT_EQ(journey.legs_.back().arr_time_, start_time + 1h + 30min);
  EXPECT_EQ(journey.legs_.back().dep_time_, start_time + 1h + 30min);

  EXPECT_EQ(journey.legs_.front().to_, journey.legs_.back().from_);

  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .back()}
          .location_idx();
  EXPECT_EQ(result.els_.front().dest_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().to_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().from_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .front()}
          .location_idx();
  EXPECT_EQ(result.els_.front().legs_.front().from_, start);
  EXPECT_EQ(result.els_.front().legs_.front().to_,
            result.els_.front().legs_.back().from_);
}

loader::mem_dir transport_starts_before_start_time_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,,Europe/London

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S0,S0,,,,,,
S1,S1,,,,,,
S2,S2,,,,,,

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
MON,1,0,0,0,0,0,0,20210301,20210307

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DTA,R0,R0,"S0 -> S2",2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,MON,R0_MON,R0_MON,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
R0_MON,23:30:00,23:30:00,S0,0,0,0
R0_MON,24:30:00,24:30:00,S1,1,0,0
R0_MON,25:00:00,25:00:00,S2,2,0,0
)");
}
TEST(tb_a_star, tranpsport_starts_before_start_time) {
  auto const tt = load_gtfs(transport_starts_before_start_time_files);
  auto const tbd = preprocess(tt, profile_idx_t{0});
  static auto search_state = routing::search_state{};
  auto algo_state = a_star_state{tbd};
  unixtime_t start_time = sys_days{year{2021} / March / day{2}};

  auto constexpr src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = start_time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"S1", src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({"S2", src}),
                        0_minutes, 0U}},
      .max_travel_time_ = duration_t{24h}};

  routing::search<direction::kForward, tb_a_star> searcher{
      tt, nullptr, search_state, algo_state, std::move(q)};
  auto result = *searcher.execute().journeys_;
  EXPECT_EQ(result.size(), 1);
  auto journey = result.els_.front();
  EXPECT_EQ(journey.legs_.size(), 2);

  EXPECT_EQ(journey.legs_.front().dep_time_, start_time + 30min);
  EXPECT_EQ(journey.legs_.front().arr_time_, start_time + 1h);
  EXPECT_EQ(journey.legs_.back().arr_time_, start_time + 1h);
  EXPECT_EQ(journey.legs_.back().dep_time_, start_time + 1h);

  EXPECT_EQ(journey.legs_.front().to_, journey.legs_.back().from_);

  auto dest =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]]
               .back()}
          .location_idx();
  EXPECT_EQ(result.els_.front().dest_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().to_, dest);
  EXPECT_EQ(result.els_.front().legs_.back().from_, dest);
  auto start =
      stop{tt.route_location_seq_[tt.transport_route_[transport_idx_t{0}]][1]}
          .location_idx();
  EXPECT_EQ(result.els_.front().legs_.front().from_, start);
  EXPECT_EQ(result.els_.front().legs_.front().to_,
            result.els_.front().legs_.back().from_);
}
}  // namespace nigiri::routing::tb::a_star
