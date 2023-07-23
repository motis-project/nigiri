#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace std::string_view_literals;

constexpr auto const test_files = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
H1,H1,,49.874446125541,8.649950760650633,,
H2,H2,,50.118039690771,8.669541533394973,,
H3,H3,,50.586661845548,8.681833607424945,,
L11,L11,,49.8938671974,8.650568450613406,,
L12,L12,,49.8606820772,8.636774999023730,,
L21,L21,,50.1567601578,8.657005394688590,,
L22,L22,,50.0883693880,8.627579364630613,,
L31,L31,,50.5662229997,8.682753171380188,,
L32,L32,,50.5948349022,8.734248723981645,,
HK1,HK1,,50.4410610968,8.657005394805864,,
HK2,HK2,,50.4404754493,8.704822693650074,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RHL1,DB,HL1,,,3
RHL2,DB,HL2,,,3
RHR1,DB,HR1,,,3
RHR2,DB,HR2,,,3
RL11,DB,L11,,,3
RL12,DB,L12,,,3
RL21,DB,L21,,,3
RL22,DB,L22,,,3
RL31,DB,L31,,,3
RL32,DB,L32,,,3
RK1,DB,K1,,,3
RK2,DB,K2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RHL1,S,HL1,,
RHL2,S,HL2,,
RHR1,S,HR1,,
RHR2,S,HR2,,
RL11,S,L11,,
RL12,S,L12,,
RL21,S,L21,,
RL22,S,L22,,
RL31,S,L31,,
RL32,S,L32,,
RK1,S,K1,,
RK2,S,K2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
HL1,00:00:00,00:00:00,H1,1,0,0
HL1,02:00:00,02:00:00,H2,2,0,0
HL1,03:00:00,03:00:00,HK1,3,0,0
HR1,00:00:00,00:00:00,HK2,1,0,0
HR1,02:00:00,02:00:00,H3,2,0,0
HL2,00:00:00,00:00:00,HK1,1,0,0
HL2,02:00:00,02:00:00,H2,2,0,0
HL2,03:00:00,03:00:00,H1,3,0,0
HR2,00:00:00,00:00:00,H3,1,0,0
HR2,01:00:00,01:00:00,HK2,2,0,0
K1,00:00:00,00:00:00,HK1,1,0,0
K1,00:10:00,00:10:00,HK2,2,0,0
K2,00:00:00,00:00:00,HK2,1,0,0
K2,00:10:00,00:10:00,HK1,2,0,0
L11,00:00:00,00:00:00,L11,1,0,0
L11,00:10:00,00:10:00,H1,2,0,0
L11,00:20:00,00:20:00,L12,3,0,0
L12,00:00:00,00:00:00,L12,1,0,0
L12,00:10:00,00:10:00,H1,2,0,0
L12,00:20:00,00:20:00,L11,3,0,0
L21,00:00:00,00:00:00,L21,1,0,0
L21,00:10:00,00:10:00,H2,2,0,0
L21,00:20:00,00:20:00,L22,3,0,0
L22,00:00:00,00:00:00,L22,1,0,0
L22,00:10:00,00:10:00,H2,2,0,0
L22,00:20:00,00:20:00,L21,3,0,0
L31,00:00:00,00:00:00,L31,1,0,0
L31,00:10:00,00:10:00,H3,2,0,0
L31,00:20:00,00:20:00,L32,3,0,0
L32,00:00:00,00:00:00,L32,1,0,0
L32,00:10:00,00:10:00,H3,2,0,0
L32,00:20:00,00:20:00,L31,3,0,0

# frequencies.txt
trip_id,start_time,end_time,headway_secs
HL1,00:00:00,24:00:00,7200
HR1,00:00:00,24:00:00,7200
K1,00:00:00,24:00:00,600
K2,00:00:00,24:00:00,600
L11,00:00:00,24:00:00,600
L12,00:00:00,24:00:00,600
L21,00:00:00,24:00:00,600
L22,00:00:00,24:00:00,600
L31,00:00:00,24:00:00,600
L32,00:00:00,24:00:00,600

# calendar_dates.txt
service_id,date,exception_type
S,20190501,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
C1,D,0,2
)"sv;

TEST(routing, reach_test) {
  constexpr auto const src = source_idx_t{0U};

  timetable tt;
  tt.date_range_ = {sys_days{2019_y / May / 1}, sys_days{2019_y / May / 2}};
  gtfs::load_timetable(
      {.link_stop_distance_ = 0, .default_tz_ = "Europe/Berlin"}, src,
      mem_dir::read(test_files), tt);
  finalize(tt);

  auto s = routing::raptor_state{};
  auto is_dest = std::vector<bool>(tt.n_locations());
  auto dist_to_dest = std::vector<std::uint16_t>(
      tt.n_locations(), kInvalidDelta<direction::kForward>);
  auto lb = std::vector<std::uint16_t>(tt.n_locations(), 0U);
  auto p = pareto_set<routing::journey>{};
  auto const base_day = tt.day_idx(date::sys_days{May / 1 / 2019});
  auto r = routing::raptor<direction::kForward, false, true>{
      tt, nullptr, s, is_dest, dist_to_dest, lb, base_day};

  auto const start_time = unixtime_t{sys_days{2019_y / May / 1}} + 5_hours;
  auto const l12 = tt.locations_.get(location_id{.id_ = "L12", .src_ = src}).l_;
  auto const l31 = tt.locations_.get(location_id{.id_ = "L31", .src_ = src}).l_;
  r.next_start_time();
  r.add_start(l12, start_time);
  r.execute(start_time, routing::kMaxTransfers,
            start_time + routing::kMaxTravelTime, p);

  auto results = pareto_set<>

                     p.clear();
  for (auto k = 1U; k != routing::kMaxTransfers + 1U; ++k) {
    auto const i = to_idx(l31);
    auto const dest_time = s.round_times_[k][i];
    if (dest_time == kInvalidDelta<direction::kForward>) {
      continue;
    }
    fmt::print("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}\n",
               start_time, delta_to_unix(r.base(), s.round_times_[k][i]),
               location{tt, location_idx_t{i}}, k - 1);
    auto const [optimal, it, dominated_by] =
        p.add(routing::journey{.legs_ = {},
                               .start_time_ = start_time,
                               .dest_time_ = delta_to_unix(r.base(), dest_time),
                               .dest_ = location_idx_t{i},
                               .transfers_ = static_cast<std::uint8_t>(k - 1)});
    if (!optimal) {
      fmt::print("  DOMINATED BY: start={}, dest={} @ {}, transfers={}\n",
                 dominated_by->start_time_, dominated_by->dest_time_,
                 location{tt, dominated_by->dest_}, dominated_by->transfers_);
    }
  }

  auto q = routing::query{};
  q.start_time_ = start_time;
  q.start_match_mode_ = routing::location_match_mode::kEquivalent;
  q.dest_match_mode_ = routing::location_match_mode::kEquivalent;
  q.destination_ = {routing::offset{l31, 0_minutes, 0U}};
  q.start_ = {routing::offset{l12, 0_minutes, 0U}};
  for (auto& j : p) {
    r.reconstruct(q, j);
  }

  for (auto const& x : p) {
    x.print(std::cout, tt);
    std::cout << "\n\n";
  }
}