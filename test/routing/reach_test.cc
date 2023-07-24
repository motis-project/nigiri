#include "gtest/gtest.h"

#include "geo/box.h"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/parser/split.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/rt/frun.h"
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
HL2,00:00:00,24:00:00,7200
HR2,00:00:00,24:00:00,7200
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
S,20190502,1
S,20190501,1
S,20190430,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
C1,D,0,2
)"sv;

TEST(routing, reach_test) {
  constexpr auto const src = source_idx_t{0U};

  timetable tt;
  tt.date_range_ = {sys_days{2019_y / April / 30}, sys_days{2019_y / May / 3}};
  gtfs::load_timetable(
      {.link_stop_distance_ = 0, .default_tz_ = "Europe/Berlin"}, src,
      mem_dir::read(test_files), tt);
  finalize(tt);

  auto s = routing::raptor_state{};
  auto is_dest = std::vector<bool>(tt.n_locations());
  auto dist_to_dest = std::vector<std::uint16_t>(
      tt.n_locations(), kInvalidDelta<direction::kForward>);
  auto lb = std::vector<std::uint16_t>(tt.n_locations(), 0U);
  auto const base_day = tt.day_idx(date::sys_days{May / 1 / 2019});
  auto r = routing::raptor<direction::kForward, false, true>{
      tt, nullptr, s, is_dest, dist_to_dest, lb, base_day};

  auto const full_day = interval{unixtime_t{sys_days{2019_y / May / 1}},
                                 unixtime_t{sys_days{2019_y / May / 2}}};

  std::vector<std::vector<pareto_set<routing::journey>>> results;
  results.resize(tt.n_locations(),
                 std::vector<pareto_set<routing::journey>>(tt.n_locations()));

  std::vector<
      std::tuple<routing::journey, double, location_idx_t, location_idx_t>>
      route_reachs;
  route_reachs.resize(tt.n_routes(),
                      {routing::journey{}, 0.0, location_idx_t::invalid(),
                       location_idx_t::invalid()});

  auto const update_route_reachs = [&](routing::journey const& j, bool debug) {
    //    auto const prev_debug = debug;
    for (auto const& l : j.legs_) {
      if (!std::holds_alternative<routing::journey::run_enter_exit>(l.uses_)) {
        continue;
      }

      auto const ree = std::get<routing::journey::run_enter_exit>(l.uses_);
      auto const r = tt.transport_route_[ree.r_.t_.t_idx_];

      //      if (r == route_idx_t{5}) {
      //        debug = true;
      //      } else {
      //        debug = prev_debug;
      //      }

      auto const start_l = location{tt, j.legs_.front().from_};
      auto const dest_l = location{tt, j.legs_.back().to_};

      auto const start = tt.locations_.coordinates_[j.legs_.front().from_];
      auto const dest = tt.locations_.coordinates_[j.legs_.back().to_];

      auto const fr = rt::frun{tt, nullptr, ree.r_};

      auto& [journey, reach, stop, x] = route_reachs[to_idx(r)];
      for (auto i = ree.stop_range_.from_; i != ree.stop_range_.to_; ++i) {
        auto const stp = tt.locations_.coordinates_[fr[i].get_location_idx()];
        auto const stp_l = location{tt, fr[i].get_location_idx()};

        auto const new_reach =
            std::min(geo::distance(start, stp), geo::distance(stp, dest));

        if (debug) {
          std::cout << "stop " << i << " [" << reach << "]:\n";
          std::cout << "  " << stp_l << " - " << start_l << ": "
                    << geo::distance(start, stp) << "\n";
          std::cout << "  " << stp_l << " - " << dest_l << ": "
                    << geo::distance(stp, dest) << "\n";
          std::cout << "    -> new_reach=" << new_reach;
        }

        if (new_reach > reach) {
          if (debug) std::cout << " -> UPDATE\n";
          journey = j;
          reach = new_reach;
          stop = fr[i].get_location_idx();
          if (geo::distance(start, stp) < geo::distance(stp, dest)) {
            x = j.legs_.front().from_;
          } else {
            x = j.legs_.back().to_;
          }
        } else {
          if (debug) std::cout << " -> NO UPDATE\n";
        }
      }
    }
  };

  std::vector<routing::start> starts;
  for (auto l = 0U; l != tt.n_locations(); ++l) {
    r.reset_arrivals();

    auto q = routing::query{};
    q.start_match_mode_ = routing::location_match_mode::kEquivalent;
    q.dest_match_mode_ = routing::location_match_mode::kEquivalent;
    q.start_ = {routing::offset{location_idx_t{l}, 0_minutes, 0U}};

    starts.clear();
    routing::get_starts(direction::kForward, tt, nullptr, full_day, q.start_,
                        routing::location_match_mode::kEquivalent, true, starts,
                        false);

    utl::equal_ranges_linear(
        starts,
        [](routing::start const& a, routing::start const& b) {
          return a.time_at_start_ == b.time_at_start_;
        },
        [&](auto&& from_it, auto&& to_it) {
          r.next_start_time();
          auto const start_time = from_it->time_at_start_;
          trace("START TIME: {}\n", start_time);

          q.start_time_ = start_time;

          for (auto const& st : it_range{from_it, to_it}) {
            trace("init: time_at_start={}, time_at_stop={} at {}\n",
                  st.time_at_start_, st.time_at_stop_, location_idx_t{l});
            r.add_start(st.stop_, st.time_at_stop_);
          }

          auto const worst_time_at_dest = start_time + routing::kMaxTravelTime;
          trace("worst at dest: {}\n", worst_time_at_dest);
          r.execute(start_time, routing::kMaxTransfers, worst_time_at_dest,
                    results[l][l]);

          for (auto t = 0U; t != tt.n_locations(); ++t) {
            if (t == l) {
              continue;
            }

            for (auto k = 1U; k != routing::kMaxTransfers + 1U; ++k) {
              auto const dest_time = s.round_times_[k][t];
              if (dest_time == kInvalidDelta<direction::kForward>) {
                continue;
              }
              trace("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}\n",
                    start_time, delta_to_unix(r.base(), s.round_times_[k][t]),
                    location{tt, location_idx_t{t}}, k - 1);
              auto const [optimal, it, dominated_by] =
                  results[l][t].add(routing::journey{
                      .legs_ = {},
                      .start_time_ = start_time,
                      .dest_time_ = delta_to_unix(r.base(), dest_time),
                      .dest_ = location_idx_t{t},
                      .transfers_ = static_cast<std::uint8_t>(k - 1)});
              if (!optimal) {
                trace("  DOMINATED BY: start={}, dest={} @ {}, transfers={}\n",
                      dominated_by->start_time_, dominated_by->dest_time_,
                      location{tt, dominated_by->dest_},
                      dominated_by->transfers_);
              }
            }

            for (auto& j : results[l][t]) {
              if (!j.legs_.empty()) {
                continue;
              }
              q.destination_ = {
                  routing::offset{location_idx_t{t}, 0_minutes, 0U}};
              try {
                trace("RECONSTRUCT FROM={}, TO={}, START_TIME={}\n",
                      location{tt, location_idx_t{l}},
                      location{tt, location_idx_t{t}}, start_time);
                r.reconstruct(q, j);
                update_route_reachs(j, false);
              } catch (std::exception const& e) {
                std::cout << "fail for " << location{tt, location_idx_t{l}}
                          << " -> " << location{tt, location_idx_t{t}} << ": "
                          << e.what() << "\n";
                throw;
              }
            }
          }
        });
  }

  for (auto const [route, reach_tuple] : utl::enumerate(route_reachs)) {
    auto const [j, reach, stp, x] = reach_tuple;
    auto const t = tt.route_transport_ranges_[route_idx_t{route}][0];
    auto const [type, name] =
        utl::split<' ', utl::cstr, utl::cstr>(tt.transport_name(t));
    if (stp == location_idx_t::invalid()) {
      std::cout << route << " " << name.view() << " INVALID\n";
      continue;
    }

    auto b = geo::box{};
    auto const location_seq = tt.route_location_seq_[route_idx_t{route}];
    for (auto i = 0U; i != location_seq.size(); ++i) {
      auto const l = stop{location_seq[i]}.location_idx();
      b.extend(tt.locations_.coordinates_[l]);
    }
    std::cout << reach << " " << geo::distance(b.max_, b.min_) << " "
              << name.view() << "\n";
  }

  // clang-format off
  // PLOT WITH
  // gnuplot  -persist -e "plot 'routes.dat' using 1:2:3 with labels point pt 7" out.png
}