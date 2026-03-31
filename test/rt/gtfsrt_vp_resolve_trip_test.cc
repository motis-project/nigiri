#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/delay_prediction.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/util.h"
#include "nigiri/timetable.h"

#include <fstream>
#include "nigiri/loader/dir.h"
#include "gtfsrt/gtfs-realtime.pb.h"

#include "./util.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::rt;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_literals;
using namespace std::string_view_literals;

namespace {

mem_dir test_files() {
  return mem_dir::read(R"(
     "(
# agency.txt
agency_name,agency_url,agency_timezone,agency_lang,agency_phone,agency_fare_url,agency_id
"grt",https://grt.ca,America/New_York,en,519-585-7555,http://www.grt.ca/en/fares/FarePrices.asp,grt

# stops.txt
stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station,wheelchair_boarding,platform_code
2351,2351,Block Line Station,,  43.422095, -80.462740,,
1033,1033,Block Line / Hanover,,  43.419023, -80.466600,,,0,,1,
2086,2086,Block Line / Kingswood,,  43.417796, -80.473666,,,0,,1,
2885,2885,Block Line / Strasburg,,  43.415733, -80.480340,,,0,,1,
2888,2888,Block Line / Laurentian,,  43.412766, -80.491494,,,0,,1,
3189,3189,Block Line / Westmount,,  43.411515, -80.498966,,,0,,1,
3895,3895,Fischer-Hallman / Westmount,,  43.406717, -80.500091,,,0,,1,
3893,3893,Fischer-Hallman / Activa,,  43.414221, -80.508534,,,0,,1,
2969,2969,Fischer-Hallman / Ottawa,,  43.416570, -80.510880,,,0,,1,
2971,2971,Fischer-Hallman / Mcgarry,,  43.423420, -80.518818,,,0,,1,
2986,2986,Fischer-Hallman / Queens,,  43.428585, -80.523337,,,0,,1,
3891,3891,Fischer-Hallman / Highland,,  43.431587, -80.525376,,,0,,1,
3143,3143,Fischer-Hallman / Victoria,,  43.436843, -80.529202,,,0,,1,
3144,3144,Fischer-Hallman / Stoke,,  43.439462, -80.535435,,,0,,1,
3146,3146,Fischer-Hallman / University Ave.,,  43.444402, -80.545691,,,0,,1,
1992,1992,Fischer-Hallman / Thorndale,,  43.448678, -80.550034,,,0,,1,
1972,1972,Fischer-Hallman / Erb,,  43.452906, -80.553686,,,0,,1,
3465,3465,Fischer-Hallman / Keats Way,,  43.458370, -80.557824,,,0,,1,
3890,3890,Fischer-Hallman / Columbia,,  43.467368, -80.565646,,,0,,1,
1117,1117,Columbia / U.W. - Columbia Lake Village,,  43.469091, -80.561788,,,0,,1,
3899,3899,Columbia / University Of Waterloo,,  43.474462, -80.546591,,,0,,1,
1223,1223,University Of Waterloo Station,,  43.474023, -80.540433,,
3887,3887,Phillip / Columbia,,  43.476409, -80.539399,,,0,,1,
2524,2524,Columbia / Hazel,,  43.480027, -80.531130,,,0,,1,
4073,4073,King / Columbia,,  43.482448, -80.526106,,,0,,1,
1916,1916,King / Weber,,  43.484988, -80.526677,,,0,,1,
1918,1918,King / Manulife,,  43.491207, -80.528026,,,0,,1,
1127,1127,Conestoga Station,,  43.498036, -80.528999,,

# calendar_dates.txt
service_id,date,exception_type
201-Weekday-66-23SUMM-1111100,20230703,1
201-Weekday-66-23SUMM-1111100,20230704,1
201-Weekday-66-23SUMM-1111100,20230705,1
201-Weekday-66-23SUMM-1111100,20230706,1
201-Weekday-66-23SUMM-1111100,20230707,1
201-Weekday-66-23SUMM-1111100,20230710,1
201-Weekday-66-23SUMM-1111100,20230711,1
201-Weekday-66-23SUMM-1111100,20230712,1
201-Weekday-66-23SUMM-1111100,20230713,1
201-Weekday-66-23SUMM-1111100,20230714,1
201-Weekday-66-23SUMM-1111100,20230717,1
201-Weekday-66-23SUMM-1111100,20230718,1
201-Weekday-66-23SUMM-1111100,20230719,1
201-Weekday-66-23SUMM-1111100,20230720,1
201-Weekday-66-23SUMM-1111100,20230721,1
201-Weekday-66-23SUMM-1111100,20230724,1
201-Weekday-66-23SUMM-1111100,20230725,1
201-Weekday-66-23SUMM-1111100,20230726,1
201-Weekday-66-23SUMM-1111100,20230727,1
201-Weekday-66-23SUMM-1111100,20230728,1
201-Weekday-66-23SUMM-1111100,20230731,1
201-Weekday-66-23SUMM-1111100,20230801,1
201-Weekday-66-23SUMM-1111100,20230802,1
201-Weekday-66-23SUMM-1111100,20230803,1
201-Weekday-66-23SUMM-1111100,20230804,1
201-Weekday-66-23SUMM-1111100,20230808,1
201-Weekday-66-23SUMM-1111100,20230809,1
201-Weekday-66-23SUMM-1111100,20230810,1
201-Weekday-66-23SUMM-1111100,20230811,1
201-Weekday-66-23SUMM-1111100,20230814,1
201-Weekday-66-23SUMM-1111100,20230815,1
201-Weekday-66-23SUMM-1111100,20230816,1
201-Weekday-66-23SUMM-1111100,20230817,1
201-Weekday-66-23SUMM-1111100,20230818,1
201-Weekday-66-23SUMM-1111100,20230821,1
201-Weekday-66-23SUMM-1111100,20230822,1
201-Weekday-66-23SUMM-1111100,20230823,1
201-Weekday-66-23SUMM-1111100,20230824,1
201-Weekday-66-23SUMM-1111100,20230825,1
201-Weekday-66-23SUMM-1111100,20230828,1
201-Weekday-66-23SUMM-1111100,20230829,1
201-Weekday-66-23SUMM-1111100,20230830,1
201-Weekday-66-23SUMM-1111100,20230831,1
201-Weekday-66-23SUMM-1111100,20230901,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
201,grt,iXpress Fischer-Hallman,,3,https://www.grt.ca/en/schedules-maps/schedules.aspx
202,grt,iXpress Station-Fischer,,3,https://www.grt.ca/en/schedules-maps/schedules.aspx

# trips.txt
route_id,service_id,trip_id,trip_headsign,direction_id,block_id,shape_id,wheelchair_accessible,bikes_allowed
201,201-Weekday-66-23SUMM-1111100,3248651,Conestoga Station,0,340341,2010025,1,1
201,201-Weekday-66-23SUMM-1111100,3248652,Conestoga Station,0,340342,2010025,1,1
202,201-Weekday-66-23SUMM-1111100,3248653,Conestoga Station,0,340343,2010025,1,1


# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
3248651,05:15:00,05:15:00,2351,1,0,0
3248651,05:16:00,05:16:00,1033,2,0,0
3248651,05:18:00,05:18:00,2086,3,0,0
3248651,05:19:00,05:19:00,2885,4,0,0
3248651,05:21:00,05:21:00,2888,5,0,0
3248651,05:22:00,05:22:00,3189,6,0,0
3248651,05:24:00,05:24:00,3895,7,0,0
3248651,05:26:00,05:26:00,3893,8,0,0
3248651,05:27:00,05:27:00,2969,9,0,0
3248651,05:29:00,05:29:00,2971,10,0,0
3248651,05:31:00,05:31:00,2986,11,0,0
3248651,05:32:00,05:32:00,3891,12,0,0
3248651,05:33:00,05:33:00,3143,13,0,0
3248651,05:35:00,05:35:00,3144,14,0,0
3248651,05:37:00,05:37:00,3146,15,0,0
3248651,05:38:00,05:38:00,1992,16,0,0
3248651,05:39:00,05:39:00,1972,17,0,0
3248651,05:40:00,05:40:00,3465,18,0,0
3248651,05:42:00,05:42:00,3890,19,0,0
3248651,05:43:00,05:43:00,1117,20,0,0
3248651,05:46:00,05:46:00,3899,21,0,0
3248651,05:47:00,05:49:00,1223,22,0,0
3248651,05:50:00,05:50:00,3887,23,0,0
3248651,05:53:00,05:53:00,2524,24,0,0
3248651,05:54:00,05:54:00,4073,25,0,0
3248651,05:55:00,05:55:00,1916,26,0,0
3248651,05:56:00,05:56:00,1918,27,0,0
3248651,05:58:00,05:58:00,1127,28,1,0
3248652,06:15:00,06:15:00,2351,1,0,0
3248652,06:16:00,06:16:00,1033,2,0,0
3248652,06:18:00,06:18:00,2086,3,0,0
3248652,06:19:00,06:19:00,2885,4,0,0
3248652,06:21:00,06:21:00,2888,5,0,0
3248652,06:22:00,06:22:00,3189,6,0,0
3248652,06:24:00,06:24:00,3895,7,0,0
3248652,06:26:00,06:26:00,3893,8,0,0
3248652,06:27:00,06:27:00,2969,9,0,0
3248652,06:29:00,06:29:00,2971,10,0,0
3248652,06:31:00,06:31:00,2986,11,0,0
3248652,06:32:00,06:32:00,3891,12,0,0
3248652,06:33:00,06:33:00,3143,13,0,0
3248652,06:35:00,06:35:00,3144,14,0,0
3248652,06:37:00,06:37:00,3146,15,0,0
3248652,06:38:00,06:38:00,1992,16,0,0
3248652,06:39:00,06:39:00,1972,17,0,0
3248652,06:40:00,06:40:00,3465,18,0,0
3248652,06:42:00,06:42:00,3890,19,0,0
3248652,06:43:00,06:43:00,1117,20,0,0
3248652,06:46:00,06:46:00,3899,21,0,0
3248652,06:47:00,06:49:00,1223,22,0,0
3248652,06:50:00,06:50:00,3887,23,0,0
3248652,06:53:00,06:53:00,2524,24,0,0
3248652,06:54:00,06:54:00,4073,25,0,0
3248652,06:55:00,06:55:00,1916,26,0,0
3248652,06:56:00,06:56:00,1918,27,0,0
3248652,06:58:00,06:58:00,1127,28,1,0
3248653,07:15:00,07:15:00,2351,1,0,0
3248653,07:16:00,07:16:00,1033,2,0,0
3248653,07:18:00,07:18:00,2086,3,0,0
3248653,07:19:00,07:19:00,2885,4,0,0
3248653,07:21:00,07:21:00,2888,5,0,0
3248653,07:22:00,07:22:00,3189,6,0,0
3248653,07:24:00,07:24:00,3895,7,0,0
3248653,07:26:00,07:26:00,3893,8,0,0
3248653,07:27:00,07:27:00,2969,9,0,0
3248653,07:29:00,07:29:00,2971,10,0,0
)");
}

auto const kVehiclePosition =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "32486517",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.420946",
      "longitude": "-80.463974"
    },
    "timestamp": "1691658931",
    "vehicle": {
      "id": "v1"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486518",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.409295",
      "longitude": "-80.499004"
    },
    "timestamp": "1691659380",
    "vehicle": {
      "id": "v1"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486519",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.407145",
      "longitude": "-80.498714"
    },
    "timestamp": "1691659680",
    "vehicle": {
      "id": "v1"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486520",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.412195",
      "longitude": "-80.506761"
    },
    "timestamp": "1691659800",
    "vehicle": {
      "id": "v1"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePosition2 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "32486517",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.420946",
      "longitude": "-80.463974"
    },
    "timestamp": "1691662571",
    "vehicle": {
      "id": "v2"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486518",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.409295",
      "longitude": "-80.499004"
    },
    "timestamp": "1691662985",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486519",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.407145",
      "longitude": "-80.498714"
    },
    "stop_id": "3893",
    "timestamp": "1691663281",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486520",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.412195",
      "longitude": "-80.506761"
    },
    "timestamp": "1691663405",
    "vehicle": {
      "id": "v2"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePosition3 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "32486517",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.420946",
      "longitude": "-80.463974"
    },
    "timestamp": "1691666700",
    "vehicle": {
      "id": "v3"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486518",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.409295",
      "longitude": "-80.499004"
    },
    "timestamp": "1691666760",
    "vehicle": {
      "id": "v3"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486519",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.407145",
      "longitude": "-80.498714"
    },
    "stop_id": "3893",
    "timestamp": "1691666820",
    "vehicle": {
      "id": "v3"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486520",
    "isDeleted": false,
    "vehicle": {
    "position": {
      "latitude": "43.412195",
      "longitude": "-80.506761"
    },
    "timestamp": "1691666880",
    "vehicle": {
      "id": "v3"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

}  // namespace

TEST(rt, gtfsrt_vp_resolve_trip_test) {
  std::cout << "Test rt::gtfsrt_vp_resolve_trip_test" << std::endl;

  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 10});

  // Create empty vehicle_trip_matching
  auto vtm = vehicle_trip_matching{};

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230810");
  td.set_trip_id("3248651");
  td.set_start_time("05:15:00");
  auto const [r, t] = gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                         &rtt, source_idx_t{0}, td);

  transit_realtime::TripDescriptor td2;
  td2.set_start_date("20230810");
  td2.set_trip_id("3248652");
  td2.set_start_time("06:15:00");
  auto const [r2, t2] = gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                           &rtt, source_idx_t{0}, td2);

  transit_realtime::TripDescriptor td3;
  td3.set_start_date("20230810");
  td3.set_trip_id("3248653");
  td3.set_start_time("07:15:00");
  auto const [r3, t3] = gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                           &rtt, source_idx_t{0}, td3);

  ASSERT_TRUE(r.valid());
  ASSERT_TRUE(r2.valid());
  ASSERT_TRUE(r3.valid());

  transit_realtime::FeedMessage msg1;
  msg1.Clear();
  msg1.ParseFromArray(
      reinterpret_cast<void const*>(json_to_protobuf(kVehiclePosition).data()),
      static_cast<int>(json_to_protobuf(kVehiclePosition).size()));
  for (auto const& entity1 : msg1.entity()) {
    auto const vp_r1 =
        gtfsrt_vp_resolve_run(tt, source_idx_t{0}, entity1.vehicle(), &vtm);
    ASSERT_TRUE(vp_r1.valid());
  }

  transit_realtime::FeedMessage msg2;
  msg2.Clear();
  msg2.ParseFromArray(
      reinterpret_cast<void const*>(json_to_protobuf(kVehiclePosition2).data()),
      static_cast<int>(json_to_protobuf(kVehiclePosition2).size()));
  for (auto const& entity2 : msg2.entity()) {
    auto const vp_r2 =
        gtfsrt_vp_resolve_run(tt, source_idx_t{0}, entity2.vehicle(), &vtm);
    ASSERT_TRUE(vp_r2.valid());
  }

  transit_realtime::FeedMessage msg3;
  msg3.Clear();
  msg3.ParseFromArray(
      reinterpret_cast<void const*>(json_to_protobuf(kVehiclePosition3).data()),
      static_cast<int>(json_to_protobuf(kVehiclePosition3).size()));
  auto vp_r3 = run{};
  for (auto const& entity3 : msg3.entity()) {
    vp_r3 = gtfsrt_vp_resolve_run(tt, source_idx_t{0}, entity3.vehicle(), &vtm);
  }
  ASSERT_TRUE(vp_r3.valid());
}

TEST(rt, gtfsrt_vp_file_resolve_trip_test) {
  std::cout << "Test rt::gtfsrt_vp_file_resolve_trip_test" << std::endl;

  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2026_y / March / 25},
                    date::sys_days{2026_y / March / 30}};
  load_timetable({}, source_idx_t{0},
                 loader::zip_dir{"test/test_data/gtfs-nl.zip"},
                 tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2026_y / March / 28});

  auto vtm = vehicle_trip_matching{};

  std::ifstream f{"test/test_data/vehiclePositions.pb", std::ios::binary};
  ASSERT_TRUE(f.is_open());

  transit_realtime::FeedMessage msg;
  ASSERT_TRUE(msg.ParseFromIstream(&f));

  int counter_vps = 0;
  int num_st = 0;
  int num_sd = 0;
  int num_trip_id = 0;
  int num_route_id = 0;
  int num_ts = 0;
  int num_direction_id = 0;
  int num_vehicle_id = 0;
  int num_position = 0;
  int num_stop_id = 0;

  int counter_resolved_runs = 0;
  int counter_resolved_runs_without_trip_id = 0;
  int counter_correctly_resolved_runs_without_trip_id = 0;

  vector<std::string> trip_ids;
  vector<run> runs;

  std::cout << "VPs mit allem\n" << std::endl;
  for (auto const& entity : msg.entity()) {
    counter_vps++;
    if (entity.has_vehicle()) {
      auto vp = entity.vehicle();

      if (vp.has_stop_id()) {
        num_stop_id++;
      }
      if (vp.has_position()) {
        num_position++;
      }
      if (vp.has_timestamp()) {
        num_ts++;
      }
      if (vp.has_vehicle() && vp.vehicle().has_id()) {
        num_vehicle_id++;
      }
      if (vp.has_trip()) {
        if (vp.trip().has_start_time()) {
          num_st++;
        }
        if (vp.trip().has_start_date()) {
          num_sd++;
        }
        if (vp.trip().has_trip_id()) {
          num_trip_id++;
        }
        if (vp.trip().has_route_id()) {
          num_route_id++;
        }
        if (vp.trip().has_direction_id()) {
          num_direction_id++;
        }
      }

      run r;
      if (vp.has_trip() &&
                (vp.trip().has_trip_id() ||
                 (vp.trip().has_route_id() && vp.trip().has_direction_id() &&
                  vp.trip().has_start_date() && vp.trip().has_start_time()))) {
        r = gtfsrt_resolve_run(date::sys_days{2026_y / March / 28}, tt, &rtt, source_idx_t{0}, vp.trip()).first;
        if (r.valid()) {
          counter_resolved_runs++;
          if (!runs.contains(&r)) {
            runs.emplace_back(r);
          }
        }
      }

      vp.mutable_vehicle()->set_id(vp.trip().route_id() + vp.trip().start_time() + vp.trip().start_date() + std::to_string(vp.trip().direction_id()));
      vp.mutable_trip()->clear_trip_id();
      vp.mutable_trip()->clear_route_id();

      if (!trip_ids.contains(&vp.vehicle().id())) {
        trip_ids.emplace_back(vp.vehicle().id());
      }

      if (vp.has_vehicle() && vp.vehicle().has_id()) {
        num_vehicle_id++;
      }

      auto r2 = gtfsrt_vp_resolve_run(tt, source_idx_t{0}, vp, &vtm);
      if (r2.valid()) {
        counter_resolved_runs_without_trip_id++;
      }
      if (r == r2) {
        counter_correctly_resolved_runs_without_trip_id++;
      }
    }
  }
  std::cout << "Number of VehiclePositions: " << counter_vps << "\n";
  std::cout << "Number of VPs with start_time: " << num_st << "\n";
  std::cout << "Number of VPs with start_date: " << num_sd << "\n";
  std::cout << "Number of VPs with trip_id: " << num_trip_id << "\n";
  std::cout << "Number of VPs with route_id: " << num_route_id << "\n";
  std::cout << "Number of VPs with direction_id: " << num_direction_id << "\n";
  std::cout << "Number of VPs with vehicle_id: " << num_vehicle_id << "\n";
  std::cout << "Number of VPs with timestamp: " << num_ts << "\n";
  std::cout << "Number of VPs with position: " << num_position << "\n";
  std::cout << "Number of VPs with stop_id: " << num_stop_id << "\n";

  std::cout << "Number of different trip_ids: " << trip_ids.size() << "\n";
  std::cout << "Number of resolved runs: " << counter_resolved_runs << "\n";
  std::cout << "Number of resolved runs with candidates: " << counter_resolved_runs_without_trip_id << "\n";
  std::cout << "Number of different runs: " << runs.size() << "\n";
  std::cout << "Number of correctly resolved runs with candidates: " << counter_correctly_resolved_runs_without_trip_id << std::endl;

  for (const auto& str : trip_ids) {
    std::cout << "\n" << str;
  }
  std::cout << "\n" << std::endl;

  /**
  std::cout << "\nVPs ohne trip_id und route_id\n" << std::endl;
  int counter2_vps = 0;
  int counter2_resolved_runs = 0;
  int counter2_resolved_runs_without_trip_id = 0;
  for (auto const& entity : msg.entity()) {
    counter2_vps++;
    if (entity.has_vehicle()) {
      auto vp = entity.vehicle();
      vp.mutable_trip()->clear_trip_id();
      vp.mutable_trip()->clear_route_id();
      run r;
      if (vp.has_trip() &&
                (vp.trip().has_trip_id() ||
                 (vp.trip().has_route_id() && vp.trip().has_direction_id() &&
                  vp.trip().has_start_date() && vp.trip().has_start_time()))) {
        r = gtfsrt_resolve_run(date::sys_days{2026_y / March / 28}, tt, &rtt, source_idx_t{0}, vp.trip()).first;
        if (r.valid()) {
          counter2_resolved_runs++;
        }
      } else {
        std::cout << "unresolved vp with: " << vp.has_position()
                  << vp.has_timestamp() << vp.has_vehicle() << "\n"
                  << std::endl;
        r = gtfsrt_vp_resolve_run(tt, source_idx_t{0}, vp, &vtm);
        if (r.valid()) {
          counter2_resolved_runs++;
          counter2_resolved_runs_without_trip_id++;
        }
      }
    }
  }
  std::cout << "Number of VehiclePositions: " << counter2_vps << "\n";
  std::cout << "Number of resolved runs: " << counter2_resolved_runs << "\n";
  std::cout << "Number of resolved runs with candidates: " << counter2_resolved_runs_without_trip_id << std::endl;
  **/
}