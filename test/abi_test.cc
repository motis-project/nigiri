#include "gtest/gtest.h"

#include "date/date.h"

#include <cstdint>
#include "nigiri/loader/dir.h"
#include "nigiri/abi.h"
#include "nigiri/rt/util.h"

nigiri_timetable_t* nigiri_load_from_dir(nigiri::loader::dir const& d,
                                         int64_t from_ts,
                                         int64_t to_ts,
                                         unsigned link_stop_distance);
void nigiri_update_with_rt_from_buf(nigiri_timetable_t const* t,
                                    std::string_view protobuf,
                                    void (*callback)(nigiri_event_change_t,
                                                     void* context),
                                    void* context);

using namespace nigiri;
using namespace nigiri::loader;
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
1916,1916,King / Weber,,  43.484988, -80.526677,,,0,4073,1,
1918,1918,King / Manulife,,  43.491207, -80.528026,,,0,4073,1,
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

# trips.txt
route_id,service_id,trip_id,trip_headsign,direction_id,block_id,shape_id,wheelchair_accessible,bikes_allowed
201,201-Weekday-66-23SUMM-1111100,3248651,Conestoga Station,0,340341,2010025,1,1

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
)");
}

auto const kTripUpdate =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691660324"
 },
 "entity": [
  {
    "id": "3248651",
    "isDeleted": false,
    "tripUpdate": {
     "trip": {
      "tripId": "3248651",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "201"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 15,
       "arrival": {
        "time": "1691660288"
       },
       "departure": {
        "time": "1691660288"
       },
       "stopId": "3146",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 16,
       "arrival": {
        "time": "1691660351"
       },
       "departure": {
        "time": "1691660351"
       },
       "stopId": "1992",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 17,
       "arrival": {
        "time": "1691660431"
       },
       "departure": {
        "time": "1691660431"
       },
       "stopId": "1972",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 18,
       "arrival": {
        "time": "1691660496"
       },
       "departure": {
        "time": "1691660496"
       },
       "stopId": "3465",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 19,
       "arrival": {
        "time": "1691660669"
       },
       "departure": {
        "time": "1691660669"
       },
       "stopId": "3890",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 20,
       "arrival": {
        "time": "1691660718"
       },
       "departure": {
        "time": "1691660718"
       },
       "stopId": "1117",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 21,
       "arrival": {
        "time": "1691660869"
       },
       "departure": {
        "time": "1691660869"
       },
       "stopId": "3899",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 22,
       "arrival": {
        "time": "1691660943"
       },
       "departure": {
        "time": "1691660943"
       },
       "stopId": "1223",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 23,
       "arrival": {
        "time": "1691661004"
       },
       "departure": {
        "time": "1691661004"
       },
       "stopId": "3887",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 24,
       "arrival": {
        "time": "1691661152"
       },
       "departure": {
        "time": "1691661152"
       },
       "stopId": "2524",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 25,
       "arrival": {
        "time": "1691661240"
       },
       "departure": {
        "time": "1691661240"
       },
       "stopId": "4073",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 26,
       "arrival": {
        "time": "1691661276"
       },
       "departure": {
        "time": "1691661276"
       },
       "stopId": "1918",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 27,
       "arrival": {
        "time": "1691661366"
       },
       "departure": {
        "time": "1691661366"
       },
       "stopId": "1918",
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 28,
       "arrival": {
        "time": "1691661480"
       },
       "departure": {
        "time": "1691661480"
       },
       "stopId": "1127",
       "scheduleRelationship": "SCHEDULED"
      }
     ]
    }
  }
 ]
})"s;

}  // namespace

TEST(rt, abi_timetable) {

  auto const t = nigiri_load_from_dir(
      test_files(),
      std::chrono::system_clock::to_time_t(date::sys_days{2023_y / August / 9}),
      std::chrono::system_clock::to_time_t(
          date::sys_days{2023_y / August / 12}),
      0);

  auto const route_count = nigiri_get_route_count(t);
  EXPECT_EQ(1, route_count);

  auto const transport_count = nigiri_get_transport_count(t);
  EXPECT_EQ(1, transport_count);

  auto const location_count = nigiri_get_location_count(t);
  EXPECT_EQ(37, location_count);

  auto const l0 = nigiri_get_location(t, 0);
  auto const l0_name = std::string{l0->name, l0->name_len};
  EXPECT_EQ("START", l0_name);
  EXPECT_EQ(0, l0->n_footpaths);
  nigiri_destroy_location(l0);

  auto const l9 = nigiri_get_location(t, 9);
  auto const l9_name = std::string{l9->name, l9->name_len};
  EXPECT_EQ("Block Line Station", l9_name);
  auto const l9_id = std::string{l9->id, l9->id_len};
  EXPECT_EQ("2351", l9_id);
  EXPECT_EQ(0, l9->n_footpaths);
  EXPECT_FLOAT_EQ(43.422095, l9->lat);
  EXPECT_FLOAT_EQ(-80.462740, l9->lon);
  EXPECT_EQ(2, l9->transfer_time);
  EXPECT_EQ(0, l9->parent);
  nigiri_destroy_location(l9);

  auto const l35 = nigiri_get_location_with_footpaths(t, 35, false);
  auto const l35_name = std::string{l35->name, l35->name_len};
  EXPECT_EQ("King / Manulife", l35_name);
  auto const l35_id = std::string{l35->id, l35->id_len};
  EXPECT_EQ("1918", l35_id);
  EXPECT_EQ(2, l35->n_footpaths);
  EXPECT_EQ(34, l35->footpaths[0].target_location_idx);
  EXPECT_EQ(8, l35->footpaths[0].duration);
  EXPECT_FLOAT_EQ(43.491207, l35->lat);
  EXPECT_FLOAT_EQ(-80.528026, l35->lon);
  EXPECT_EQ(2, l35->transfer_time);
  EXPECT_EQ(33, l35->parent);
  nigiri_destroy_location(l35);

  auto const l36 =
      nigiri_get_location_with_footpaths(t, location_count - 1, true);
  auto const l36_name = std::string{l36->name, l36->name_len};
  EXPECT_EQ("Conestoga Station", l36_name);
  auto const l36_id = std::string{l36->id, l36->id_len};
  EXPECT_EQ("1127", l36_id);
  EXPECT_EQ(0, l36->n_footpaths);
  EXPECT_FLOAT_EQ(43.498036, l36->lat);
  EXPECT_FLOAT_EQ(-80.528999, l36->lon);
  EXPECT_EQ(2, l36->transfer_time);
  EXPECT_EQ(0, l36->parent);
  nigiri_destroy_location(l36);

  EXPECT_EQ(
      std::chrono::system_clock::to_time_t(date::sys_days{2023_y / August / 4}),
      nigiri_get_start_day_ts(t));
  EXPECT_EQ(9, nigiri_get_day_count(t));

  auto const transport = nigiri_get_transport(t, 0);
  EXPECT_EQ(0, transport->route_idx);
  EXPECT_EQ(54, transport->n_event_mams);
  EXPECT_EQ(555, transport->event_mams[0]);
  EXPECT_EQ(556, transport->event_mams[1]);
  EXPECT_EQ(556, transport->event_mams[2]);
  EXPECT_EQ(596, transport->event_mams[transport->n_event_mams - 2]);
  EXPECT_EQ(598, transport->event_mams[transport->n_event_mams - 1]);
  auto const t_name = std::string{transport->name, transport->name_len};
  EXPECT_EQ("iXpress Fischer-Hallman", t_name);

  auto const route = nigiri_get_route(t, transport->route_idx);
  EXPECT_EQ(28, route->n_stops);
  EXPECT_EQ(9, route->clasz);
  EXPECT_EQ(9, route->stops[0].location_idx);
  EXPECT_EQ(1, route->stops[0].in_allowed);
  EXPECT_EQ(1, route->stops[0].out_allowed);
  EXPECT_EQ(36, route->stops[route->n_stops - 1].location_idx);
  EXPECT_EQ(0, route->stops[route->n_stops - 1].in_allowed);
  EXPECT_EQ(1, route->stops[route->n_stops - 1].out_allowed);

  EXPECT_EQ(false, nigiri_is_transport_active(t, 0, 0));
  EXPECT_EQ(false, nigiri_is_transport_active(t, 0, 1));
  EXPECT_EQ(false, nigiri_is_transport_active(t, 0, 2));
  EXPECT_EQ(false, nigiri_is_transport_active(t, 0, 3));
  EXPECT_EQ(false, nigiri_is_transport_active(t, 0, 4));
  EXPECT_EQ(true, nigiri_is_transport_active(t, 0, 5));
  EXPECT_EQ(true, nigiri_is_transport_active(t, 0, 6));
  EXPECT_EQ(true, nigiri_is_transport_active(t, 0, 7));
  EXPECT_EQ(false, nigiri_is_transport_active(t, 0, 8));

  nigiri_destroy_route(route);
  nigiri_destroy_transport(transport);

  auto const msg = rt::json_to_protobuf(kTripUpdate);

  int test_event_change_counter = 0;

  auto const my_test_callback = [](nigiri_event_change_t evt, void* context) {
    auto test_event_change_counter_ptr = static_cast<int*>(context);

    EXPECT_EQ(0, evt.transport_idx);
    EXPECT_EQ(6, evt.day_idx);

    if (*test_event_change_counter_ptr == 0) {
      EXPECT_EQ(false, evt.stop_change);
      EXPECT_EQ(true, evt.stop_in_out_allowed);
      EXPECT_EQ(UINT32_MAX, evt.stop_location_idx);
      EXPECT_EQ(14, evt.stop_idx);
      EXPECT_EQ(false, evt.is_departure);
      EXPECT_EQ(1, evt.delay);
    }
    if (*test_event_change_counter_ptr == 1) {
      EXPECT_EQ(false, evt.stop_change);
      EXPECT_EQ(true, evt.stop_in_out_allowed);
      EXPECT_EQ(UINT32_MAX, evt.stop_location_idx);
      EXPECT_EQ(14, evt.stop_idx);
      EXPECT_EQ(true, evt.is_departure);
      EXPECT_EQ(1, evt.delay);
    }
    if (*test_event_change_counter_ptr == 8) {
      EXPECT_EQ(false, evt.stop_change);
      EXPECT_EQ(true, evt.stop_in_out_allowed);
      EXPECT_EQ(UINT32_MAX, evt.stop_location_idx);
      EXPECT_EQ(18, evt.stop_idx);
      EXPECT_EQ(false, evt.is_departure);
      EXPECT_EQ(2, evt.delay);
    }
    if (*test_event_change_counter_ptr == 9) {
      EXPECT_EQ(false, evt.stop_change);
      EXPECT_EQ(true, evt.stop_in_out_allowed);
      EXPECT_EQ(UINT32_MAX, evt.stop_location_idx);
      EXPECT_EQ(18, evt.stop_idx);
      EXPECT_EQ(true, evt.is_departure);
      EXPECT_EQ(2, evt.delay);
    }
    if (*test_event_change_counter_ptr == 22) {
      EXPECT_EQ(true, evt.stop_change);
      EXPECT_EQ(true, evt.stop_in_out_allowed);
      EXPECT_EQ(35, evt.stop_location_idx);
      EXPECT_EQ(25, evt.stop_idx);
      EXPECT_EQ(false, evt.is_departure);
      EXPECT_EQ(0, evt.delay);
    }
    if (*test_event_change_counter_ptr == 23) {
      EXPECT_EQ(true, evt.stop_change);
      EXPECT_EQ(true, evt.stop_in_out_allowed);
      EXPECT_EQ(35, evt.stop_location_idx);
      EXPECT_EQ(25, evt.stop_idx);
      EXPECT_EQ(true, evt.is_departure);
      EXPECT_EQ(0, evt.delay);
    }
    if (*test_event_change_counter_ptr == 24) {
      EXPECT_EQ(false, evt.stop_change);
      EXPECT_EQ(true, evt.stop_in_out_allowed);
      EXPECT_EQ(UINT32_MAX, evt.stop_location_idx);
      EXPECT_EQ(25, evt.stop_idx);
      EXPECT_EQ(false, evt.is_departure);
      EXPECT_EQ(-1, evt.delay);
    }
    if (*test_event_change_counter_ptr == 25) {
      EXPECT_EQ(false, evt.stop_change);
      EXPECT_EQ(true, evt.stop_in_out_allowed);
      EXPECT_EQ(UINT32_MAX, evt.stop_location_idx);
      EXPECT_EQ(25, evt.stop_idx);
      EXPECT_EQ(true, evt.is_departure);
      EXPECT_EQ(-1, evt.delay);
    }
    if (*test_event_change_counter_ptr == 26) {
      EXPECT_EQ(true, evt.stop_change);
      EXPECT_EQ(false, evt.stop_in_out_allowed);
      EXPECT_EQ(35, evt.stop_location_idx);
      EXPECT_EQ(26, evt.stop_idx);
      EXPECT_EQ(false, evt.is_departure);
      EXPECT_EQ(0, evt.delay);
    }
    if (*test_event_change_counter_ptr == 27) {
      EXPECT_EQ(true, evt.stop_change);
      EXPECT_EQ(false, evt.stop_in_out_allowed);
      EXPECT_EQ(35, evt.stop_location_idx);
      EXPECT_EQ(26, evt.stop_idx);
      EXPECT_EQ(true, evt.is_departure);
      EXPECT_EQ(0, evt.delay);
    }
    if (*test_event_change_counter_ptr == 30) {
      EXPECT_EQ(false, evt.stop_change);
      EXPECT_EQ(true, evt.stop_in_out_allowed);
      EXPECT_EQ(UINT32_MAX, evt.stop_location_idx);
      EXPECT_EQ(27, evt.stop_idx);
      EXPECT_EQ(false, evt.is_departure);
      EXPECT_EQ(0, evt.delay);
    }
    (*test_event_change_counter_ptr)++;
  };

  nigiri_update_with_rt_from_buf(t, msg, my_test_callback,
                                 &test_event_change_counter);
  EXPECT_EQ(31, test_event_change_counter);

  nigiri_destroy(t);
}

TEST(rt, abi_journeys) {

  auto const t = nigiri_load_from_dir(
      test_files(),
      std::chrono::system_clock::to_time_t(date::sys_days{2023_y / August / 9}),
      std::chrono::system_clock::to_time_t(
          date::sys_days{2023_y / August / 12}),
      0);

  auto const journeys = nigiri_get_journeys(t, 10, 15, 1691660000, false);

  ASSERT_EQ(1, journeys->n_journeys);
  EXPECT_EQ(1691659980, journeys->journeys[0].start_time);
  EXPECT_EQ(1691745840, journeys->journeys[0].dest_time);

  ASSERT_EQ(1, journeys->journeys[0].n_legs);
  auto const l0 = journeys->journeys[0].legs[0];
  EXPECT_EQ(0, l0.is_footpath);
  EXPECT_EQ(0, l0.transport_idx);
  EXPECT_EQ(7, l0.day_idx);
  EXPECT_EQ(1, l0.from_stop_idx);
  EXPECT_EQ(10, l0.from_location_idx);
  EXPECT_EQ(6, l0.to_stop_idx);
  EXPECT_EQ(15, l0.to_location_idx);
  EXPECT_EQ(8, l0.duration);

  nigiri_destroy_journeys(journeys);
  nigiri_destroy(t);
}
