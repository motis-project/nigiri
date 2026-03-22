#include "gtest/gtest.h"

#include "utl/zip.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/load.h"
#include "nigiri/flex.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable_metrics.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

namespace {

constexpr auto const kTimetable = R"__(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone,agency_lang,agency_phone
"801","PostAuto AG","https://www.postauto.ch","Europe/Zurich","DE",""

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
"14-","1","1","1","1","0","0","0","20241215","20251213"
"14-","1","1","1","1","0","0","0","20241215","20251213"
"56-","0","0","0","0","1","1","0","20241215","20251213"
"77+","0","0","0","0","0","0","1","20241215","20251213"
"55-","0","0","0","0","1","0","0","20241215","20251213"
"66-","0","0","0","0","0","1","0","20241215","20251213"
"77+","0","0","0","0","0","0","1","20241215","20251213"

# calendar_dates.txt
service_id,date,exception_type
"14-","20241225","2"
"14-","20241226","2"
"14-","20250101","2"
"14-","20250102","2"
"14-","20250421","2"
"14-","20250529","2"
"14-","20250609","2"
"56-","20250418","2"
"56-","20250801","2"
"77+","20241225","1"
"77+","20241226","1"
"77+","20250101","1"
"77+","20250102","1"
"77+","20250418","1"
"77+","20250421","1"
"77+","20250529","1"
"77+","20250609","1"
"77+","20250801","1"

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon
"ch:1:sloid:15735","Falkenhaus (Spw)","","46.90244423","7.48410718"
"ch:1:sloid:7073","Thurnen","","46.81356469","7.51387735"
"ch:1:sloid:7074","Kaufdorf","","46.83789233","7.50184890"

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
"odv_j25_1","801","Publicar Appenzell","","715"
"odv_j25_13","9039","myBuxi-Belp","","715"

# trips.txt
route_id,service_id,trip_id,trip_headsign,trip_short_name,direction_id,block_id
"odv_j25_1","14-","odv_j25_1_1_29_29_14-_1","Publicar Appenzell","Innerhalb Publicar Appenzell","",""
"odv_j25_1","56-","odv_j25_1_1_29_29_56-_3","Publicar Appenzell","Innerhalb Publicar Appenzell","",""
"odv_j25_1","77+","odv_j25_1_1_29_29_77+_4","Publicar Appenzell","Innerhalb Publicar Appenzell","",""
"odv_j25_13","14-","odv_j25_13_1_48_48_14-_54","mybuxi Belp 2025","myBuxi Belp","",""
"odv_j25_13","55-","odv_j25_13_1_48_48_55-_55","mybuxi Belp 2025","myBuxi Belp","",""
"odv_j25_13","66-","odv_j25_13_1_48_48_66-_56","mybuxi Belp 2025","myBuxi Belp","",""
"odv_j25_13","77+","odv_j25_13_1_48_48_77+_57","mybuxi Belp 2025","myBuxi Belp","",""

# booking_rules.txt
booking_rule_id,booking_type,info_url,message,phone_number,prior_notice_duration_max,prior_notice_duration_min
"booking_rule_j25_1","1","https://www.postauto.ch/de/publicar-appenzell-ai","","+41 848 55 30 60","43200","30"
"booking_rule_j25_13","1","https://mybuxi.ch/fahrgaeste/regionen/belp/","Buchungen nur via App möglich","","20160","0"

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,location_group_id, location_id,stop_sequence,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_booking_rule_id,drop_off_booking_rule_id,stop_headsign,pickup_type,drop_off_type
"odv_j25_1_1_29_29_14-_1","","","","","odv_29","2","06:00:00","19:00:00","booking_rule_j25_1","booking_rule_j25_1","","2","2"
"odv_j25_1_1_29_29_14-_1","","","","","odv_29","1","06:00:00","19:00:00","booking_rule_j25_1","booking_rule_j25_1","","2","2"
"odv_j25_1_1_29_29_56-_3","","","","","odv_29","2","06:00:00","23:30:00","booking_rule_j25_1","booking_rule_j25_1","","2","2"
"odv_j25_1_1_29_29_56-_3","","","","","odv_29","1","06:00:00","23:30:00","booking_rule_j25_1","booking_rule_j25_1","","2","2"
"odv_j25_1_1_29_29_77+_4","","","","","odv_29","2","07:00:00","19:00:00","booking_rule_j25_1","booking_rule_j25_1","","2","2"
"odv_j25_1_1_29_29_77+_4","","","","","odv_29","1","07:00:00","19:00:00","booking_rule_j25_1","booking_rule_j25_1","","2","2"
"odv_j25_13_1_48_48_14-_54","","","","odv_location_group_48","","1","06:00:00","24:00:00","booking_rule_j25_13","booking_rule_j25_13","","2","2"
"odv_j25_13_1_48_48_14-_54","","","","odv_location_group_48","","2","06:00:00","24:00:00","booking_rule_j25_13","booking_rule_j25_13","","2","2"
"odv_j25_13_1_48_48_55-_55","","","","odv_location_group_48","","1","06:00:00","24:40:00","booking_rule_j25_13","booking_rule_j25_13","","2","2"
"odv_j25_13_1_48_48_55-_55","","","","odv_location_group_48","","2","06:00:00","24:40:00","booking_rule_j25_13","booking_rule_j25_13","","2","2"
"odv_j25_13_1_48_48_66-_56","","","","odv_location_group_48","","1","07:00:00","24:40:00","booking_rule_j25_13","booking_rule_j25_13","","2","2"
"odv_j25_13_1_48_48_66-_56","","","","odv_location_group_48","","2","07:00:00","24:40:00","booking_rule_j25_13","booking_rule_j25_13","","2","2"
"odv_j25_13_1_48_48_77+_57","","","","odv_location_group_48","","1","07:00:00","24:00:00","booking_rule_j25_13","booking_rule_j25_13","","2","2"
"odv_j25_13_1_48_48_77+_57","","","","odv_location_group_48","","2","07:00:00","24:00:00","booking_rule_j25_13","booking_rule_j25_13","","2","2"

# locations.geojson
{"type":"FeatureCollection","features":[{"id":"odv_29","type":"Feature","geometry":{"type":"Polygon","coordinates":[[[9.3890941,47.3906506],[9.3791592,47.3863287],[9.3529379,47.3830743],[9.3609631,47.3739349],[9.3577015,47.3717842],[9.3624222,47.3610731],[9.3523371,47.3662182],[9.3557918,47.3627155],[9.3469298,47.3537179],[9.3447411,47.349502],[9.3542469,47.3445443],[9.3544185,47.3439336],[9.3537533,47.3411565],[9.3443549,47.3381611],[9.3299353,47.3351946],[9.3297851,47.3316026],[9.3244421,47.326905],[9.3108809,47.3241706],[9.3099797,47.3214942],[9.3142283,47.3212615],[9.3190777,47.3154137],[9.4736803,47.3123659],[9.4682515,47.3202869],[9.4891298,47.3274576],[9.4873273,47.3291593],[9.4823062,47.3366342],[9.4756114,47.3447042],[9.4429851,47.356429],[9.4447714,47.3593852],[9.4467133,47.3597231],[9.4486392,47.3600992],[9.4500473,47.3607061],[9.4504201,47.3606152],[9.4514313,47.3610058],[9.4511765,47.3613638],[9.4524318,47.3618797],[9.452877,47.3622031],[9.4549531,47.3623194],[9.4558838,47.3621486],[9.4560957,47.3620396],[9.4712663,47.3679186],[9.4750857,47.3709704],[9.4602162,47.3650771],[9.4609321,47.365087],[9.4610363,47.3658694],[9.4618732,47.3659856],[9.4601163,47.3650664],[9.4595128,47.3648665],[9.4586465,47.3640835],[9.3890941,47.3906506]]]},"properties":{"stop_name":"Publicar Appenzell"}}]}

# location_groups.txt
location_group_id,location_group_name
"odv_location_group_48","mybuxi Belp 2025"
"odv_location_group_48","ch:1:sloid:15735"
"odv_location_group_48","ch:1:sloid:7073"
"odv_location_group_48","ch:1:sloid:7074"
)__";

}  // namespace

TEST(flex, simple) {
  constexpr auto const kArea = flex_area_idx_t{0};

  auto const tt =
      loader::load({{.tag_ = "test",
                     .path_ = kTimetable,
                     .loader_config_ = {.default_tz_ = "Europe/Berlin"}}},
                   {},
                   {date::sys_days{2025_y / January / 1},
                    date::sys_days{2025_y / December / 1}});

  auto const outside = geo::latlng{47.357516806408995, 9.446811993220308};
  auto const inside = geo::latlng{47.35780140178716, 9.440695867171229};

  auto found = 0U;
  tt.flex_area_rtree_.search(outside.lnglat_float(), outside.lnglat_float(),
                             [&](auto&&, auto&&, flex_area_idx_t const a) {
                               ++found;
                               EXPECT_EQ(kArea, a);
                               return true;
                             });
  EXPECT_EQ(1U, found);

  ASSERT_EQ(1U, tt.flex_area_name_.size());
  EXPECT_FALSE(is_in_flex_area(tt, kArea, outside));
  EXPECT_TRUE(is_in_flex_area(tt, kArea, inside));
  EXPECT_EQ("Publicar Appenzell",
            tt.get_default_translation(tt.flex_area_name_[kArea]));
  EXPECT_EQ("", tt.get_default_translation(tt.flex_area_desc_[kArea]));

  auto ss = std::stringstream{};
  for (auto const& t : tt.flex_area_transports_[kArea]) {
    auto const trip_id = tt.trip_ids_[tt.flex_transport_trip_[t]].front();
    ss << "TRANSPORT " << t << " [" << tt.trip_id_strings_[trip_id].view()
       << "]\n";
    for (auto const [stop, window] :
         utl::zip(tt.flex_stop_seq_[tt.flex_transport_stop_seq_[t]],
                  tt.flex_transport_stop_time_windows_[t])) {
      stop.apply(utl::overloaded{[&](flex_area_idx_t const area) {
                                   ss << "  AREA "
                                      << tt.get_default_translation(
                                             tt.flex_area_name_[area]);
                                 },
                                 [](location_group_idx_t) {}});
      ss << ": " << window << "\n";
    }
  }

  EXPECT_EQ(R"(TRANSPORT 0 [odv_j25_1_1_29_29_14-_1]
  AREA Publicar Appenzell: [05:00.0, 18:00.0[
  AREA Publicar Appenzell: [05:00.0, 18:00.0[
TRANSPORT 1 [odv_j25_1_1_29_29_14-_1]
  AREA Publicar Appenzell: [04:00.0, 17:00.0[
  AREA Publicar Appenzell: [04:00.0, 17:00.0[
TRANSPORT 2 [odv_j25_1_1_29_29_56-_3]
  AREA Publicar Appenzell: [05:00.0, 22:30.0[
  AREA Publicar Appenzell: [05:00.0, 22:30.0[
TRANSPORT 3 [odv_j25_1_1_29_29_56-_3]
  AREA Publicar Appenzell: [04:00.0, 21:30.0[
  AREA Publicar Appenzell: [04:00.0, 21:30.0[
TRANSPORT 4 [odv_j25_1_1_29_29_77+_4]
  AREA Publicar Appenzell: [06:00.0, 18:00.0[
  AREA Publicar Appenzell: [06:00.0, 18:00.0[
TRANSPORT 5 [odv_j25_1_1_29_29_77+_4]
  AREA Publicar Appenzell: [05:00.0, 17:00.0[
  AREA Publicar Appenzell: [05:00.0, 17:00.0[
)",
            ss.str());

  // 2 x 185 (14-) + 94 (56-) + 2 x 55 (77+) + 48 (55-) + 48 (66-)
  EXPECT_EQ(
      R"([{"idx":0,"firstDay":"2025-01-01","lastDay":"2025-11-30","noLocations":3,"noTrips":7,"transportsXDays":670}])",
      to_str(get_metrics(tt), tt));
}
