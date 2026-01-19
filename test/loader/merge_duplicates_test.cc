#include "gtest/gtest.h"

#include "utl/zip.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/timetable.h"
#include "nigiri/timetable_metrics.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace date;

namespace {

mem_dir rbo500_a_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:27-500_3",1171,2593445697,"Hoyerswerda Bahnhof","",0,,50181,0,0
"de:von:27-500_3",1171,2593399070,"Hoyerswerda Bahnhof","",0,,89021,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:27-500_3",8197,"500","",3,"","",""
"de:von:27-500_3",7874,"500","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8197,"RBO-Busverkehr","https://www.delfi.de","Europe/Berlin","",""
7874,"RBO-Bus","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593445697,9:23:00,9:23:00,"de:14625:7501:0:7",0,0,0,""
2593445697,9:24:00,9:24:00,"de:14625:7500:3:1",1,0,0,""
2593445697,9:27:00,9:27:00,"de:14625:7502:0:1",2,0,0,""
2593445697,9:29:00,9:29:00,"de:14625:7507:0:1",3,0,0,""
2593445697,9:31:00,9:31:00,"de:14625:7578:0:1",4,0,0,""
2593445697,9:33:00,9:33:00,"de:14625:7577:0:1",5,0,0,""
2593445697,9:36:00,9:36:00,"de:14625:7652:0:1",6,0,0,""
2593445697,9:38:00,9:38:00,"de:14625:7661:0:1",7,0,0,""
2593445697,9:40:00,9:40:00,"de:14625:7770:0:1",8,0,0,""
2593445697,9:42:00,9:42:00,"de:14625:7685:0:1",9,0,0,""
2593445697,9:45:00,9:45:00,"de:14625:7687:0:1",10,0,0,""
2593445697,9:46:00,9:46:00,"de:14625:7689:0:1",11,0,0,""
2593445697,9:48:00,9:48:00,"de:14625:7692:0:1",12,0,0,""
2593445697,9:51:00,9:51:00,"de:14625:7733:0:1",13,0,0,""
2593445697,9:54:00,9:54:00,"de:14625:7720:0:1",14,0,0,""
2593445697,9:55:00,9:55:00,"de:14625:7721:0:1",15,0,0,""
2593445697,9:58:00,9:58:00,"de:14625:6969:1:1",16,0,0,""
2593445697,10:01:00,10:01:00,"de:14625:6967:1:1",17,0,0,""
2593445697,10:06:00,10:06:00,"de:14625:6954:1:1",18,0,0,""
2593445697,10:12:00,10:12:00,"de:14625:8063:0:1",19,0,0,""
2593445697,10:16:00,10:16:00,"de:14625:8044:0:1",20,0,0,""
2593445697,10:18:00,10:18:00,"de:14625:8041:1:1",21,0,0,""
2593445697,10:20:00,10:20:00,"de:14625:8010:1:2",22,0,0,""
2593445697,10:23:00,10:23:00,"de:14625:8000:3:3",23,0,0,""
2593399070,9:23:00,9:23:00,"de:14625:7501:0:7_G",0,0,0,""
2593399070,9:24:00,9:24:00,"de:14625:7500:3:1",1,0,0,""
2593399070,9:27:00,9:27:00,"de:14625:7502:0:1_G",2,0,0,""
2593399070,9:29:00,9:29:00,"de:14625:7507:0:1_G",3,0,0,""
2593399070,9:31:00,9:31:00,"de:14625:7578:0:1_G",4,0,0,""
2593399070,9:33:00,9:33:00,"de:14625:7577:0:1_G",5,0,0,""
2593399070,9:36:00,9:36:00,"de:14625:7652:0:1_G",6,0,0,""
2593399070,9:38:00,9:38:00,"de:14625:7661:0:3",7,0,0,""
2593399070,9:40:00,9:40:00,"de:14625:7770:0:1_G",8,0,0,""
2593399070,9:42:00,9:42:00,"de:14625:7685:0:1_G",9,0,0,""
2593399070,9:45:00,9:45:00,"de:14625:7687:0:1_G",10,0,0,""
2593399070,9:46:00,9:46:00,"de:14625:7689:0:1_G",11,0,0,""
2593399070,9:48:00,9:48:00,"de:14625:7692:0:1_G",12,0,0,""
2593399070,9:51:00,9:51:00,"de:14625:7733:0:1",13,0,0,""
2593399070,9:54:00,9:54:00,"de:14625:7720:0:1_G",14,0,0,""
2593399070,9:55:00,9:55:00,"de:14625:7721:0:1_G",15,0,0,""
2593399070,9:58:00,9:58:00,"de:14625:6969:1:1",16,0,0,""
2593399070,10:01:00,10:01:00,"de:14625:6967:1:1",17,0,0,""
2593399070,10:06:00,10:06:00,"de:14625:6954:1:1_G",18,0,0,""
2593399070,10:12:00,10:12:00,"de:14625:8063:0:1_G",19,0,0,""
2593399070,10:16:00,10:16:00,"de:14625:8044:0:1",20,0,0,""
2593399070,10:18:00,10:18:00,"de:14625:8041:0:1",21,0,0,""
2593399070,10:20:00,10:20:00,"de:14625:8010:1:2",22,0,0,""
2593399070,10:23:00,10:23:00,"de:14625:8000:3:3",23,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14625:8000:3:3","","Hoyerswerda Bahnhof","Bus","51.433460000000","14.231371000000",0,,0,"3","2"
"de:14625:8010:1:2","","Hoyerswerda Behördenpark","Haltestelle","51.435196000000","14.245690000000",0,,0,"2","2"
"de:14625:8041:1:1","","Hoyersw. Albert-Einstein-Str.",,"51.438769000000","14.255931000000",0,,0,"",""
"de:14625:8044:0:1","","Hoyerswerda Lausitzer Platz","Hoyerswerda Lausitzer Platz","51.438909000000","14.262749000000",0,,0,"1","2"
"de:14625:8063:0:1","","Hoyerswerda Straße E",,"51.430979000000","14.285477000000",0,,0,"",""
"de:14625:6967:1:1","","Groß Särchen Cafe","Haltestelle","51.365574000000","14.310145000000",0,,0,"1","2"
"de:14625:7721:0:1","","Caminau Kaolinwerk",,"51.337281000000","14.342223000000",0,,0,"",""
"de:14625:6954:1:1","","Maukendorf, B96",,"51.396982000000","14.294828000000",0,,0,"",""
"de:14625:7577:0:1","","Bautzen Hoyerswerdaer Straße",,"51.196612000000","14.409462000000",0,,0,"",""
"de:14625:7689:0:1","","Neudorf (b Neschwitz) B96",,"51.275680000000","14.339798000000",0,,0,"",""
"de:14625:7770:0:1","","Schwarzadler",,"51.236410000000","14.370826000000",0,,0,"",""
"de:14625:6969:1:1","","Wartha (b Königswartha) B 96","Haltestelle","51.350478000000","14.326431000000",0,,0,"1","2"
"de:14625:7685:0:1","","Luga (b Neschwitz) B 96",,"51.247304000000","14.357117000000",0,,0,"",""
"de:14625:7661:0:1","","Cölln Goldene Höhe",,"51.223349000000","14.388262000000",0,,0,"",""
"de:14625:7578:0:1","","Bautzen Abzw Seidau",,"51.191849000000","14.412400000000",0,,0,"",""
"de:14625:7502:0:1","","Bautzen Lauengraben",,"51.179670000000","14.425210000000",0,,0,"",""
"de:14625:7652:0:1","","Kleinwelka Gasthof",,"51.213283000000","14.392942000000",0,,0,"",""
"de:14625:7500:3:1","","Bautzen Bahnhof","Bus","51.173723000000","14.429764000000",0,,0,"1","2"
"de:14625:7501:0:7","","Bautzen August-Bebel-Pl (ZOB)",,"51.177395000000","14.433501000000",0,,0,"",""
"de:14625:7507:0:1","","Bautzen Fiedlerstraße",,"51.181241000000","14.414960000000",0,,0,"",""
"de:14625:7733:0:1","","Königswartha Kirchplatz","Königswartha Kirchplatz","51.309930000000","14.328767000000",0,,0,"1","2"
"de:14625:7692:0:1","","Zescha B 96",,"51.293148000000","14.327994000000",0,,0,"",""
"de:14625:7687:0:1","","Holscha B 96",,"51.266817000000","14.344775000000",0,,0,"",""
"de:14625:7720:0:1","","Caminau Dorf",,"51.328065000000","14.341298000000",0,,0,"",""
"de:14625:8063:0:1_G","","Hoyerswerda Straße E",,"51.430979000000","14.285477000000",0,,0,"",""
"de:14625:6954:1:1_G","","Maukendorf, B96",,"51.396982000000","14.294828000000",0,,0,"",""
"de:14625:7689:0:1_G","","Neudorf (b Neschwitz) B96",,"51.275579000000","14.339825000000",0,,0,"",""
"de:14625:7501:0:7_G","","Bautzen August-Bebel-Pl (ZOB)",,"51.177395000000","14.433501000000",0,,0,"",""
"de:14625:7692:0:1_G","","Zescha B 96",,"51.293098000000","14.328012000000",0,,0,"",""
"de:14625:7685:0:1_G","","Luga (b Neschwitz) B 96",,"51.247191000000","14.357171000000",0,,0,"",""
"de:14625:7770:0:1_G","","Schwarzadler",,"51.236348000000","14.370835000000",0,,0,"",""
"de:14625:7720:0:1_G","","Caminau Dorf",,"51.327913000000","14.341181000000",0,,0,"",""
"de:14625:7502:0:1_G","","Bautzen Lauengraben",,"51.179602000000","14.424958000000",0,,0,"",""
"de:14625:7661:0:3","","Cölln Goldene Höhe",,"51.224749000000","14.386394000000",0,,0,"",""
"de:14625:7577:0:1_G","","Bautzen Hoyerswerdaer Straße",,"51.196443000000","14.409480000000",0,,0,"",""
"de:14625:7578:0:1_G","","Bautzen Abzw Seidau",,"51.191781000000","14.412436000000",0,,0,"",""
"de:14625:7721:0:1_G","","Caminau Kaolinwerk",,"51.337202000000","14.342205000000",0,,0,"",""
"de:14625:7507:0:1_G","","Bautzen Fiedlerstraße",,"51.181100000000","14.415014000000",0,,0,"",""
"de:14625:7652:0:1_G","","Kleinwelka Gasthof",,"51.213204000000","14.392987000000",0,,0,"",""
"de:14625:8041:0:1","","Hoyersw. Albert-Einstein-Str.",,"51.438769000000","14.255931000000",0,,0,"",""
"de:14625:7687:0:1_G","","Holscha B 96",,"51.266778000000","14.344802000000",0,,0,"",""

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
1171,1,1,1,1,1,0,0,20240805,20241214

# calendar_dates.txt
"service_id","date","exception_type"
1171,20240805,2
1171,20240812,2
1171,20240806,2
1171,20240813,2
1171,20240807,2
1171,20241120,2
1171,20240808,2
1171,20241003,2
1171,20241031,2
1171,20240809,2

)__");
}

}  // namespace

TEST(loader, merge_intra_src) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / August / 5},
                    date::sys_days{2024_y / December / 14}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, rbo500_a_files(), tt);

  ASSERT_TRUE(!tt.bitfields_.empty() &&
              tt.bitfields_[bitfield_idx_t{0U}].none());

  finalize(tt, false, true, false);

  for (auto a = transport_idx_t{0U}; a != tt.next_transport_idx(); ++a) {
    for (auto b = transport_idx_t{0U}; b != tt.next_transport_idx(); ++b) {
      if (a != b) {
        EXPECT_TRUE((tt.bitfields_[tt.transport_traffic_days_[a]] &
                     tt.bitfields_[tt.transport_traffic_days_[b]])
                        .none());
      }
    }
  }

  auto td = transit_realtime::TripDescriptor();

  *td.mutable_trip_id() = "2593445697";
  auto const [r0, t0] = rt::gtfsrt_resolve_run(
      date::sys_days{2024_y / September / 3}, tt, nullptr, source_idx_t{0}, td);
  ASSERT_TRUE(r0.valid());

  *td.mutable_trip_id() = "2593399070";
  auto const [r1, t1] = rt::gtfsrt_resolve_run(
      date::sys_days{2024_y / September / 3}, tt, nullptr, source_idx_t{0}, td);
  ASSERT_TRUE(r1.valid());

  EXPECT_EQ(r0.t_, r1.t_);
  EXPECT_NE(t0, t1);

  for (auto [tr_range_a, tr_range_b] :
       utl::zip(tt.trip_transport_ranges_[trip_idx_t{0U}],
                tt.trip_transport_ranges_[trip_idx_t{1U}])) {
    EXPECT_EQ(tr_range_a.first, tr_range_b.first);
    EXPECT_EQ(tr_range_a.second, tr_range_b.second);
    EXPECT_FALSE(
        tt.bitfields_[tt.transport_traffic_days_[tr_range_a.first]].none());
  }
}

namespace {

mem_dir line1_2593432458_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:28-1_3",1194,2593432458,"Hoyerswerda Bahnhof","",0,,49500,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:28-1_3",8198,"1 (Hoyerswerda)","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8198,"VGH Busverkehr","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593432458,12:56:00,12:56:00,"de:14625:8090:0:1",0,0,0,""
2593432458,12:58:00,12:58:00,"de:14625:8089:0:1",1,0,0,""
2593432458,12:59:00,12:59:00,"de:14625:8088:1:1",2,0,0,""
2593432458,13:01:00,13:01:00,"de:14625:8087:1:1",3,0,0,""
2593432458,13:02:00,13:02:00,"de:14625:8086:1:1",4,0,0,""
2593432458,13:04:00,13:04:00,"de:14625:8077:0:1",5,0,0,""
2593432458,13:06:00,13:06:00,"de:14625:8075:0:1",6,0,0,""
2593432458,13:08:00,13:08:00,"de:14625:8091:0:1",7,0,0,""
2593432458,13:10:00,13:12:00,"de:14625:8044:0:3",8,0,0,""
2593432458,13:14:00,13:14:00,"de:14625:8041:1:1",9,0,0,""
2593432458,13:16:00,13:16:00,"de:14625:8040:0:2",10,0,0,""
2593432458,13:17:00,13:17:00,"de:14625:8021:0:1",11,0,0,""
2593432458,13:18:00,13:18:00,"de:14625:8023:0:1",12,0,0,""
2593432458,13:20:00,13:20:00,"de:14625:8022:0:1",13,0,0,""
2593432458,13:21:00,13:21:00,"de:14625:8020:0:1",14,0,0,""
2593432458,13:22:00,13:22:00,"de:14625:8000:3:1",15,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14625:8000:3:1","","Hoyerswerda Bahnhof","Bus","51.433343000000","14.231317000000",0,,0,"1","2"
"de:14625:8020:0:1","","Hoyerswerda Am Bahnhofsvorpl.",,"51.435051000000","14.232809000000",0,,0,"",""
"de:14625:8022:0:1","","Hoyerswerda Heinrich-Heine-Str",,"51.435566000000","14.238675000000",0,,0,"",""
"de:14625:8023:0:1","","HY Lessinghaus",,"51.436501000000","14.244963000000",0,,0,"",""
"de:14625:8021:0:1","","Hoyerswerda Zoo",,"51.439480000000","14.247496000000",0,,0,"",""
"de:14625:8040:0:2","","Hoyerswerda Am Elsterbogen","Hoyerswerda Am Elsterbogen","51.439637000000","14.251044000000",0,,0,"2","2"
"de:14625:8087:1:1","","Hoyerswerda Cottbuser Tor","Haltestelle","51.453175000000","14.267726000000",0,,0,"1","2"
"de:14625:8089:0:1","","Hoyerswerda Mittelweg",,"51.456679000000","14.261824000000",0,,0,"",""
"de:14625:8041:1:1","","Hoyersw. Albert-Einstein-Str.",,"51.438769000000","14.255931000000",0,,0,"",""
"de:14625:8091:0:1","","Hoyerswerda E.-Weinert-Straße",,"51.438892000000","14.268984000000",0,,0,"",""
"de:14625:8044:0:3","","Hoyerswerda Lausitzer Platz","Hoyerswerda Lausitzer Platz","51.438724000000","14.262983000000",0,,0,"3","2"
"de:14625:8075:0:1","","Hoyersw. von-Stauffenberg-Str.",,"51.443613000000","14.269109000000",0,,0,"",""
"de:14625:8077:0:1","","Hoyerswerda Kühnichter Heide",,"51.447487000000","14.267834000000",0,,0,"",""
"de:14625:8086:1:1","","Hoyerswerda Thomas-Müntzer-Str",,"51.449878000000","14.271831000000",0,,0,"",""
"de:14625:8088:1:1","","Hoyerswerda Käthe-Kollwitz-Str","Haltestelle","51.454977000000","14.263872000000",0,,0,"1","2"
"de:14625:8090:0:1","","Hoyerswerda Am Speicher",,"51.457177000000","14.267205000000",0,,0,"",""

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
1194,1,1,1,1,1,1,0,20240805,20241214

# calendar_dates.txt
"service_id","date","exception_type"
1194,20240805,2
1194,20240812,2
1194,20240806,2
1194,20240813,2
1194,20240807,2
1194,20241120,2
1194,20240808,2
1194,20241003,2
1194,20241031,2
1194,20240809,2
1194,20240810,2

)__");
}

mem_dir line1_2593402613_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:28-1_D_3",1194,2593402613,"Hoyerswerda Bahnhof","",0,,89918,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:28-1_D_3",14223,"1","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
14223,"VGH-Verkehrsgesellschaft Hoyerswerda","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593402613,12:56:00,12:56:00,"de:14625:8090:0:1_G",0,0,0,""
2593402613,12:58:00,12:58:00,"de:14625:8089:0:1_G",1,0,0,""
2593402613,12:59:00,12:59:00,"de:14625:8088:1:1",2,0,0,""
2593402613,13:01:00,13:01:00,"de:14625:8087:1:1",3,0,0,""
2593402613,13:03:00,13:03:00,"de:14625:8086:0:1",4,0,0,""
2593402613,13:04:00,13:04:00,"de:14625:8077:0:1_G",5,0,0,""
2593402613,13:06:00,13:06:00,"de:14625:8075:0:1_G",6,0,0,""
2593402613,13:08:00,13:08:00,"de:14625:8091:0:1_G",7,0,0,""
2593402613,13:10:00,13:12:00,"de:14625:8044:0:3",8,0,0,""
2593402613,13:14:00,13:14:00,"de:14625:8041:0:1",9,0,0,""
2593402613,13:16:00,13:16:00,"de:14625:8040:0:1",10,0,0,""
2593402613,13:17:00,13:17:00,"de:14625:8021:0:1_G",11,0,0,""
2593402613,13:18:00,13:18:00,"de:14625:8023:0:1_G",12,0,0,""
2593402613,13:20:00,13:20:00,"de:14625:8022:0:1_G",13,0,0,""
2593402613,13:21:00,13:21:00,"de:14625:8020:0:1_G",14,0,0,""
2593402613,13:22:00,13:22:00,"de:14625:8000:3:1",15,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14625:8000:3:1","","Hoyerswerda Bahnhof","Bus","51.433343000000","14.231317000000",0,,0,"1","2"
"de:14625:8020:0:1_G","","Hoyerswerda Am Bahnhofsvorpl.",,"51.435084000000","14.232755000000",0,,0,"",""
"de:14625:8023:0:1_G","","HY Lessinghaus",,"51.436501000000","14.244954000000",0,,0,"",""
"de:14625:8021:0:1_G","","Hoyerswerda Zoo",,"51.439480000000","14.247496000000",0,,0,"",""
"de:14625:8091:0:1_G","","Hoyerswerda E.-Weinert-Straße",,"51.438892000000","14.268984000000",0,,0,"",""
"de:14625:8077:0:1_G","","Hoyerswerda Kühnichter Heide",,"51.447481000000","14.267834000000",0,,0,"",""
"de:14625:8040:0:1","","Hoyerswerda Am Elsterbogen","Hoyerswerda Am Elsterbogen","51.439816000000","14.250425000000",0,,0,"1","2"
"de:14625:8022:0:1_G","","Hoyerswerda Heinrich-Heine-Str",,"51.435583000000","14.238693000000",0,,0,"",""
"de:14625:8090:0:1_G","","Hoyerswerda Am Speicher",,"51.457177000000","14.267205000000",0,,0,"",""
"de:14625:8044:0:3","","Hoyerswerda Lausitzer Platz","Hoyerswerda Lausitzer Platz","51.438724000000","14.262983000000",0,,0,"3","2"
"de:14625:8075:0:1_G","","Hoyersw. von-Stauffenberg-Str.",,"51.443613000000","14.269109000000",0,,0,"",""
"de:14625:8041:0:1","","Hoyersw. Albert-Einstein-Str.",,"51.438769000000","14.255931000000",0,,0,"",""
"de:14625:8087:1:1","","Hoyerswerda Cottbuser Tor","Haltestelle","51.453175000000","14.267726000000",0,,0,"1","2"
"de:14625:8086:0:1","","Hoyerswerda Thomas-Müntzer-Str",,"51.449878000000","14.271831000000",0,,0,"",""
"de:14625:8088:1:1","","Hoyerswerda Käthe-Kollwitz-Str","Haltestelle","51.454977000000","14.263872000000",0,,0,"1","2"
"de:14625:8089:0:1_G","","Hoyerswerda Mittelweg",,"51.456701000000","14.261788000000",0,,0,"",""

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
1194,1,1,1,0,1,1,0,20240805,20241214

# calendar_dates.txt
"service_id","date","exception_type"
1194,20240805,2
1194,20240812,2
1194,20240806,2
1194,20240813,2
1194,20240807,2
1194,20241120,2
1194,20240808,2
1194,20241003,2
1194,20241031,2
1194,20240809,2
1194,20240810,2

)__");
}

}  // namespace

TEST(loader, merge_inter_src) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / August / 5},
                    date::sys_days{2024_y / December / 14}};
  register_special_stations(tt);
  // 2593432458 and 2593402613 are mostly the same as, except no trips on
  // Thursdays for 2593402613
  load_timetable({}, source_idx_t{0}, line1_2593432458_files(), tt);
  load_timetable({}, source_idx_t{1}, line1_2593402613_files(), tt);

  ASSERT_TRUE(!tt.bitfields_.empty() &&
              tt.bitfields_[bitfield_idx_t{0U}].none());

  constexpr auto const kMetricTemplate = std::string_view{
      "["
      R"({{"idx":0,"firstDay":"2024-08-14","lastDay":"2024-12-13","noLocations":16,"noTrips":1,"transportsXDays":102}},)"
      R"({{"idx":1,"firstDay":"2024-08-14","lastDay":"2024-12-13","noLocations":16,"noTrips":1,"transportsXDays":{0}}})"
      "]"};
  // No duplicates removed; No transfers on Thursdays for 2593402613
  EXPECT_EQ(fmt::format(kMetricTemplate, 86), to_str(get_metrics(tt), tt));
  finalize(tt, false, false, true);
  // With duplicates removed; With transfers on Thursdays for both trips
  EXPECT_EQ(fmt::format(kMetricTemplate, 102), to_str(get_metrics(tt), tt));

  for (auto a = transport_idx_t{0U}; a != tt.next_transport_idx(); ++a) {
    for (auto b = transport_idx_t{0U}; b != tt.next_transport_idx(); ++b) {
      if (a != b) {
        EXPECT_TRUE((tt.bitfields_[tt.transport_traffic_days_[a]] &
                     tt.bitfields_[tt.transport_traffic_days_[b]])
                        .none());
      }
    }
  }

  auto td = transit_realtime::TripDescriptor();

  *td.mutable_trip_id() = "2593432458";
  auto const [r0, t0] = rt::gtfsrt_resolve_run(
      date::sys_days{2024_y / September / 3}, tt, nullptr, source_idx_t{0}, td);
  ASSERT_TRUE(r0.valid());

  *td.mutable_trip_id() = "2593402613";
  auto const [r1, t1] = rt::gtfsrt_resolve_run(
      date::sys_days{2024_y / September / 3}, tt, nullptr, source_idx_t{1}, td);
  ASSERT_TRUE(r1.valid());

  EXPECT_EQ(r0.t_, r1.t_);
  EXPECT_NE(t0, t1);

  for (auto [tr_range_a, tr_range_b] :
       utl::zip(tt.trip_transport_ranges_[trip_idx_t{0U}],
                tt.trip_transport_ranges_[trip_idx_t{1U}])) {
    EXPECT_EQ(tr_range_a.first, tr_range_b.first);
    EXPECT_EQ(tr_range_a.second, tr_range_b.second);
    EXPECT_FALSE(
        tt.bitfields_[tt.transport_traffic_days_[tr_range_a.first]].none());
  }
}

namespace {

mem_dir rbo500_b_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:27-500_3",1171,2593445670,"Bautzen Bahnhof","",1,,50186,0,0
"de:von:27-500_3",1171,2593399038,"Bautzen Bahnhof","",1,,89025,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:27-500_3",8197,"500","",3,"","",""
"de:von:27-500_3",7874,"500","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8197,"RBO-Busverkehr","https://www.delfi.de","Europe/Berlin","",""
7874,"RBO-Bus","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593445670,12:32:00,12:32:00,"de:14625:8000:3:3",0,0,0,""
2593445670,12:34:00,12:34:00,"de:14625:8010:1:1",1,0,0,""
2593445670,12:36:00,12:36:00,"de:14625:8041:1:2",2,0,0,""
2593445670,12:38:00,12:38:00,"de:14625:8044:0:8",3,0,0,""
2593445670,12:42:00,12:42:00,"de:14625:8063:0:2",4,0,0,""
2593445670,12:48:00,12:48:00,"de:14625:6954:1:2",5,0,0,""
2593445670,12:53:00,12:53:00,"de:14625:6967:1:2",6,0,0,""
2593445670,12:56:00,12:56:00,"de:14625:6969:1:2",7,0,0,""
2593445670,12:58:00,12:58:00,"de:14625:7721:0:2",8,0,0,""
2593445670,13:00:00,13:00:00,"de:14625:7720:0:2",9,0,0,""
2593445670,13:03:00,13:03:00,"de:14625:7733:0:2",10,0,0,""
2593445670,13:05:00,13:05:00,"de:14625:7692:0:2",11,0,0,""
2593445670,13:07:00,13:07:00,"de:14625:7689:0:2",12,0,0,""
2593445670,13:08:00,13:08:00,"de:14625:7687:0:2",13,0,0,""
2593445670,13:11:00,13:11:00,"de:14625:7685:0:2",14,0,0,""
2593445670,13:13:00,13:13:00,"de:14625:7770:0:2",15,0,0,""
2593445670,13:15:00,13:15:00,"de:14625:7661:0:2",16,0,0,""
2593445670,13:17:00,13:17:00,"de:14625:7652:0:2",17,0,0,""
2593445670,13:19:00,13:19:00,"de:14625:7577:0:2",18,0,0,""
2593445670,13:20:00,13:20:00,"de:14625:7578:0:2",19,0,0,""
2593445670,13:23:00,13:23:00,"de:14625:7507:0:2",20,0,0,""
2593445670,13:26:00,13:26:00,"de:14625:7502:0:2",21,0,0,""
2593445670,13:28:00,13:28:00,"de:14625:7555:0:2",22,0,0,""
2593445670,13:30:00,13:30:00,"de:14625:7501:0:9",23,0,0,""
2593445670,13:31:00,13:31:00,"de:14625:7500:3:1",24,0,0,""
2593399038,12:32:00,12:32:00,"de:14625:8000:3:3",0,0,0,""
2593399038,12:34:00,12:34:00,"de:14625:8010:1:1",1,0,0,""
2593399038,12:36:00,12:36:00,"de:14625:8041:0:2",2,0,0,""
2593399038,12:38:00,12:38:00,"de:14625:8044:0:8",3,0,0,""
2593399038,12:42:00,12:42:00,"de:14625:8063:0:2_G",4,0,0,""
2593399038,12:48:00,12:48:00,"de:14625:6954:1:2_G",5,0,0,""
2593399038,12:53:00,12:53:00,"de:14625:6967:1:2",6,0,0,""
2593399038,12:56:00,12:56:00,"de:14625:6969:1:2",7,0,0,""
2593399038,12:58:00,12:58:00,"de:14625:7721:0:2_G",8,0,0,""
2593399038,13:00:00,13:00:00,"de:14625:7720:0:2_G",9,0,0,""
2593399038,13:03:00,13:03:00,"de:14625:7733:0:2",10,0,0,""
2593399038,13:05:00,13:05:00,"de:14625:7692:0:2_G",11,0,0,""
2593399038,13:07:00,13:07:00,"de:14625:7689:0:2_G",12,0,0,""
2593399038,13:08:00,13:08:00,"de:14625:7687:0:2_G",13,0,0,""
2593399038,13:11:00,13:11:00,"de:14625:7685:0:2_G",14,0,0,""
2593399038,13:13:00,13:13:00,"de:14625:7770:0:2_G",15,0,0,""
2593399038,13:15:00,13:15:00,"de:14625:7661:0:4",16,0,0,""
2593399038,13:17:00,13:17:00,"de:14625:7652:0:2_G",17,0,0,""
2593399038,13:19:00,13:19:00,"de:14625:7577:0:2_G",18,0,0,""
2593399038,13:20:00,13:20:00,"de:14625:7578:0:2_G",19,0,0,""
2593399038,13:23:00,13:23:00,"de:14625:7507:0:2_G",20,0,0,""
2593399038,13:26:00,13:26:00,"de:14625:7502:0:2_G",21,0,0,""
2593399038,13:28:00,13:28:00,"de:14625:7555:0:2_G",22,0,0,""
2593399038,13:30:00,13:30:00,"de:14625:7501:0:9_G",23,0,0,""
2593399038,13:31:00,13:31:00,"de:14625:7500:3:1",24,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14625:7500:3:1","","Bautzen Bahnhof","Bus","51.173723000000","14.429764000000",0,,0,"1","2"
"de:14625:7501:0:9","","Bautzen August-Bebel-Pl (ZOB)",,"51.177017000000","14.433968000000",0,,0,"",""
"de:14625:7507:0:2","","Bautzen Fiedlerstraße",,"51.180633000000","14.415202000000",0,,0,"",""
"de:14625:7577:0:2","","Bautzen Hoyerswerdaer Straße",,"51.196189000000","14.409480000000",0,,0,"",""
"de:14625:7652:0:2","","Kleinwelka Gasthof",,"51.212827000000","14.393418000000",0,,0,"",""
"de:14625:7661:0:2","","Cölln Goldene Höhe",,"51.223225000000","14.388289000000",0,,0,"",""
"de:14625:7770:0:2","","Schwarzadler",,"51.235145000000","14.372020000000",0,,0,"",""
"de:14625:7555:0:2","","Bautzen Taucherstraße/Friedhof",,"51.180492000000","14.437436000000",0,,0,"",""
"de:14625:7502:0:2","","Bautzen Lauengraben",,"51.179698000000","14.425955000000",0,,0,"",""
"de:14625:7685:0:2","","Luga (b Neschwitz) B 96",,"51.247967000000","14.356273000000",0,,0,"",""
"de:14625:7687:0:2","","Holscha B 96",,"51.266941000000","14.344837000000",0,,0,"",""
"de:14625:7689:0:2","","Neudorf (b Neschwitz) B96",,"51.275427000000","14.339807000000",0,,0,"",""
"de:14625:7733:0:2","","Königswartha Kirchplatz","Königswartha Kirchplatz","51.309672000000","14.328452000000",0,,0,"2","2"
"de:14625:8041:1:2","","Hoyersw. Albert-Einstein-Str.",,"51.438741000000","14.256093000000",0,,0,"",""
"de:14625:8010:1:1","","Hoyerswerda Behördenpark","Haltestelle","51.435135000000","14.246202000000",0,,0,"1","2"
"de:14625:6969:1:2","","Wartha (b Königswartha) B 96","Haltestelle","51.350254000000","14.326395000000",0,,0,"2","2"
"de:14625:7578:0:2","","Bautzen Abzw Seidau",,"51.192108000000","14.411555000000",0,,0,"",""
"de:14625:7721:0:2","","Caminau Kaolinwerk",,"51.336983000000","14.342116000000",0,,0,"",""
"de:14625:6954:1:2","","Maukendorf, B96",,"51.396848000000","14.294685000000",0,,0,"",""
"de:14625:7720:0:2","","Caminau Dorf",,"51.327851000000","14.341118000000",0,,0,"",""
"de:14625:6967:1:2","","Groß Särchen Cafe","Haltestelle","51.365288000000","14.310243000000",0,,0,"2","2"
"de:14625:8063:0:2","","Hoyerswerda Straße E",,"51.430806000000","14.285944000000",0,,0,"",""
"de:14625:8044:0:8","","Hoyerswerda Lausitzer Platz","Hoyerswerda Lausitzer Platz","51.438416000000","14.263486000000",0,,0,"8","2"
"de:14625:7692:0:2","","Zescha B 96",,"51.292294000000","14.328138000000",0,,0,"",""
"de:14625:8000:3:3","","Hoyerswerda Bahnhof","Bus","51.433460000000","14.231371000000",0,,0,"3","2"
"de:14625:7501:0:9_G","","Bautzen August-Bebel-Pl (ZOB)",,"51.177006000000","14.433986000000",0,,0,"",""
"de:14625:7507:0:2_G","","Bautzen Fiedlerstraße",,"51.180734000000","14.415077000000",0,,0,"",""
"de:14625:7577:0:2_G","","Bautzen Hoyerswerdaer Straße",,"51.196330000000","14.409372000000",0,,0,"",""
"de:14625:7652:0:2_G","","Kleinwelka Gasthof",,"51.212832000000","14.393337000000",0,,0,"",""
"de:14625:7661:0:4","","Cölln Goldene Höhe",,"51.223630000000","14.387202000000",0,,0,"",""
"de:14625:8041:0:2","","Hoyersw. Albert-Einstein-Str.",,"51.438696000000","14.256057000000",0,,0,"",""
"de:14625:6954:1:2_G","","Maukendorf, B96",,"51.396848000000","14.294685000000",0,,0,"",""
"de:14625:7685:0:2_G","","Luga (b Neschwitz) B 96",,"51.248051000000","14.356183000000",0,,0,"",""
"de:14625:7502:0:2_G","","Bautzen Lauengraben",,"51.179692000000","14.425848000000",0,,0,"",""
"de:14625:7687:0:2_G","","Holscha B 96",,"51.266963000000","14.344748000000",0,,0,"",""
"de:14625:7689:0:2_G","","Neudorf (b Neschwitz) B96",,"51.275545000000","14.339726000000",0,,0,"",""
"de:14625:7578:0:2_G","","Bautzen Abzw Seidau",,"51.192114000000","14.411474000000",0,,0,"",""
"de:14625:7692:0:2_G","","Zescha B 96",,"51.292396000000","14.328066000000",0,,0,"",""
"de:14625:7721:0:2_G","","Caminau Kaolinwerk",,"51.337006000000","14.342080000000",0,,0,"",""
"de:14625:7770:0:2_G","","Schwarzadler",,"51.235207000000","14.371913000000",0,,0,"",""
"de:14625:7720:0:2_G","","Caminau Dorf",,"51.327896000000","14.341029000000",0,,0,"",""
"de:14625:7555:0:2_G","","Bautzen Taucherstraße/Friedhof",,"51.180503000000","14.437472000000",0,,0,"",""
"de:14625:8063:0:2_G","","Hoyerswerda Straße E",,"51.430806000000","14.285944000000",0,,0,"",""

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
1171,1,1,1,1,1,0,0,20240805,20241214

# calendar_dates.txt
"service_id","date","exception_type"
1171,20240805,2
1171,20240812,2
1171,20240806,2
1171,20240813,2
1171,20240807,2
1171,20241120,2
1171,20240808,2
1171,20241003,2
1171,20241031,2
1171,20240809,2

)__");
}

}  // namespace

TEST(loader, merge_reflexive_matching) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 5},
                    date::sys_days{2024_y / December / 14}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, rbo500_b_files(), tt);

  ASSERT_TRUE(!tt.bitfields_.empty() &&
              tt.bitfields_[bitfield_idx_t{0U}].none());

  finalize(tt, false, true, false);

  for (auto a = transport_idx_t{0U}; a != tt.next_transport_idx(); ++a) {
    for (auto b = transport_idx_t{0U}; b != tt.next_transport_idx(); ++b) {
      if (a != b) {
        EXPECT_TRUE((tt.bitfields_[tt.transport_traffic_days_[a]] &
                     tt.bitfields_[tt.transport_traffic_days_[b]])
                        .none());
      }
    }
  }

  auto td = transit_realtime::TripDescriptor();

  *td.mutable_trip_id() = "2593445670";
  auto const [r0, t0] = rt::gtfsrt_resolve_run(
      date::sys_days{2024_y / September / 3}, tt, nullptr, source_idx_t{0}, td);
  ASSERT_TRUE(r0.valid());

  *td.mutable_trip_id() = "2593399038";
  auto const [r1, t1] = rt::gtfsrt_resolve_run(
      date::sys_days{2024_y / September / 3}, tt, nullptr, source_idx_t{0}, td);
  ASSERT_TRUE(r1.valid());

  EXPECT_EQ(r0.t_, r1.t_);
  EXPECT_NE(t0, t1);

  for (auto [tr_range_a, tr_range_b] :
       utl::zip(tt.trip_transport_ranges_[trip_idx_t{0U}],
                tt.trip_transport_ranges_[trip_idx_t{1U}])) {
    EXPECT_EQ(tr_range_a.first, tr_range_b.first);
    EXPECT_EQ(tr_range_a.second, tr_range_b.second);
    EXPECT_FALSE(
        tt.bitfields_[tt.transport_traffic_days_[tr_range_a.first]].none());
  }
}
