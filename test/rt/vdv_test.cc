#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/vdv/vdv_update.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::rt;
using namespace date;
using namespace std::chrono_literals;

namespace {

mem_dir vdv_test_files() {
  return mem_dir::read(R"__(

# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
D,20240710,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,,,,,
B,B,,,,,,
C,C,,,,,,
D,D,,,,,,
E,E,,,,,,


# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
AE,MTA,AE,AE,A -> E,0
BC,MTA,BC,BC,B -> C,0
BD,MTA,BD,BD,B -> D,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
AE,D,AE_TRIP,AE_TRIP,1
BC,D,BC_TRIP,BC_TRIP,2
BD,D,BD_TRIP,BD_TRIP,3

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AE_TRIP,00:00,00:00,A,0,0,0
AE_TRIP,01:00,01:00,B,1,0,0
AE_TRIP,02:00,02:00,C,2,0,0
AE_TRIP,03:00,03:00,D,3,0,0
AE_TRIP,04:00,04:00,E,4,0,0
BC_TRIP,01:00,01:00,B,0,0,0
BC_TRIP,02:00,02:00,C,1,0,0
BD_TRIP,01:00,01:00,B,0,0,0
BD_TRIP,02:00,02:00,C,1,0,0
BD_TRIP,03:00,03:00,D,2,0,0

)__");
}

constexpr auto const vdv_update_msg0 = R"(
<?xml version="1.0" encoding="iso-8859-1"?>
<DatenAbrufenAntwort>
  <Bestaetigung Zst="2024-07-10T00:00:00" Ergebnis="ok" Fehlernummer="0" />
  <AUSNachricht AboID="1">
    <IstFahrt Zst="2024-07-10T00:00:00">
      <LinienID>AE</LinienID>
      <RichtungsID>1</RichtungsID>
      <FahrtRef>
        <FahrtID>
          <FahrtBezeichner>AE</FahrtBezeichner>
          <Betriebstag>2024-07-10</Betriebstag>
        </FahrtID>
      </FahrtRef>
      <Komplettfahrt>false</Komplettfahrt>
      <BetreiberID>MTA</BetreiberID>
      <IstHalt>
        <HaltID>A</HaltID>
        <Abfahrtszeit>2024-07-09T22:00:00</Abfahrtszeit>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <IstHalt>
        <HaltID>B</HaltID>
        <Ankunftszeit>2024-07-09T23:00:00</Ankunftszeit>
        <Abfahrtszeit>2024-07-09T23:00:00</Abfahrtszeit>
        <IstAnkunftPrognose>2024-07-09T23:30:00</IstAnkunftPrognose>
        <IstAbfahrtPrognose>2024-07-09T23:30:00</IstAbfahrtPrognose>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <IstHalt>
        <HaltID>D</HaltID>
        <Ankunftszeit>2024-07-10T01:00:00</Ankunftszeit>
        <Abfahrtszeit>2024-07-10T01:00:00</Abfahrtszeit>
        <IstAnkunftPrognose>2024-07-10T01:15:00</IstAnkunftPrognose>
        <IstAbfahrtPrognose>2024-07-10T01:15:00</IstAbfahrtPrognose>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <IstHalt>
        <HaltID>E</HaltID>
        <Ankunftszeit>2024-07-10T02:00:00</Ankunftszeit>
        <IstAnkunftPrognose>2024-07-10T02:00:00</IstAnkunftPrognose>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <LinienText>AE</LinienText>
      <ProduktID>Space Train</ProduktID>
      <RichtungsText>E</RichtungsText>
      <Zusatzfahrt>false</Zusatzfahrt>
      <FaelltAus>false</FaelltAus>
    </IstFahrt>
  </AUSNachricht>
</DatenAbrufenAntwort>
)";

constexpr auto const vdv_update_msg1 = R"(
<?xml version="1.0" encoding="iso-8859-1"?>
<DatenAbrufenAntwort>
  <Bestaetigung Zst="2024-07-10T00:00:00" Ergebnis="ok" Fehlernummer="0" />
  <AUSNachricht AboID="1">
    <IstFahrt Zst="2024-07-10T00:00:00">
      <LinienID>AE</LinienID>
      <RichtungsID>1</RichtungsID>
      <FahrtRef>
        <FahrtID>
          <FahrtBezeichner>AE</FahrtBezeichner>
          <Betriebstag>2024-07-10</Betriebstag>
        </FahrtID>
      </FahrtRef>
      <Komplettfahrt>false</Komplettfahrt>
      <BetreiberID>MTA</BetreiberID>
      <IstHalt>
        <HaltID>A</HaltID>
        <Abfahrtszeit>2024-07-09T22:00:00</Abfahrtszeit>
        <IstAbfahrtPrognose>2024-07-09T23:00:00</IstAbfahrtPrognose>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <LinienText>AE</LinienText>
      <ProduktID>Space Train</ProduktID>
      <RichtungsText>E</RichtungsText>
      <Zusatzfahrt>false</Zusatzfahrt>
      <FaelltAus>false</FaelltAus>
    </IstFahrt>
  </AUSNachricht>
</DatenAbrufenAntwort>
)";

constexpr auto const vdv_update_msg2 = R"(
<?xml version="1.0" encoding="iso-8859-1"?>
<DatenAbrufenAntwort>
  <Bestaetigung Zst="2024-07-10T00:00:00" Ergebnis="ok" Fehlernummer="0" />
  <AUSNachricht AboID="1">
    <IstFahrt Zst="2024-07-10T00:00:00">
      <LinienID>AE</LinienID>
      <RichtungsID>1</RichtungsID>
      <FahrtRef>
        <FahrtID>
          <FahrtBezeichner>AE</FahrtBezeichner>
          <Betriebstag>2024-07-10</Betriebstag>
        </FahrtID>
      </FahrtRef>
      <BetreiberID>MTA</BetreiberID>
      <IstHalt>
        <HaltID>A</HaltID>
        <Abfahrtszeit>2024-07-09T22:00:00</Abfahrtszeit>
        <IstAbfahrtPrognose>2024-07-09T22:01:00</IstAbfahrtPrognose>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <IstHalt>
        <HaltID>B</HaltID>
        <Ankunftszeit>2024-07-09T23:00:00</Ankunftszeit>
        <Abfahrtszeit>2024-07-09T23:00:00</Abfahrtszeit>
        <IstAnkunftPrognose>2024-07-09T22:55:00</IstAnkunftPrognose>
        <IstAbfahrtPrognose>2024-07-09T23:05:00</IstAbfahrtPrognose>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <IstHalt>
        <HaltID>C</HaltID>
        <Ankunftszeit>2024-07-10T00:00:00</Ankunftszeit>
        <Abfahrtszeit>2024-07-10T00:00:00</Abfahrtszeit>
        <IstAnkunftPrognose>2024-07-09T23:55:00</IstAnkunftPrognose>
        <IstAbfahrtPrognose>2024-07-10T00:05:00</IstAbfahrtPrognose>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <IstHalt>
        <HaltID>E</HaltID>
        <Ankunftszeit>2024-07-10T02:00:00</Ankunftszeit>
        <IstAnkunftPrognose>2024-07-10T02:07:00</IstAnkunftPrognose>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <LinienText>AE</LinienText>
      <ProduktID>Space Train</ProduktID>
      <RichtungsText>E</RichtungsText>
      <Zusatzfahrt>false</Zusatzfahrt>
      <FaelltAus>false</FaelltAus>
    </IstFahrt>
  </AUSNachricht>
</DatenAbrufenAntwort>
)";

}  // namespace

TEST(vdv_update, basic) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / July / 1},
                    date::sys_days{2024_y / July / 31}};
  load_timetable({}, source_idx_t{0}, vdv_test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / July / 10});

  auto doc = pugi::xml_document{};
  doc.load_string(vdv_update_msg0);
  rt::vdv::vdv_update(tt, rtt, source_idx_t{0}, doc);

  auto fr = rt::frun(
      tt, &rtt,
      {{transport_idx_t{0}, day_idx_t{13}}, {stop_idx_t{0}, stop_idx_t{5}}});

  EXPECT_EQ(fr[0].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 22_hours);
  EXPECT_EQ(fr[0].time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 22_hours);

  EXPECT_EQ(fr[1].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 9} + 23_hours);
  EXPECT_EQ(fr[1].time(event_type::kArr),
            date::sys_days{2024_y / July / 9} + 23_hours + 30_minutes);
  EXPECT_EQ(fr[1].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 23_hours);
  EXPECT_EQ(fr[1].time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 23_hours + 30_minutes);

  EXPECT_EQ(fr[2].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 10});
  EXPECT_EQ(fr[2].time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 30_minutes);
  EXPECT_EQ(fr[2].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 10});
  EXPECT_EQ(fr[2].time(event_type::kDep),
            date::sys_days{2024_y / July / 10} + 30_minutes);

  EXPECT_EQ(fr[3].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 1_hours);
  EXPECT_EQ(fr[3].time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 1_hours + 15_minutes);
  EXPECT_EQ(fr[3].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 10} + 1_hours);
  EXPECT_EQ(fr[3].time(event_type::kDep),
            date::sys_days{2024_y / July / 10} + 1_hours + 15_minutes);

  EXPECT_EQ(fr[4].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 2_hours);
  EXPECT_EQ(fr[4].time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 2_hours);

  doc.load_string(vdv_update_msg1);
  rt::vdv::vdv_update(tt, rtt, source_idx_t{0}, doc);

  EXPECT_EQ(fr[0].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 22_hours);
  EXPECT_EQ(fr[0].time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 23_hours);

  EXPECT_EQ(fr[1].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 9} + 23_hours);
  EXPECT_EQ(fr[1].time(event_type::kArr), date::sys_days{2024_y / July / 10});
  EXPECT_EQ(fr[1].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 23_hours);
  EXPECT_EQ(fr[1].time(event_type::kDep), date::sys_days{2024_y / July / 10});

  EXPECT_EQ(fr[2].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 10});
  EXPECT_EQ(fr[2].time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 1_hours);
  EXPECT_EQ(fr[2].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 10});
  EXPECT_EQ(fr[2].time(event_type::kDep),
            date::sys_days{2024_y / July / 10} + 1_hours);

  EXPECT_EQ(fr[3].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 1_hours);
  EXPECT_EQ(fr[3].time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 2_hours);
  EXPECT_EQ(fr[3].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 10} + 1_hours);
  EXPECT_EQ(fr[3].time(event_type::kDep),
            date::sys_days{2024_y / July / 10} + 2_hours);

  EXPECT_EQ(fr[4].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 2_hours);
  EXPECT_EQ(fr[4].time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 3_hours);

  doc.load_string(vdv_update_msg2);
  rt::vdv::vdv_update(tt, rtt, source_idx_t{0}, doc);

  EXPECT_EQ(fr[0].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 22_hours);
  EXPECT_EQ(fr[0].time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 22_hours + 1_minutes);

  EXPECT_EQ(fr[1].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 9} + 23_hours);
  EXPECT_EQ(fr[1].time(event_type::kArr),
            date::sys_days{2024_y / July / 9} + 22_hours + 55_minutes);
  EXPECT_EQ(fr[1].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 23_hours);
  EXPECT_EQ(fr[1].time(event_type::kDep),
            date::sys_days{2024_y / July / 9} + 23_hours + 05_minutes);

  EXPECT_EQ(fr[2].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 10});
  EXPECT_EQ(fr[2].time(event_type::kArr),
            date::sys_days{2024_y / July / 9} + 23_hours + 55_minutes);
  EXPECT_EQ(fr[2].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 10});
  EXPECT_EQ(fr[2].time(event_type::kDep),
            date::sys_days{2024_y / July / 10} + 5_minutes);

  EXPECT_EQ(fr[3].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 1_hours);
  EXPECT_EQ(fr[3].time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 1_hours + 5_minutes);
  EXPECT_EQ(fr[3].scheduled_time(event_type::kDep),
            date::sys_days{2024_y / July / 10} + 1_hours);
  EXPECT_EQ(fr[3].time(event_type::kDep),
            date::sys_days{2024_y / July / 10} + 1_hours + 5_minutes);

  EXPECT_EQ(fr[4].scheduled_time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 2_hours);
  EXPECT_EQ(fr[4].time(event_type::kArr),
            date::sys_days{2024_y / July / 10} + 2_hours + 7_minutes);
}

namespace {

mem_dir ev11_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:11-11:EV11_3",10408,2581118895,"A.-/Leipziger Str.","",0,,68745,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:11-11:EV11_3",8194,"EV 11","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8194,"DVB-Bus","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2581118895,17:24:00,17:24:00,"de:14612:302:1:5",0,0,0,""
2581118895,17:25:00,17:25:00,"de:14612:302:1:2",1,0,0,""
2581118895,17:26:00,17:26:00,"de:14612:301:1:1",2,0,0,""
2581118895,17:27:00,17:27:00,"de:14612:305:1:1",3,0,0,""
2581118895,17:28:00,17:28:00,"de:14612:300:2:1",4,0,0,""
2581118895,17:29:00,17:29:00,"de:14612:299:1:1",5,0,0,""
2581118895,17:31:00,17:31:00,"de:14612:298:1:1",6,0,0,""
2581118895,17:33:00,17:33:00,"de:14612:297:1:1",7,0,0,""
2581118895,17:35:00,17:35:00,"de:14612:296:1:1",8,0,0,""
2581118895,17:36:00,17:36:00,"de:14612:295:1:1",9,0,0,""
2581118895,17:37:00,17:37:00,"de:14612:294:1:1",10,0,0,""
2581118895,17:38:00,17:38:00,"de:14612:293:1:1",11,0,0,""
2581118895,17:39:00,17:39:00,"de:14612:292:2:91",12,0,0,""
2581118895,17:41:00,17:41:00,"de:14612:291:1:1",13,0,0,""
2581118895,17:43:00,17:43:00,"de:14612:290:1:1",14,0,0,""
2581118895,17:45:00,17:45:00,"de:14612:12:1:1",15,0,0,""
2581118895,17:47:00,17:47:00,"de:14612:13:2:3",16,0,0,""
2581118895,17:49:00,17:49:00,"de:14612:16:6:92",17,0,0,""
2581118895,17:52:00,17:52:00,"de:14612:17:2:9",18,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14612:17:2:9","","Dresden Anton-/Leipziger Str.","Eisenbahnstraße","51.064082000000","13.736939000000",0,,0,"5","2"
"de:14612:13:2:3","","Dresden Albertplatz","Bautzner Straße","51.063608000000","13.747655000000",0,,0,"3","2"
"de:14612:12:1:1","","DD Bautzner/Rothenburger Str.","Bautzner Str.","51.062970000000","13.752506000000",0,,0,"1","2"
"de:14612:290:1:1","","Dresden Pulsnitzer Straße","Haltestelle","51.063388000000","13.757878000000",0,,0,"1","2"
"de:14612:291:1:1","","Dresden Diakonissenkrankenhaus","Haltestelle","51.064495000000","13.763583000000",0,,0,"1","2"
"de:14612:292:2:91","","Dresden Nordstraße","SEV","51.066318000000","13.770086000000",0,,0,"91","2"
"de:14612:298:1:1","","Dresden Plattleite","Haltestelle","51.063761000000","13.821497000000",0,,0,"1","2"
"de:14612:295:1:1","","Dresden Wilhelminenstraße","Haltestelle","51.067114000000","13.788601000000",0,,0,"1","2"
"de:14612:296:1:1","","Dresden Elbschlösser","Haltestelle","51.065991000000","13.797781000000",0,,0,"1","2"
"de:14612:301:1:1","","Dresden Grundstraße","Ost-Tramstw+Bus","51.061847000000","13.848348000000",0,,0,"1","2"
"de:14612:299:1:1","","Dresden Am Weißen Adler","Haltestelle","51.061852000000","13.831549000000",0,,0,"1","2"
"de:14612:294:1:1","","Dresden Angelikastraße","Haltestelle","51.067035000000","13.782959000000",0,,0,"1","2"
"de:14612:16:6:92","","Dresden Bahnhof Neustadt","Zentralhaltest.","51.064822000000","13.741295000000",0,,0,"92","2"
"de:14612:302:1:2","","Dresden Bühlau Ullersdorfer Pl","Hst. Strab","51.062259000000","13.854187000000",0,,0,"2","2"
"de:14612:300:2:1","","Dresden Schwimmhalle Bühlau","Haltestelle Btf","51.062259000000","13.838727000000",0,,0,"1","2"
"de:14612:293:1:1","","Dresden Waldschlößchen","Haltestelle","51.067453000000","13.776940000000",0,,0,"1","2"
"de:14612:297:1:1","","Dresden Mordgrundbrücke","Haltestelle","51.064274000000","13.811696000000",0,,0,"1","2"
"de:14612:305:1:1","","Dresden Hegereiterstraße","Haltestelle","51.061960000000","13.843721000000",0,,0,"1","2"
"de:14612:302:1:5","","Dresden Bühlau Ullersdorfer Pl","Hst. Strab","51.062321000000","13.855049000000",0,,0,"5","2"

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
10408,1,1,1,1,1,0,0,20240729,20241214

# calendar_dates.txt
"service_id","date","exception_type"
10408,20240729,2
10408,20240805,2
10408,20240730,2
10408,20240731,2
10408,20241120,2
10408,20240801,2
10408,20241003,2
10408,20241031,2
10408,20240802,2

)__");
}

constexpr auto const update_rvs261 = R"(
<IstFahrt Zst="2024-08-19T17:31:43">
	<LinienID>RVS261</LinienID>
	<RichtungsID>1</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>RVS77874_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-19</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>false</Komplettfahrt>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14612:300:2:1</HaltID>
		<Abfahrtszeit>2024-08-19T15:28:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-19T15:28:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-19T15:32:41</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-19T15:32:41</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Hauptbahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:298:1:1</HaltID>
		<Abfahrtszeit>2024-08-19T15:31:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-19T15:31:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-19T15:35:41</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-19T15:35:41</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Hauptbahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:294:1:1</HaltID>
		<Abfahrtszeit>2024-08-19T15:36:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-19T15:36:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-19T15:40:11</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-19T15:40:11</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Hauptbahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:293:1:1</HaltID>
		<Abfahrtszeit>2024-08-19T15:37:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-19T15:37:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-19T15:41:11</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-19T15:41:11</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Hauptbahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:13:3:6</HaltID>
		<Abfahrtszeit>2024-08-19T15:44:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-19T15:44:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-19T15:48:41</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-19T15:48:41</IstAnkunftPrognose>
		<AbfahrtssteigText>6</AbfahrtssteigText>
		<RichtungsText>Hauptbahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:5:3:5</HaltID>
		<Abfahrtszeit>2024-08-19T15:48:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-19T15:48:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-19T15:52:41</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-19T15:52:41</IstAnkunftPrognose>
		<AbfahrtssteigText>5</AbfahrtssteigText>
		<RichtungsText>Hauptbahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:29:2:4</HaltID>
		<Abfahrtszeit>2024-08-19T15:50:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-19T15:50:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-19T15:54:41</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-19T15:54:41</IstAnkunftPrognose>
		<AbfahrtssteigText>4</AbfahrtssteigText>
		<RichtungsText>Hauptbahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:28:2:6</HaltID>
		<Abfahrtszeit>2024-08-19T15:53:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-19T15:53:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-19T15:57:41</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-19T15:57:41</IstAnkunftPrognose>
		<AbfahrtssteigText>6</AbfahrtssteigText>
		<RichtungsText>Fahrt endet hier</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:36:0:8</HaltID>
		<Ankunftszeit>2024-08-19T15:54:00</Ankunftszeit>
		<IstAnkunftPrognose>2024-08-19T15:58:41</IstAnkunftPrognose>
		<RichtungsText>Fahrt endet hier</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>261</LinienText>
	<ProduktID>RVS261</ProduktID>
	<RichtungsText>Dresden Hbf.</RichtungsText>
	<PrognoseMoeglich>true</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
	<Besetztgrad>Unbekannt</Besetztgrad>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, rvs261) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  load_timetable({}, source_idx_t{0}, ev11_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 19});

  auto doc = pugi::xml_document{};
  doc.load_string(update_rvs261);
  rt::vdv::vdv_update(tt, rtt, source_idx_t{0}, doc);

  auto fr = rt::frun(
      tt, &rtt,
      {{transport_idx_t{0}, day_idx_t{23}}, {stop_idx_t{0}, stop_idx_t{19}}});

  // ev11 and rvs261 have matching departure times but line name should prevent
  // match
  EXPECT_FALSE(fr.is_rt());
}

namespace {

mem_dir sv270_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"6703723_3",15050,2581141228,"Radeburg Meißner Berg","",1,,71016,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"6703723_3",8196,"Sv270","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8196,"VGM-Busverkehr","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2581141228,14:27:00,14:27:00,"de:14627:4172:0:1",0,0,0,""
2581141228,14:29:00,14:29:00,"de:14627:4162:0:1",1,0,0,""
2581141228,14:30:00,14:30:00,"de:14627:4167:1:1",2,0,0,""
2581141228,14:32:00,14:32:00,"de:14627:4154:0:1",3,0,0,""
2581141228,14:33:00,14:33:00,"de:14627:4155:0:1",4,0,0,""
2581141228,14:35:00,14:35:00,"de:14627:4229:1:91",5,0,0,""
2581141228,14:38:00,14:38:00,"de:14627:4226:1:1",6,0,0,""
2581141228,14:39:00,14:39:00,"de:14627:4225:1:3",7,0,0,""
2581141228,14:40:00,14:40:00,"de:14627:4233:1:1",8,0,0,""
2581141228,14:41:00,14:41:00,"de:14627:4224:1:1",9,0,0,""
2581141228,14:42:00,14:42:00,"de:14627:4223:1:1",10,0,0,""
2581141228,14:44:00,14:44:00,"de:14627:4221:1:1",11,0,0,""
2581141228,14:46:00,14:46:00,"de:14627:4260:0:1",12,0,0,""
2581141228,14:49:00,14:49:00,"de:14627:4317:0:1",13,0,0,""
2581141228,14:50:00,14:50:00,"de:14627:4328:0:1",14,0,0,""
2581141228,14:52:00,14:52:00,"de:14627:4324:0:1",15,0,0,""
2581141228,14:54:00,14:54:00,"de:14627:4323:0:1",16,0,0,""
2581141228,14:54:30,14:54:30,"de:14627:4320:0:1",17,0,0,""
2581141228,14:57:00,14:57:00,"de:14627:4312:2:2",18,0,0,""
2581141228,15:00:00,15:00:00,"de:14627:4299:0:1",19,0,0,""
2581141228,15:01:00,15:01:00,"de:14627:4298:0:1",20,0,0,""
2581141228,15:03:00,15:03:00,"de:14627:4301:2:3",21,0,0,""
2581141228,15:04:00,15:04:00,"de:14627:4309:1:2",22,0,0,""
2581141228,15:06:00,15:06:00,"de:14627:4306:1:2",23,0,0,""
2581141228,15:12:00,15:12:00,"de:14627:4398:0:1",24,0,0,""
2581141228,15:13:00,15:13:00,"de:14627:4399:0:1",25,0,0,""
2581141228,15:14:00,15:14:00,"de:14627:4392:0:1",26,0,0,""
2581141228,15:15:00,15:15:00,"de:14627:4391:0:1",27,0,0,""
2581141228,15:16:00,15:16:00,"de:14627:4390:0:1",28,0,0,""
2581141228,15:17:00,15:17:00,"de:14627:4384:0:1",29,0,0,""
2581141228,15:18:00,15:18:00,"de:14627:4381:0:1",30,0,0,""
2581141228,15:19:00,15:19:00,"de:14627:4382:0:1",31,0,0,""
2581141228,15:20:00,15:20:00,"de:14627:4387:0:1",32,0,0,""
2581141228,15:24:00,15:24:00,"de:14627:4362:0:1",33,0,0,""
2581141228,15:25:00,15:25:00,"de:14627:4363:0:1",34,0,0,""
2581141228,15:26:00,15:26:00,"de:14627:4368:0:1",35,0,0,""
2581141228,15:27:00,15:27:00,"de:14627:4377:0:1",36,0,0,""
2581141228,15:28:00,15:28:00,"de:14627:4360:0:1",37,0,0,""
2581141228,15:29:00,15:29:00,"de:14627:4367:0:1",38,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14627:4377:0:1","","Radeburg Großenhainer Platz",,"51.216805000000","13.721595000000",0,,0,"",""
"de:14627:4368:0:1","","Radeburg Rathaus","Radeburg Rathaus","51.214959000000","13.725647000000",0,,0,"1","2"
"de:14627:4381:0:1","","Berbisdorf Bärnsdorfer Straße",,"51.183133000000","13.719547000000",0,,0,"",""
"de:14627:4360:0:1","","Radeburg Busbahnhof","Radeburg Busbahnhof","51.215325000000","13.720652000000",0,,0,"1","2"
"de:14627:4392:0:1","","Bärnsdorf Siedlung",,"51.156863000000","13.732725000000",0,,0,"",""
"de:14627:4382:0:1","","Berbisdorf Landgasthof","Berbisdorf Landgasthof","51.186849000000","13.723841000000",0,,0,"1","2"
"de:14627:4384:0:1","","Berbisdorf Siedlung",,"51.177389000000","13.718568000000",0,,0,"",""
"de:14627:4390:0:1","","Bärnsdorf Zur Kleinbahn","Bärnsdorf Zur Kleinbahn","51.168490000000","13.723149000000",0,,0,"1","2"
"de:14627:4398:0:1","","Volkersdorf Grenzweg","Volkersdorf Grenzweg","51.147058000000","13.740056000000",0,,0,"1","2"
"de:14627:4306:1:2","","Boxdorf An der Triebe","Haltestelle","51.128734000000","13.703126000000",0,,0,"2","2"
"de:14627:4391:0:1","","Bärnsdorf Marsdorfer Straße",,"51.163066000000","13.730785000000",0,,0,"",""
"de:14627:4323:0:1","","Friedewald Grundschule",,"51.139585000000","13.658974000000",0,,0,"",""
"de:14627:4312:2:2","","Reichenberg Gasthof","August-Bebel-Str.","51.128182000000","13.679895000000",0,,0,"2","2"
"de:14627:4324:0:1","","Friedewald Wartehalle","Friedewald Wartehalle","51.133966000000","13.647412000000",0,,0,"1","2"
"de:14627:4229:1:91","","Radebeul Gerhart-Hauptmann-Str","SEV","51.118546000000","13.601598000000",0,,0,"91","2"
"de:14627:4317:0:1","","Friedewald Lößnitzgrundstr Süd",,"51.126643000000","13.653512000000",0,,0,"",""
"de:14627:4320:0:1","","Dippelsdorf Großenhainer Str.",,"51.140194000000","13.660869000000",0,,0,"",""
"de:14627:4260:0:1","","Radebeul Paradiesstraße",,"51.112185000000","13.656431000000",0,,0,"",""
"de:14627:4221:1:1","","Radebeul Landesbühnen Sachsen","Standard","51.106037000000","13.660222000000",0,,0,"1","2"
"de:14627:4225:1:3","","Radebeul Moritzburger Straße","Strab _stw","51.108457000000","13.629195000000",0,,0,"3","2"
"de:14627:4387:0:1","","Berbisdorf Abzweig Bärwalde",,"51.190081000000","13.725207000000",0,,0,"",""
"de:14627:4172:0:1","","Coswig Sachsenlaufweg",,"51.138870000000","13.597116000000",0,,0,"",""
"de:14627:4399:0:1","","Volkersdorf Unterer Gasthof","Volkersdorf Unterer Gasthof","51.150541000000","13.734531000000",0,,0,"1","2"
"de:14627:4155:0:1","","Coswig Melanchthonstraße","Coswig Melanchthonstraße","51.125887000000","13.586794000000",0,,0,"1","2"
"de:14627:4226:1:1","","Radebeul West (Flemmingstraße)","Hst_stadtwärts","51.110566000000","13.624505000000",0,,0,"1","2"
"de:14627:4309:1:2","","Boxdorf Feuerwehr","Haltestelle","51.122284000000","13.698607000000",0,,0,"2","2"
"de:14627:4154:0:1","","Coswig Oststraße",,"51.128813000000","13.588555000000",0,,0,"",""
"de:14627:4328:0:1","","Friedewld Lößnitzgrundstr Nord",,"51.129884000000","13.650512000000",0,,0,"",""
"de:14627:4362:0:1","","Radeburg Friedhof","Radeburg Friedhof","51.211898000000","13.730758000000",0,,0,"1","2"
"de:14627:4233:1:1","","Radebeul Gradsteg","Haltestelle","51.107385000000","13.635411000000",0,,0,"1","2"
"de:14627:4301:2:3","","Boxdorf Am Grunde","Schulstr. 475 kurz","51.118129000000","13.696371000000",0,,0,"3","2"
"de:14627:4298:0:1","","Wahnsdorf Graue Presse","Wahnsdorf Graue Presse","51.117531000000","13.677165000000",0,,0,"1","2"
"de:14627:4162:0:1","","Coswig Genossenschaftsstraße",,"51.136322000000","13.585186000000",0,,0,"",""
"de:14627:4224:1:1","","Radebeul Borstraße","Haltestelle","51.106759000000","13.639417000000",0,,0,"1","2"
"de:14627:4367:0:1","","Radeburg Meißner Berg",,"51.213615000000","13.715478000000",0,,0,"",""
"de:14627:4167:1:1","","Coswig Kastanienstraße","Haltestelle","51.132951000000","13.584512000000",0,,0,"1","2"
"de:14627:4363:0:1","","Radeburg Tankstelle",,"51.214847000000","13.728189000000",0,,0,"",""
"de:14627:4299:0:1","","Wahnsdorf Schulstraße","Haltestelle","51.118811000000","13.671487000000",0,,0,"1","2"
"de:14627:4223:1:1","","Radebeul Dr.-Külz-Straße","Standard","51.106229000000","13.646649000000",0,,0,"1","2"

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
15050,1,1,1,1,1,0,0,20240729,20241214

# calendar_dates.txt
"service_id","date","exception_type"
15050,20240729,2
15050,20240805,2
15050,20241007,2
15050,20241014,2
15050,20240730,2
15050,20241008,2
15050,20241015,2
15050,20240731,2
15050,20241009,2
15050,20241016,2
15050,20241120,2
15050,20240801,2
15050,20241003,2
15050,20241010,2
15050,20241017,2
15050,20241031,2
15050,20240802,2
15050,20241011,2
15050,20241018,2

)__");
}

constexpr auto const update_vgm270 = R"(
<IstFahrt Zst="2024-08-20T15:18:58">
	<LinienID>VGM270</LinienID>
	<RichtungsID>2</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>VGM27000402_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-20</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>true</Komplettfahrt>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14627:4172:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:27:00</Abfahrtszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4162:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:29:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:29:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4167:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:30:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:30:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4154:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:32:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:32:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4155:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:33:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:33:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4229:1:91</HaltID>
		<Abfahrtszeit>2024-08-20T12:35:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:35:00</Ankunftszeit>
		<AbfahrtssteigText>91</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4226:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:38:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:38:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4225:1:3</HaltID>
		<Abfahrtszeit>2024-08-20T12:39:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:39:00</Ankunftszeit>
		<AbfahrtssteigText>3</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4233:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:40:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:40:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4224:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:41:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:41:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4223:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:42:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:42:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4221:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:44:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:44:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4260:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:46:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:46:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4317:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:49:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:49:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4328:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:50:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:50:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4324:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:52:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:52:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4323:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:54:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:54:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4320:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T12:54:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:54:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4312:2:2</HaltID>
		<Abfahrtszeit>2024-08-20T12:57:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T12:57:01</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4299:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:00:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:00:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4298:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:01:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:01:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4301:2:3</HaltID>
		<Abfahrtszeit>2024-08-20T13:03:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:03:01</Ankunftszeit>
		<AbfahrtssteigText>3</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4309:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T13:04:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:04:01</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4306:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T13:06:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:06:01</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4398:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:12:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:12:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4399:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:13:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:13:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4392:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:14:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:14:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4391:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:15:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:15:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4390:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:16:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:16:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4384:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:17:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:17:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4381:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:18:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:18:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4382:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:19:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:19:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4387:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:20:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:20:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4362:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:24:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:24:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4363:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:25:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:25:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4368:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:26:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:26:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4377:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:27:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:27:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4360:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T13:28:01</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T13:28:01</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4367:0:1</HaltID>
		<Ankunftszeit>2024-08-20T13:29:01</Ankunftszeit>
		<AnkunftssteigText>1</AnkunftssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>270</LinienText>
	<ProduktID>VGM270</ProduktID>
	<RichtungsText>270 Schulbus - Radeburg</RichtungsText>
	<PrognoseMoeglich>true</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, vgm270) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  load_timetable({}, source_idx_t{0}, sv270_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 20});

  auto doc = pugi::xml_document{};
  doc.load_string(update_vgm270);
  rt::vdv::vdv_update(tt, rtt, source_idx_t{0}, doc);

  auto fr = rt::frun(
      tt, &rtt,
      {{transport_idx_t{0}, day_idx_t{24}}, {stop_idx_t{0}, stop_idx_t{39}}});

  std::cout << fr << "\n";

  // Sv270 from GTFS and VGM270 from the VDV update should match even though:
  // - Prefix of line name does not match
  // - the VDV update has minor time differences
  EXPECT_TRUE(fr.is_rt());
}

namespace {

// manually added stops from VDV update to stops.txt, originally they were
// missing in the GTFS timetable
mem_dir smd712_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:von:69-712_3",15131,2586108206,"Königswartha Kirchplatz","",0,,73059,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:von:69-712_3",7891,"712","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
7891,"Schmidt-Reisen","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2586108206,18:00:00,18:00:00,"de:14625:7501:0:9_G",0,0,0,""
2586108206,18:03:00,18:03:00,"de:14625:7502:0:1_G",1,0,0,""
2586108206,18:05:00,18:05:00,"de:14625:7507:0:1_G",2,0,0,""
2586108206,18:08:00,18:08:00,"de:14625:7578:0:1_G",3,0,0,""
2586108206,18:09:00,18:09:00,"de:14625:7577:0:1_G",4,0,0,""
2586108206,18:12:00,18:12:00,"de:14625:7652:0:1_G",5,0,0,""
2586108206,18:13:00,18:13:00,"de:14625:7651:0:E1",6,0,0,""
2586108206,18:20:00,18:20:00,"de:14625:7648:0:1",7,0,0,""
2586108206,18:22:00,18:22:00,"de:14625:7667:0:1",8,0,0,""
2586108206,18:24:00,18:24:00,"de:14625:7668:0:1",9,0,0,""
2586108206,18:27:00,18:27:00,"de:14625:7670:0:1",10,0,0,""
2586108206,18:29:00,18:29:00,"de:14625:7673:0:1",11,0,0,""
2586108206,18:31:00,18:31:00,"de:14625:7682:0:1",12,0,0,""
2586108206,18:33:00,18:33:00,"de:14625:7675:0:1",13,0,0,""
2586108206,18:36:00,18:36:00,"de:14625:7677:0:1_G",14,0,0,""
2586108206,18:40:00,18:40:00,"de:14625:7691:0:1_G",15,0,0,""
2586108206,18:42:00,18:42:00,"de:14625:7716:0:1_G",16,0,0,""
2586108206,18:43:00,18:43:00,"de:14625:7731:0:1_G",17,0,0,""
2586108206,18:44:00,18:44:00,"de:14625:7732:0:1_G",18,0,0,""
2586108206,18:45:00,18:45:00,"de:14625:7733:0:2",19,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14625:7731:0:1_G","","Königswartha Gewerbegebiet",,"51.309200000000","14.312678000000",0,,0,"",""
"de:14625:7731:0:1","","Königswartha Gewerbegebiet",,"51.309178000000","14.312651000000",0,,0,"",""
"de:14625:7716:0:1_G","","Niesendorf",,"51.298670000000","14.316972000000",0,,0,"",""
"de:14625:7716:0:1","","Niesendorf",,"51.298737000000","14.316945000000",0,,0,"",""
"de:14625:7691:0:1_G","","Zescha Dorfclub",,"51.287137000000","14.319101000000",0,,0,"",""
"de:14625:7691:0:1","","Zescha Dorfclub",,"51.287182000000","14.319047000000",0,,0,"",""
"de:14625:7677:0:1_G","","Neschwitz Dorfschänke",,"51.269335000000","14.329350000000",0,,0,"",""
"de:14625:7677:0:1","","Neschwitz Dorfschänke",,"51.269375000000","14.329324000000",0,,0,"",""
"de:14625:7675:0:1","","Uebigau (b Neschwitz)",,"51.253359000000","14.333923000000",0,,0,"",""
"de:14625:7682:0:1","","Krinitz b Bautzen Dorfplatz",,"51.243620000000","14.337804000000",0,,0,"",""
"de:14625:7732:0:1_G","","Königswartha Bahnhof",,"51.310222000000","14.322757000000",0,,0,"",""
"de:14625:7732:0:1","","Königswartha Bahnhof",,"51.310290000000","14.322847000000",0,,0,"",""
"de:14625:7670:0:1","","Loga An der Schanze",,"51.231106000000","14.332171000000",0,,0,"",""
"de:14625:7502:0:1_G","","Bautzen Lauengraben",,"51.179602000000","14.424958000000",0,,0,"",""
"de:14625:7502:0:1","","Bautzen Lauengraben",,"51.179670000000","14.425210000000",0,,0,"",""
"de:14625:7577:0:1_G","","Bautzen Hoyerswerdaer Straße",,"51.196443000000","14.409480000000",0,,0,"",""
"de:14625:7577:0:1","","Bautzen Hoyerswerdaer Straße",,"51.196612000000","14.409462000000",0,,0,"",""
"de:14625:7668:0:1","","Milkwitz",,"51.230021000000","14.358950000000",0,,0,"",""
"de:14625:7667:0:1","","Großbrösern",,"51.223619000000","14.365023000000",0,,0,"",""
"de:14625:7651:0:E1","","Großwelka",,"51.211352000000","14.390472000000",0,,0,"",""
"de:14625:7651:0:1","","Großwelka",,"51.210756000000","14.383672000000",0,,0,"",""
"de:14625:7733:0:2","","Königswartha Kirchplatz","Königswartha Kirchplatz","51.309672000000","14.328452000000",0,,0,"2","2"
"de:14625:7733:0:1","","Königswartha Kirchplatz","Königswartha Kirchplatz","51.309930000000","14.328767000000",0,,0,"1","2"
"de:14625:7578:0:1_G","","Bautzen Abzw Seidau",,"51.191781000000","14.412436000000",0,,0,"",""
"de:14625:7578:0:1","","Bautzen Abzw Seidau",,"51.191849000000","14.412400000000",0,,0,"",""
"de:14625:7507:0:1_G","","Bautzen Fiedlerstraße",,"51.181100000000","14.415014000000",0,,0,"",""
"de:14625:7507:0:1","","Bautzen Fiedlerstraße",,"51.181241000000","14.414960000000",0,,0,"",""
"de:14625:7673:0:1","","Saritsch",,"51.235657000000","14.332755000000",0,,0,"",""
"de:14625:7501:0:9_G","","Bautzen August-Bebel-Pl (ZOB)",,"51.177006000000","14.433986000000",0,,0,"",""
"de:14625:7501:0:9","","Bautzen August-Bebel-Pl (ZOB)",,"51.177017000000","14.433968000000",0,,0,"",""
"de:14625:7652:0:1_G","","Kleinwelka Gasthof",,"51.213204000000","14.392987000000",0,,0,"",""
"de:14625:7652:0:1","","Kleinwelka Gasthof",,"51.213283000000","14.392942000000",0,,0,"",""
"de:14625:7648:0:1","","Schmochtitz",,"51.213429000000","14.362040000000",0,,0,"",""

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
15131,1,1,1,1,1,0,0,20240729,20241214

# calendar_dates.txt
"service_id","date","exception_type"
15131,20240729,2
15131,20240805,2
15131,20240730,2
15131,20240806,2
15131,20240731,2
15131,20241120,2
15131,20240801,2
15131,20241003,2
15131,20241031,2
15131,20240802,2

)__");
}

constexpr auto const update_smd712 = R"(
<IstFahrt Zst="2024-08-20T18:29:08">
	<LinienID>SMD712</LinienID>
	<RichtungsID>1</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>SMD13712015_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-20</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>true</Komplettfahrt>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14625:7501:0:9</HaltID>
		<Abfahrtszeit>2024-08-20T16:00:00</Abfahrtszeit>
		<AbfahrtssteigText>9</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7502:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:03:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:03:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7507:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:05:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:05:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7578:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:08:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:08:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7577:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:09:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:09:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7652:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:12:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:12:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7651:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:14:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:14:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7648:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:16:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:16:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7667:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:18:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:18:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7668:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:21:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:21:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7670:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:24:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:24:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7673:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:26:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:26:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7682:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:28:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:28:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7675:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:30:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:30:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7677:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:33:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:33:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7691:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:37:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:37:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7716:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:39:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:39:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7731:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:40:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:40:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7732:0:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:41:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:41:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7733:0:1</HaltID>
		<Ankunftszeit>2024-08-20T16:42:00</Ankunftszeit>
		<AnkunftssteigText>1</AnkunftssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>712</LinienText>
	<ProduktID>SMD712</ProduktID>
	<RichtungsText>Königswartha</RichtungsText>
	<PrognoseMoeglich>false</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, smd712) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  load_timetable({}, source_idx_t{0}, smd712_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 20});

  auto doc = pugi::xml_document{};
  doc.load_string(update_smd712);
  rt::vdv::vdv_update(tt, rtt, source_idx_t{0}, doc);

  auto fr = rt::frun(
      tt, &rtt,
      {{transport_idx_t{0}, day_idx_t{24}}, {stop_idx_t{0}, stop_idx_t{20}}});

  std::cout << fr << "\n";

  // The reference time of line 712 in the update is from the year-long
  // timetable. However, the GTFS timetable contains different times due to the
  // line being redirected starting from June 29
  // the time differences is up to 4 minutes
  EXPECT_FALSE(fr.is_rt());
}

namespace {

mem_dir rbo920_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:27-920_3",4047,2593448036,"Kamenz Bahnhof","",0,,50455,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:27-920_3",8197,"920","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8197,"RBO-Busverkehr","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593448036,18:32:00,18:32:00,"de:14625:6201:2:7",0,0,0,""
2593448036,18:34:00,18:34:00,"de:14625:6224:1:2",1,0,0,""
2593448036,18:36:00,18:36:00,"de:14625:6200:1:1",2,0,0,""
2593448036,18:37:00,18:37:00,"de:14625:6237:1:1",3,0,0,""
2593448036,18:38:00,18:38:00,"de:14625:6238:1:1",4,0,0,""
2593448036,18:40:00,18:40:00,"de:14625:6249:1:1",5,0,0,""
2593448036,18:41:00,18:41:00,"de:14625:6211:1:2",6,0,0,""
2593448036,18:43:00,18:43:00,"de:14625:6215:1:1",7,0,0,""
2593448036,18:44:00,18:44:00,"de:14625:6250:1:1",8,0,0,""
2593448036,18:45:00,18:45:00,"de:14625:6231:1:2",9,0,0,""
2593448036,18:46:00,18:46:00,"de:14625:6230:1:2",10,0,0,""
2593448036,18:47:00,18:47:00,"de:14625:6229:1:2",11,0,0,""
2593448036,18:49:00,18:49:00,"de:14625:6221:1:2",12,0,0,""
2593448036,18:55:00,18:55:00,"de:14625:6223:1:2",13,0,0,""
2593448036,18:57:00,18:57:00,"de:14625:6207:1:2",14,0,0,""
2593448036,18:58:00,18:58:00,"de:14625:6230:1:1",15,0,0,""
2593448036,18:59:00,18:59:00,"de:14625:6231:1:1",16,0,0,""
2593448036,19:00:00,19:00:00,"de:14625:6250:1:2",17,0,0,""
2593448036,19:01:00,19:01:00,"de:14625:6215:1:2",18,0,0,""
2593448036,19:03:00,19:03:00,"de:14625:6211:1:1",19,0,0,""
2593448036,19:04:00,19:04:00,"de:14625:6249:1:2",20,0,0,""
2593448036,19:06:00,19:06:00,"de:14625:6238:1:2",21,0,0,""
2593448036,19:07:00,19:07:00,"de:14625:6237:1:2",22,0,0,""
2593448036,19:08:00,19:08:00,"de:14625:6200:1:6",23,0,0,""
2593448036,19:10:00,19:10:00,"de:14625:6224:1:1",24,0,0,""
2593448036,19:12:00,19:12:00,"de:14625:6201:2:8",25,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14625:6201:2:8","","Kamenz Bahnhof","Bus","51.274708000000","14.092617000000",0,,0,"8","2"
"de:14625:6224:1:1","","Kamenz Oststraße","Haltestelle","51.275956000000","14.098519000000",0,,0,"1","2"
"de:14625:6200:1:6","","Kamenz Macherstraße","Haltestelle","51.278333000000","14.102319000000",0,,0,"6","2"
"de:14625:6237:1:2","","Kamenz Nordostvorstadt","Haltestelle","51.279642000000","14.100532000000",0,,0,"2","2"
"de:14625:6238:1:2","","Kamenz Nordstraße/Stadion","Haltestelle","51.281771000000","14.098978000000",0,,0,"2","2"
"de:14625:6249:1:2","","Kamenz Friedensstraße","Haltestelle","51.289311000000","14.101942000000",0,,0,"2","2"
"de:14625:6237:1:1","","Kamenz Nordostvorstadt","Haltestelle","51.279451000000","14.100792000000",0,,0,"1","2"
"de:14625:6231:1:1","","Kamenz Jesau Jan-Skala-Straße","Haltestelle","51.281417000000","14.118723000000",0,,0,"1","2"
"de:14625:6250:1:1","","Kamenz Jesau Neschwitzer Str","Haltestelle","51.284143000000","14.122594000000",0,,0,"1","2"
"de:14625:6230:1:1","","Kamenz Jesau Elsteraue","Haltestelle","51.278305000000","14.114680000000",0,,0,"1","2"
"de:14625:6221:1:2","","Kamenz Andreas-Günther-Straße","Haltestelle","51.273089000000","14.116585000000",0,,0,"2","2"
"de:14625:6207:1:2","","Kamenz Netto-Markt","Haltestelle","51.273208000000","14.108311000000",0,,0,"2","2"
"de:14625:6229:1:2","","Kamenz Jesauer Straße","Haltestelle","51.276006000000","14.114105000000",0,,0,"2","2"
"de:14625:6224:1:2","","Kamenz Oststraße","Haltestelle","51.275927000000","14.098753000000",0,,0,"2","2"
"de:14625:6230:1:2","","Kamenz Jesau Elsteraue","Haltestelle","51.278276000000","14.114752000000",0,,0,"2","2"
"de:14625:6231:1:2","","Kamenz Jesau Jan-Skala-Straße","Haltestelle","51.281249000000","14.118255000000",0,,0,"2","2"
"de:14625:6211:1:2","","Kamenz Schwimmhalle","Haltestelle","51.286278000000","14.111141000000",0,,0,"2","2"
"de:14625:6249:1:1","","Kamenz Friedensstraße","Haltestelle","51.289261000000","14.101915000000",0,,0,"1","2"
"de:14625:6250:1:2","","Kamenz Jesau Neschwitzer Str","Haltestelle","51.284064000000","14.122666000000",0,,0,"2","2"
"de:14625:6238:1:1","","Kamenz Nordstraße/Stadion","Haltestelle","51.282176000000","14.098879000000",0,,0,"1","2"
"de:14625:6200:1:1","","Kamenz Macherstraße","Haltestelle","51.278052000000","14.101906000000",0,,0,"1","2"
"de:14625:6211:1:1","","Kamenz Schwimmhalle","Haltestelle","51.286845000000","14.111572000000",0,,0,"1","2"
"de:14625:6215:1:1","","Kamenz Schule Neschwitzer Str.","Haltestelle","51.285064000000","14.113997000000",0,,0,"1","2"
"de:14625:6215:1:2","","Kamenz Schule Neschwitzer Str.","Haltestelle","51.285384000000","14.115102000000",0,,0,"2","2"
"de:14625:6223:1:2","","Kamenz Forststraße","Haltestelle","51.267419000000","14.112740000000",0,,0,"2","2"
"de:14625:6201:2:7","","Kamenz Bahnhof","Bus","51.274579000000","14.092716000000",0,,0,"7","2"

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
4047,0,0,0,0,0,0,0,20240805,20241214

# calendar_dates.txt
"service_id","date","exception_type"
4047,20240819,1
4047,20240820,1
4047,20240821,1
4047,20240822,1
4047,20240816,1
4047,20240817,1
4047,20240818,1

)__");
}

constexpr auto const update_rbo920 = R"(
<IstFahrt Zst="2024-08-20T18:30:08">
	<LinienID>RBO920</LinienID>
	<RichtungsID>1</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>RBO13655_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-20</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>true</Komplettfahrt>
	<UmlaufID>18901</UmlaufID>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14625:6201:2:7</HaltID>
		<Abfahrtszeit>2024-08-20T16:32:00</Abfahrtszeit>
		<AbfahrtssteigText>7</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6224:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T16:34:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:34:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6200:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:36:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:36:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6237:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:37:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:37:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6238:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:38:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:38:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6249:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:40:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:40:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6211:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T16:41:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:41:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6215:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:43:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:43:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6250:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:44:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:44:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6231:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T16:45:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:45:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6230:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T16:46:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:46:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6229:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T16:47:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:47:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6221:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T16:49:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:49:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6223:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T16:55:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:51:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6207:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T16:57:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:57:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6230:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:58:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:58:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6231:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T16:59:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T16:59:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6250:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T17:00:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T17:00:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6215:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T17:01:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T17:01:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6211:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T17:03:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T17:03:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6249:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T17:04:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T17:04:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6238:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T17:06:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T17:06:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6237:1:2</HaltID>
		<Abfahrtszeit>2024-08-20T17:07:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T17:07:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6200:1:6</HaltID>
		<Abfahrtszeit>2024-08-20T17:08:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T17:08:00</Ankunftszeit>
		<AbfahrtssteigText>6</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6224:1:1</HaltID>
		<Abfahrtszeit>2024-08-20T17:10:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-20T17:10:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:6201:2:8</HaltID>
		<Ankunftszeit>2024-08-20T17:12:00</Ankunftszeit>
		<AnkunftssteigText>8</AnkunftssteigText>
		<RichtungsText>Pendelverkehr Forstfest Ri. Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>920</LinienText>
	<ProduktID>RBO920</ProduktID>
	<RichtungsText>Pendelverkehr Forstfest Ri. Forststrasse</RichtungsText>
	<PrognoseMoeglich>true</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, rbo920) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  load_timetable({}, source_idx_t{0}, rbo920_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 20});

  auto doc = pugi::xml_document{};
  doc.load_string(update_rbo920);
  rt::vdv::vdv_update(tt, rtt, source_idx_t{0}, doc);

  auto fr = rt::frun(
      tt, &rtt,
      {{transport_idx_t{0}, day_idx_t{24}}, {stop_idx_t{0}, stop_idx_t{26}}});

  std::cout << fr << "\n";

  EXPECT_TRUE(fr.is_rt());
}