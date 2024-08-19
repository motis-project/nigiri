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

mem_dir rbo739_files() {
  return mem_dir::read(R"__(

# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:von:27-734_3",15145,2586107687,"Wilthen","",0,,72868,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:von:27-734_3",7874,"734","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
7874,"RBO-Bus","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2586107687,7:10:00,7:10:00,"de:14625:7000:2:6",0,0,0,""
2586107687,7:12:00,7:12:00,"de:14625:7011:0:1_G",1,0,0,""
2586107687,7:13:00,7:13:00,"de:14625:7034:0:2_G",2,0,0,""
2586107687,7:14:00,7:14:00,"de:14625:7035:0:2_G",3,0,0,""
2586107687,7:16:00,7:16:00,"de:14625:7048:0:2",4,0,0,""
2586107687,7:17:00,7:17:00,"de:14625:7049:0:2",5,0,0,""
2586107687,7:20:00,7:20:00,"de:14625:7174:0:2",6,0,0,""
2586107687,7:22:00,7:22:00,"de:14625:7185:0:1",7,0,0,""
2586107687,7:24:00,7:24:00,"de:14625:7186:0:1",8,0,0,""
2586107687,7:27:00,7:27:00,"de:14625:7180:0:2",9,0,0,""
2586107687,7:30:00,7:30:00,"de:14625:7178:0:2",10,0,0,""
2586107687,7:32:00,7:32:00,"de:14625:7430:0:1",11,0,0,""
2586107687,7:34:00,7:34:00,"de:14625:7433:0:1",12,0,0,""
2586107687,7:36:00,7:36:00,"de:14625:7434:0:1",13,0,0,""
2586107687,7:38:00,7:38:00,"de:14625:7433:0:2",14,0,0,""
2586107687,7:41:00,7:41:00,"de:14625:7428:0:2",15,0,0,""
2586107687,7:43:00,7:43:00,"de:14625:7432:0:3",16,0,0,""
2586107687,7:46:00,7:46:00,"de:14625:7440:0:1",17,0,0,""
2586107687,7:47:00,7:47:00,"de:14625:7442:0:3",18,0,0,""
2586107687,7:48:00,7:48:00,"de:14625:7444:0:1",19,0,0,""
2586107687,7:51:00,7:51:00,"de:14625:7447:0:1",20,0,0,""
2586107687,7:54:00,7:54:00,"de:14625:7419:0:1",21,0,0,""
2586107687,7:56:00,7:56:00,"de:14625:7402:0:1",22,0,0,""
2586107687,7:57:00,7:57:00,"de:14625:7401:0:2",23,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14625:7401:0:2","","Wilthen Schulzentrum",,"51.100256000000","14.397021000000",0,,0,"",""
"de:14625:7433:0:2","","Naundorf (b Gaußig) Feuerwehr",,"51.125673000000","14.294137000000",0,,0,"",""
"de:14625:7419:0:1","","Irgersdorf",,"51.109331000000","14.386133000000",0,,0,"",""
"de:14625:7444:0:1","","Dretschen Dorf",,"51.122578000000","14.348152000000",0,,0,"",""
"de:14625:7442:0:3","","Dretschen Zur Postschänke",,"51.125025000000","14.341604000000",0,,0,"",""
"de:14625:7440:0:1","","Diehmen",,"51.125645000000","14.333294000000",0,,0,"",""
"de:14625:7432:0:3","","Gaußig Busplatz",,"51.136113000000","14.317996000000",0,,0,"",""
"de:14625:7433:0:1","","Naundorf (b Gaußig) Feuerwehr",,"51.125690000000","14.294047000000",0,,0,"",""
"de:14625:7428:0:2","","Gaußig Kirche","Gaußig Kirche","51.134484000000","14.314977000000",0,,0,"2","2"
"de:14625:7434:0:1","","Naundorf (b Gaußig) Wendeplatz",,"51.121230000000","14.291675000000",0,,0,"",""
"de:14625:7402:0:1","","Wilthen Mönchswalder Straße",,"51.102332000000","14.397560000000",0,,0,"",""
"de:14625:7447:0:1","","Arnsdorf  (b Gaußig)",,"51.115901000000","14.365508000000",0,,0,"",""
"de:14625:7430:0:1","","Cossern",,"51.134411000000","14.284516000000",0,,0,"",""
"de:14625:7178:0:2","","Medewitz Abzw Cossern",,"51.144286000000","14.281192000000",0,,0,"",""
"de:14625:7185:0:1","","Demitz-Thumitz Abzw Birkenrode",,"51.145266000000","14.247523000000",0,,0,"",""
"de:14625:7174:0:2","","Wölkau Dresdener Straße",,"51.150924000000","14.243903000000",0,,0,"",""
"de:14625:7180:0:2","","Birkenrode",,"51.143035000000","14.256407000000",0,,0,"",""
"de:14625:7186:0:1","","Demitz-Thumitz Schule","Demitz-Thumitz Schule","51.141750000000","14.244648000000",0,,0,"1","2"
"de:14625:7048:0:2","","Bischofswerda Kynitzsch",,"51.144151000000","14.215498000000",0,,0,"",""
"de:14625:7000:2:6","","Bischofswerda Bahnhof","Bus","51.125972000000","14.186482000000",0,,0,"6","2"
"de:14625:7049:0:2","","Bischofswerda Schliefermühle",,"51.146833000000","14.226727000000",0,,0,"",""
"de:14625:7035:0:2_G","","Bischofswerda Goldener Löwe",,"51.138328000000","14.195070000000",0,,0,"",""
"de:14625:7034:0:2_G","","Bischofswerda Bautzener Straße",,"51.135364000000","14.189869000000",0,,0,"",""
"de:14625:7011:0:1_G","","Bischofswerda Kulturhaus",,"51.131243000000","14.185315000000",0,,0,"",""
"de:14625:7011:0:1","","Bischofswerda Kulturhaus",,"51.131243000000","14.185315000000",0,,0,"",""
"de:14625:7034:0:2","","Bischofswerda Bautzener Straße",,"51.135375000000","14.189878000000",0,,0,"",""
"de:14625:7035:0:2","","Bischofswerda Goldener Löwe",,"51.138436000000","14.195259000000",0,,0,"",""

#calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
15145,1,1,1,1,1,0,0,20240729,20241214

#calendar_dates.txt
"service_id","date","exception_type"
15145,20240729,2
15145,20240805,2
15145,20240812,2
15145,20241007,2
15145,20241014,2
15145,20240730,2
15145,20240806,2
15145,20240813,2
15145,20241008,2
15145,20241015,2
15145,20240731,2
15145,20240807,2
15145,20240814,2
15145,20241009,2
15145,20241016,2
15145,20241120,2
15145,20240801,2
15145,20240808,2
15145,20241003,2
15145,20241010,2
15145,20241017,2
15145,20241031,2
15145,20240802,2
15145,20240809,2
15145,20241011,2
15145,20241018,2

)__");
}

constexpr auto const vdv_update_rbo739 = R"(
<IstFahrt Zst="2024-08-16T09:06:33">
  <LinienID>RBO739</LinienID>
  <RichtungsID>1</RichtungsID>
  <FahrtRef>
    <FahrtID>
      <FahrtBezeichner>RBO11670_vvorbl</FahrtBezeichner>
      <Betriebstag>2024-08-16</Betriebstag>
    </FahrtID>
  </FahrtRef>
  <Komplettfahrt>false</Komplettfahrt>
  <UmlaufID>17111</UmlaufID>
  <BetreiberID>vvorbl</BetreiberID>
  <IstHalt>
    <HaltID>de:14625:7000:2:6</HaltID>
    <Abfahrtszeit>2024-08-16T07:10:00</Abfahrtszeit>
    <IstAbfahrtPrognose>2024-08-16T07:10:00</IstAbfahrtPrognose>
    <AbfahrtssteigText>6</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14625:7011:0:1</HaltID>
    <Abfahrtszeit>2024-08-16T07:12:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-16T07:12:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-16T07:12:00</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-16T07:12:00</IstAnkunftPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14625:7034:0:2</HaltID>
    <Abfahrtszeit>2024-08-16T07:13:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-16T07:13:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-16T07:13:00</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-16T07:13:00</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14625:7035:0:2</HaltID>
    <Abfahrtszeit>2024-08-16T07:14:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-16T07:14:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-16T07:14:00</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-16T07:14:00</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14625:7050:0:1</HaltID>
    <Abfahrtszeit>2024-08-16T07:17:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-16T07:17:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-16T07:17:00</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-16T07:17:00</IstAnkunftPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14625:7147:0:1</HaltID>
    <Abfahrtszeit>2024-08-16T07:21:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-16T07:21:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-16T07:21:00</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-16T07:21:00</IstAnkunftPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14625:7143:0:1</HaltID>
    <Abfahrtszeit>2024-08-16T07:23:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-16T07:23:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-16T07:26:00</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-16T07:23:00</IstAnkunftPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14625:6530:1:1</HaltID>
    <Abfahrtszeit>2024-08-16T07:32:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-16T07:32:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-16T07:32:00</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-16T07:32:00</IstAnkunftPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14625:6400:1:1</HaltID>
    <Ankunftszeit>2024-08-16T07:35:00</Ankunftszeit>
    <IstAnkunftPrognose>2024-08-16T07:35:00</IstAnkunftPrognose>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <LinienText>739</LinienText>
  <ProduktID>RBO739</ProduktID>
  <RichtungsText>Panschwitz-Kuckau</RichtungsText>
  <PrognoseMoeglich>true</PrognoseMoeglich>
  <FaelltAus>false</FaelltAus>
  <Besetztgrad>Unbekannt</Besetztgrad>
</IstFahrt>
)";

mem_dir rvs261_files() {
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

constexpr auto const vdv_update_rvs261 = R"(
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

TEST(vdv_update, rbo739) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  load_timetable({}, source_idx_t{0}, rbo739_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 16});

  auto doc = pugi::xml_document{};
  doc.load_string(vdv_update_rbo739);
  auto const stats = rt::vdv::vdv_update(tt, rtt, source_idx_t{0}, doc);

  std::cout << "Stats: " << stats << "\n";

  auto fr = rt::frun(
      tt, &rtt,
      {{transport_idx_t{0}, day_idx_t{20}}, {stop_idx_t{0}, stop_idx_t{24}}});

  std::cout << "fr.is_rt():" << std::boolalpha << fr.is_rt() << "\n";

  for (auto rs : fr) {
    std::cout << "stop_idx: " << rs.stop_idx_ << ", stop_name: " << rs.name();
    if (rs.stop_idx_ > 0) {
      std::cout << "\nARR: " << rs.scheduled_time(event_type::kArr)
                << ", RT: " << rs.time(event_type::kArr);
    }
    if (rs.stop_idx_ < fr.size() - 1) {
      std::cout << "\nDEP: " << rs.scheduled_time(event_type::kDep)
                << ", RT: " << rs.time(event_type::kDep);
    }
    std::cout << "\n\n";
  }
}

TEST(vdv_update, rvs261) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  load_timetable({}, source_idx_t{0}, rvs261_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 19});

  auto doc = pugi::xml_document{};
  doc.load_string(vdv_update_rvs261);
  auto const stats = rt::vdv::vdv_update(tt, rtt, source_idx_t{0}, doc);

  std::cout << "Stats: " << stats << "\n";

  auto fr = rt::frun(
      tt, &rtt,
      {{transport_idx_t{0}, day_idx_t{23}}, {stop_idx_t{0}, stop_idx_t{19}}});

  std::cout << "fr.is_rt():" << std::boolalpha << fr.is_rt() << "\n";

  for (auto rs : fr) {
    std::cout << "stop_idx: " << rs.stop_idx_ << ", stop_name: " << rs.name();
    if (rs.stop_idx_ > 0) {
      std::cout << "\nARR: " << rs.scheduled_time(event_type::kArr)
                << ", RT: " << rs.time(event_type::kArr);
    }
    if (rs.stop_idx_ < fr.size() - 1) {
      std::cout << "\nDEP: " << rs.scheduled_time(event_type::kDep)
                << ", RT: " << rs.time(event_type::kDep);
    }
    std::cout << "\n\n";
  }
}