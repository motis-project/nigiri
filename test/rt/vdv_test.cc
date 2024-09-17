#include "gtest/gtest.h"

#include "pugixml.hpp"

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

TEST(vdv_update, delay_propagation) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / July / 1},
                    date::sys_days{2024_y / July / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, vdv_test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / July / 10});

  auto doc = pugi::xml_document{};
  doc.load_string(vdv_update_msg0);
  auto u = rt::vdv::updater{tt, src_idx};
  u.update(rtt, doc);

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
  u.update(rtt, doc);

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
  u.update(rtt, doc);

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

  EXPECT_EQ(u.get_stats().found_runs_, 1);
  EXPECT_EQ(u.get_stats().matched_runs_, 3);
}

namespace {
mem_dir before_midnight_files() {
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

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
AB,MTA,AB,AB,A -> B,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
AB,D,AB_TRIP,AB_TRIP,1


# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AB_TRIP,01:58,01:58,A,0,0,0
AB_TRIP,03:00,03:00,B,1,0,0

)__");
}

constexpr auto const after_midnight_update = R"(
<?xml version="1.0" encoding="iso-8859-1"?>
<DatenAbrufenAntwort>
  <Bestaetigung Zst="2024-07-10T00:00:00" Ergebnis="ok" Fehlernummer="0" />
  <AUSNachricht AboID="1">
    <IstFahrt Zst="2024-07-10T00:00:00">
      <LinienID>AB</LinienID>
      <RichtungsID>1</RichtungsID>
      <FahrtRef>
        <FahrtID>
          <FahrtBezeichner>AB</FahrtBezeichner>
          <Betriebstag>2024-07-10</Betriebstag>
        </FahrtID>
      </FahrtRef>
      <Komplettfahrt>false</Komplettfahrt>
      <BetreiberID>MTA</BetreiberID>
      <IstHalt>
        <HaltID>A</HaltID>
        <Abfahrtszeit>2024-07-10T00:02:00</Abfahrtszeit>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <IstHalt>
        <HaltID>B</HaltID>
        <Ankunftszeit>2024-07-10T01:00:00</Ankunftszeit>
        <IstAnkunftPrognose>2024-07-10T01:00:00</IstAnkunftPrognose>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <LinienText>AB</LinienText>
      <ProduktID>AB</ProduktID>
      <RichtungsText>B</RichtungsText>
      <Zusatzfahrt>false</Zusatzfahrt>
      <FaelltAus>false</FaelltAus>
    </IstFahrt>
  </AUSNachricht>
</DatenAbrufenAntwort>
)";

}  // namespace

TEST(vdv_update, tt_before_midnight_update_after_midnight) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / July / 1},
                    date::sys_days{2024_y / July / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, before_midnight_files(), tt);
  finalize(tt);

  std::cout << tt << "\n";

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / July / 9});

  auto doc = pugi::xml_document{};
  doc.load_string(after_midnight_update);
  auto u = rt::vdv::updater{tt, src_idx};
  u.update(rtt, doc);

  auto fr = rt::frun(
      tt, &rtt,
      {{transport_idx_t{0}, day_idx_t{13}}, {stop_idx_t{0}, stop_idx_t{2}}});

  EXPECT_TRUE(fr.is_rt());
}

namespace {
mem_dir after_midnight_files() {
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

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
AB,MTA,AB,AB,A -> B,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
AB,D,AB_TRIP,AB_TRIP,1


# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AB_TRIP,02:02,02:02,A,0,0,0
AB_TRIP,03:00,03:00,B,1,0,0

)__");
}

constexpr auto const before_midnight_update = R"(
<?xml version="1.0" encoding="iso-8859-1"?>
<DatenAbrufenAntwort>
  <Bestaetigung Zst="2024-07-10T00:00:00" Ergebnis="ok" Fehlernummer="0" />
  <AUSNachricht AboID="1">
    <IstFahrt Zst="2024-07-10T00:00:00">
      <LinienID>AB</LinienID>
      <RichtungsID>1</RichtungsID>
      <FahrtRef>
        <FahrtID>
          <FahrtBezeichner>AB</FahrtBezeichner>
          <Betriebstag>2024-07-10</Betriebstag>
        </FahrtID>
      </FahrtRef>
      <Komplettfahrt>false</Komplettfahrt>
      <BetreiberID>MTA</BetreiberID>
      <IstHalt>
        <HaltID>A</HaltID>
        <Abfahrtszeit>2024-07-09T23:58:00</Abfahrtszeit>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <IstHalt>
        <HaltID>B</HaltID>
        <Ankunftszeit>2024-07-10T01:00:00</Ankunftszeit>
        <IstAnkunftPrognose>2024-07-10T01:00:00</IstAnkunftPrognose>
        <Einsteigeverbot>false</Einsteigeverbot>
        <Aussteigeverbot>false</Aussteigeverbot>
        <Durchfahrt>false</Durchfahrt>
        <Zusatzhalt>false</Zusatzhalt>
      </IstHalt>
      <LinienText>AB</LinienText>
      <ProduktID>AB</ProduktID>
      <RichtungsText>B</RichtungsText>
      <Zusatzfahrt>false</Zusatzfahrt>
      <FaelltAus>false</FaelltAus>
    </IstFahrt>
  </AUSNachricht>
</DatenAbrufenAntwort>
)";

}  // namespace

TEST(vdv_update, tt_after_midnight_update_before_midnight) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / July / 1},
                    date::sys_days{2024_y / July / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, after_midnight_files(), tt);
  finalize(tt);

  std::cout << tt << "\n";

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / July / 10});

  auto doc = pugi::xml_document{};
  doc.load_string(before_midnight_update);
  auto u = rt::vdv::updater{tt, src_idx};
  u.update(rtt, doc);

  auto fr = rt::frun(
      tt, &rtt,
      {{transport_idx_t{0}, day_idx_t{14}}, {stop_idx_t{0}, stop_idx_t{2}}});

  EXPECT_TRUE(fr.is_rt());
}

namespace {

mem_dir rbo707_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:von:27-707_3",1180,2593399830,"Caßlau","",0,,89176,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:von:27-707_3",7874,"707","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
7874,"RBO-Bus","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593399830,12:15:00,12:15:00,"de:14625:7501:0:7_G",0,0,0,""
2593399830,12:19:00,12:19:00,"de:14625:7502:0:1_G",1,0,0,""
2593399830,12:21:00,12:21:00,"de:14625:7507:0:1_G",2,0,0,""
2593399830,12:23:00,12:23:00,"de:14625:7578:0:1_G",3,0,0,""
2593399830,12:25:00,12:25:00,"de:14625:7577:0:1_G",4,0,0,""
2593399830,12:28:00,12:28:00,"de:14625:7652:0:1_G",5,0,0,""
2593399830,12:31:00,12:31:00,"de:14625:7662:0:1",6,0,0,""
2593399830,12:33:00,12:33:00,"de:14625:7776:0:2",7,0,0,""
2593399830,12:37:00,12:37:00,"de:14625:7772:0:1_G",8,0,0,""
2593399830,12:41:00,12:41:00,"de:14625:7683:0:1",9,0,0,""
2593399830,12:42:00,12:42:00,"de:14625:7684:0:1",10,0,0,""
2593399830,12:46:00,12:46:00,"de:14625:7686:0:1_G",11,0,0,""
2593399830,12:48:00,12:48:00,"de:14625:7677:0:1_G",12,0,0,""
2593399830,12:50:00,12:50:00,"de:14625:7679:0:1_G",13,0,0,""
2593399830,12:56:00,12:56:00,"de:14625:7704:0:1",14,0,0,""
2593399830,12:58:00,12:58:00,"de:14625:7708:0:2",15,0,0,""
2593399830,13:00:00,13:00:00,"de:14625:7705:0:1_G",16,0,0,""
2593399830,13:02:00,13:02:00,"de:14625:7709:0:1_G",17,0,0,""
2593399830,13:04:00,13:04:00,"de:14625:7707:0:1_G",18,0,0,""
2593399830,13:09:00,13:09:00,"de:14625:7706:0:1",19,0,0,""
2593399830,13:11:00,13:11:00,"de:14625:7699:0:1",20,0,0,""
2593399830,13:12:00,13:12:00,"de:14625:7698:0:1",21,0,0,""
2593399830,13:14:00,13:14:00,"de:14625:7695:0:1_G",22,0,0,""
2593399830,13:18:00,13:18:00,"de:14625:7697:0:1",23,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14625:7695:0:1_G","","Doberschütz (b Neschwitz)",,"51.270937000000","14.267852000000",0,,0,"",""
"de:14625:7695:0:1","","Doberschütz (b Neschwitz)",,"51.270915000000","14.267852000000",0,,0,"",""
"de:14625:7708:0:2","","Wetro Werk",,"51.256384000000","14.302096000000",0,,0,"",""
"de:14625:7706:0:1","","Neu-Puschwitz",,"51.259161000000","14.294864000000",0,,0,"",""
"de:14625:7501:0:7_G","","Bautzen August-Bebel-Pl (ZOB)",,"51.177395000000","14.433501000000",0,,0,"",""
"de:14625:7501:0:7","","Bautzen August-Bebel-Pl (ZOB)",,"51.177395000000","14.433501000000",0,,0,"",""
"de:14625:7705:0:1_G","","Puschwitz",,"51.249384000000","14.290831000000",0,,0,"",""
"de:14625:7705:0:1","","Puschwitz",,"51.250402000000","14.292178000000",0,,0,"",""
"de:14625:7698:0:1","","Lissahora",,"51.269864000000","14.283932000000",0,,0,"",""
"de:14625:7704:0:1","","Wetro Dorf",,"51.250846000000","14.308519000000",0,,0,"",""
"de:14625:7677:0:1_G","","Neschwitz Dorfschänke",,"51.269335000000","14.329350000000",0,,0,"",""
"de:14625:7677:0:1","","Neschwitz Dorfschänke",,"51.269375000000","14.329324000000",0,,0,"",""
"de:14625:7707:0:1_G","","Jeßnitz (b Puschwitz) Dorfteich",,"51.249519000000","14.276952000000",0,,0,"",""
"de:14625:7707:0:1","","Jeßnitz (b Puschwitz) Dorfteich",,"51.249491000000","14.276925000000",0,,0,"",""
"de:14625:7683:0:1","","Luga (b Neschwitz) Dorfclub",,"51.249721000000","14.348592000000",0,,0,"",""
"de:14625:7652:0:1_G","","Kleinwelka Gasthof",,"51.213204000000","14.392987000000",0,,0,"",""
"de:14625:7652:0:1","","Kleinwelka Gasthof",,"51.213283000000","14.392942000000",0,,0,"",""
"de:14625:7776:0:2","","Radibor Schule",,"51.241956000000","14.395619000000",0,,0,"",""
"de:14625:7699:0:1","","Lomske (b Neschwitz)",,"51.267593000000","14.298987000000",0,,0,"",""
"de:14625:7686:0:1_G","","Holscha Teich",,"51.266587000000","14.343822000000",0,,0,"",""
"de:14625:7686:0:1","","Holscha Teich",,"51.266587000000","14.343715000000",0,,0,"",""
"de:14625:7662:0:1","","Cölln Dorfplatz",,"51.227000000000","14.390122000000",0,,0,"",""
"de:14625:7679:0:1_G","","Neschwitz Grundschule",,"51.271690000000","14.323206000000",0,,0,"",""
"de:14625:7679:0:1","","Neschwitz Grundschule",,"51.271741000000","14.323179000000",0,,0,"",""
"de:14625:7507:0:1_G","","Bautzen Fiedlerstraße",,"51.181100000000","14.415014000000",0,,0,"",""
"de:14625:7507:0:1","","Bautzen Fiedlerstraße",,"51.181241000000","14.414960000000",0,,0,"",""
"de:14625:7772:0:1_G","","Quoos Dorfplatz",,"51.249204000000","14.366792000000",0,,0,"",""
"de:14625:7772:0:1","","Quoos Dorfplatz",,"51.249165000000","14.366855000000",0,,0,"",""
"de:14625:7684:0:1","","Luga (b Neschwitz) Dorf",,"51.245009000000","14.346203000000",0,,0,"",""
"de:14625:7697:0:1","","Caßlau","Caßlau","51.281783000000","14.273502000000",0,,0,"1","2"
"de:14625:7578:0:1_G","","Bautzen Abzw Seidau",,"51.191781000000","14.412436000000",0,,0,"",""
"de:14625:7578:0:1","","Bautzen Abzw Seidau",,"51.191849000000","14.412400000000",0,,0,"",""
"de:14625:7502:0:1_G","","Bautzen Lauengraben",,"51.179602000000","14.424958000000",0,,0,"",""
"de:14625:7502:0:1","","Bautzen Lauengraben",,"51.179670000000","14.425210000000",0,,0,"",""
"de:14625:7577:0:1_G","","Bautzen Hoyerswerdaer Straße",,"51.196443000000","14.409480000000",0,,0,"",""
"de:14625:7577:0:1","","Bautzen Hoyerswerdaer Straße",,"51.196612000000","14.409462000000",0,,0,"",""
"de:14625:7709:0:1_G","","Guhra Dorfplatz",,"51.243755000000","14.286204000000",0,,0,"",""
"de:14625:7709:0:1","","Guhra Dorfplatz",,"51.243705000000","14.286231000000",0,,0,"",""
"de:14625:7699:0:2","","Lomske (b Neschwitz)",,"51.267615000000","14.299778000000",0,,0,"",""
"de:14625:7698:0:2","","Lissahora",,"51.269931000000","14.284084000000",0,,0,"",""
"de:14625:7697:0:2","","Caßlau","Caßlau","51.281721000000","14.273556000000",0,,0,"2","2"

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
1180,1,1,1,1,1,0,0,20240805,20241214

# calendar_dates.txt
"service_id","date","exception_type"
1180,20240805,2
1180,20240812,2
1180,20241007,2
1180,20241014,2
1180,20240806,2
1180,20240813,2
1180,20241008,2
1180,20241015,2
1180,20240807,2
1180,20241009,2
1180,20241016,2
1180,20241120,2
1180,20240808,2
1180,20241003,2
1180,20241010,2
1180,20241017,2
1180,20241031,2
1180,20240809,2
1180,20241011,2
1180,20241018,2

)__");
}

constexpr auto const update_rbo707 = R"(
<IstFahrt Zst="2024-08-23T13:12:24">
	<LinienID>RBO707</LinienID>
	<RichtungsID>1</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>RBO2732_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-23</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>true</Komplettfahrt>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14625:7501:0:7</HaltID>
		<Abfahrtszeit>2024-08-23T10:15:00</Abfahrtszeit>
		<AbfahrtssteigText>7</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7502:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:19:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:19:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7507:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:21:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:21:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7578:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:23:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:23:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7577:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:25:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:25:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7652:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:28:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:28:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7662:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:31:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:31:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7776:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T10:33:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:33:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Caßlau über Neschwitz</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7772:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:37:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:37:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau über Neschwitz</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7683:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:41:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:41:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau über Neschwitz</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7684:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:42:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:42:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau über Neschwitz</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7686:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:46:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:46:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau über Neschwitz</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7677:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:48:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:48:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7679:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:50:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:50:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7704:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T10:56:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:56:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7708:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T10:58:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T10:58:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7705:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T11:00:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:00:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7709:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T11:02:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:02:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7707:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T11:04:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:04:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7706:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T11:09:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:09:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7699:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:11:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:11:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7698:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:12:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:12:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7695:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T11:14:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:14:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7697:0:2</HaltID>
		<Ankunftszeit>2024-08-23T11:18:00</Ankunftszeit>
		<AnkunftssteigText>2</AnkunftssteigText>
		<RichtungsText>Caßlau</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>707</LinienText>
	<ProduktID>RBO707</ProduktID>
	<RichtungsText>Caßlau über Radibor - Neschwitz</RichtungsText>
	<PrognoseMoeglich>true</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, exact_match_1) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, rbo707_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 23});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_rbo707);
  u.update(rtt, doc);

  auto const fr = rt::frun{tt,
                           &rtt,
                           {{transport_idx_t{0U}, day_idx_t{27U}},
                            {stop_idx_t{0U}, stop_idx_t{24U}}}};

  EXPECT_TRUE(fr.is_rt());
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

TEST(vdv_update, exact_match_2) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, rbo920_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 20});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_rbo920);
  u.update(rtt, doc);

  auto fr = rt::frun{
      tt,
      &rtt,
      {{transport_idx_t{0}, day_idx_t{24}}, {stop_idx_t{0}, stop_idx_t{26}}}};
  EXPECT_TRUE(fr.is_rt());
}

namespace {

mem_dir rvs347_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:23-347_3",1171,2593427812,"Rabenau Markt","",1,85683,48284,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:23-347_3",8195,"347","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8195,"RVD-Busverkehr","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593427812,15:19:00,15:19:00,"de:14628:1030:0:4",0,0,0,""
2593427812,15:21:00,15:21:00,"de:14628:1039:0:2",1,0,0,""
2593427812,15:24:00,15:24:00,"de:14628:1276:2:91",2,0,0,""
2593427812,15:26:00,15:26:00,"de:14628:1275:2:91",3,0,0,""
2593427812,15:29:00,15:29:00,"de:14628:1293:0:1",4,0,0,""
2593427812,15:31:00,15:31:00,"de:14628:1292:0:1",5,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14628:1275:2:91","","Obernaundorf Gasthof","Marktsteig","50.967772000000","13.666151000000",0,,0,"91","2"
"de:14628:1292:0:1","","Rabenau Markt","Rabenau Markt","50.963077000000","13.641430000000",0,,0,"1","2"
"de:14628:1276:2:91","","Obernaundorf Wendeplatz","Marktsteig","50.971065000000","13.681144000000",0,,0,"91","2"
"de:14628:1293:0:1","","Rabenau Obernaundorfer Straße",,"50.965340000000","13.645050000000",0,,0,"",""
"de:14628:1039:0:2","","Possendorf Wilmsdorf",,"50.970686000000","13.702120000000",0,,0,"",""
"de:14628:1030:0:4","","Possendorf Hauptstr (Wpl)","Possendorf Hauptstr (Wpl)","50.968219000000","13.712828000000",0,,0,"4","2"

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

constexpr auto const update_rvs347 = R"(
<IstFahrt Zst="2024-08-26T15:25:23">
	<LinienID>RVS347</LinienID>
	<RichtungsID>2</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>RVS31512_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-26</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>true</Komplettfahrt>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14628:1030:0:4</HaltID>
		<Abfahrtszeit>2024-08-26T13:19:00</Abfahrtszeit>
		<AbfahrtssteigText>4</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14628:1039:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T13:21:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T13:21:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14628:1276:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T13:24:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T13:24:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14628:1275:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T13:26:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T13:26:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14628:1274:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T13:27:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T13:27:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14628:1293:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T13:29:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T13:29:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14628:1292:0:1</HaltID>
		<Ankunftszeit>2024-08-26T13:31:00</Ankunftszeit>
		<AnkunftssteigText>1</AnkunftssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>347</LinienText>
	<ProduktID>RVS347</ProduktID>
	<RichtungsText>Rabenau - 376 Dippoldiswalde</RichtungsText>
	<PrognoseMoeglich>true</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, match_despite_unresolvable_stops) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, rvs347_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 26});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_rvs347);
  u.update(rtt, doc);

  auto const fr = rt::frun{tt,
                           &rtt,
                           {{transport_idx_t{0U}, day_idx_t{30U}},
                            {stop_idx_t{0U}, stop_idx_t{6U}}}};

  EXPECT_TRUE(fr.is_rt());
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

TEST(vdv_update, match_despite_time_discrepancy_1) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, sv270_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 20});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_vgm270);
  u.update(rtt, doc);

  auto fr = rt::frun{
      tt,
      &rtt,
      {{transport_idx_t{0}, day_idx_t{24}}, {stop_idx_t{0}, stop_idx_t{39}}}};

  EXPECT_TRUE(fr.is_rt());
}

namespace {

mem_dir rbo501_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:von:27-501_3",1194,2593399129,"Mücka Bahnhof","",0,,89026,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:von:27-501_3",7874,"501","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
7874,"RBO-Bus","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593399129,10:04:00,10:04:00,"de:14625:7501:0:4",0,0,0,""
2593399129,10:05:00,10:05:00,"de:14625:7500:3:1",1,0,0,""
2593399129,10:08:00,10:08:00,"de:14625:7553:0:1",2,0,0,""
2593399129,10:09:00,10:09:00,"de:14625:7575:0:1",3,0,0,""
2593399129,10:11:00,10:11:00,"de:14625:7545:0:1",4,0,0,""
2593399129,10:13:00,10:13:00,"de:14625:7540:0:1",5,0,0,""
2593399129,10:16:00,10:16:00,"de:14625:7543:0:1",6,0,0,""
2593399129,10:19:00,10:19:00,"de:14625:7853:0:1",7,0,0,""
2593399129,10:21:00,10:21:00,"de:14625:7855:0:1",8,0,0,""
2593399129,10:24:00,10:24:00,"de:14625:7857:0:1",9,0,0,""
2593399129,10:26:00,10:26:00,"de:14625:7831:0:1",10,0,0,""
2593399129,10:27:00,10:27:00,"de:14625:7832:0:1",11,0,0,""
2593399129,10:29:00,10:29:00,"de:14625:7833:0:1",12,0,0,""
2593399129,10:30:00,10:30:00,"de:14625:7834:0:1",13,0,0,""
2593399129,10:32:00,10:32:00,"de:14625:7839:0:1",14,0,0,""
2593399129,10:36:00,10:36:00,"de:14626:8595:0:1",15,0,0,""
2593399129,10:37:00,10:37:00,"de:14626:8592:0:1",16,0,0,""
2593399129,10:39:00,10:39:00,"de:14626:8591:0:1",17,0,0,""
2593399129,10:41:00,10:41:00,"de:14626:8461:0:1",18,0,0,""
2593399129,10:44:00,10:44:00,"de:14626:8463:0:1",19,0,0,""
2593399129,10:47:00,10:47:00,"de:14626:8466:0:1",20,0,0,""
2593399129,10:49:00,10:49:00,"de:14626:8467:0:1",21,0,0,""
2593399129,10:51:00,10:51:00,"de:14626:8468:5:2",22,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14626:8468:5:2","","Mücka Bahnhof","Bus","51.319683000000","14.704622000000",0,,0,"2","2"
"de:14626:8467:0:1","","Mücka Schule","Mücka Schule","51.314995000000","14.703831000000",0,,0,"1","2"
"de:14626:8463:0:1","","Förstgen Gasthaus",,"51.296895000000","14.663739000000",0,,0,"",""
"de:14626:8591:0:1","","Weigersdorf Niederdorf",,"51.268846000000","14.645746000000",0,,0,"",""
"de:14626:8592:0:1","","Weigersdorf Landambulatorium",,"51.273505000000","14.640778000000",0,,0,"",""
"de:14626:8466:0:1","","Mücka Bildungszentrum",,"51.311722000000","14.688434000000",0,,0,"",""
"de:14626:8595:0:1","","Dauban",,"51.276843000000","14.634823000000",0,,0,"",""
"de:14625:7834:0:1","","Guttau Anbau",,"51.257840000000","14.572713000000",0,,0,"",""
"de:14625:7832:0:1","","Abzweig nach Brösa",,"51.256519000000","14.553399000000",0,,0,"",""
"de:14625:7857:0:1","","Malschwitz Schule",,"51.237518000000","14.521850000000",0,,0,"",""
"de:14625:7553:0:1","","Bautzen Ziegelwall",,"51.182615000000","14.435639000000",0,,0,"",""
"de:14625:7575:0:1","","Bautzen List-/Muskauer Straße",,"51.185301000000","14.438163000000",0,,0,"",""
"de:14625:7831:0:1","","Guttau Gewerbegebiet",,"51.249328000000","14.543302000000",0,,0,"",""
"de:14625:7540:0:1","","Bautzen Burker Straße",,"51.197833000000","14.463235000000",0,,0,"",""
"de:14625:7855:0:1","","Pließkowitz Malschwitzer Landstraße",,"51.227101000000","14.506175000000",0,,0,"",""
"de:14625:7839:0:1","","Kleinsaubernitz",,"51.264130000000","14.598809000000",0,,0,"",""
"de:14625:7833:0:1","","Guttau Hauptstraße",,"51.258464000000","14.561179000000",0,,0,"",""
"de:14625:7545:0:1","","Bautzen Gesundbrunnenring",,"51.191134000000","14.451926000000",0,,0,"",""
"de:14625:7853:0:1","","Doberschütz (b Malschwitz) Am Schafberg",,"51.219630000000","14.497982000000",0,,0,"",""
"de:14625:7543:0:1","","Bautzen Burk Talsperre",,"51.208989000000","14.471500000000",0,,0,"",""
"de:14625:7500:3:1","","Bautzen Bahnhof","Bus","51.173723000000","14.429764000000",0,,0,"1","2"
"de:14626:8461:0:1","","Abzweig nach Leipgen",,"51.282923000000","14.656283000000",0,,0,"",""
"de:14625:7501:0:4","","Bautzen August-Bebel-Pl (ZOB)",,"51.177423000000","14.433843000000",0,,0,"",""

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

constexpr auto const update_rbo501 = R"(
<IstFahrt Zst="2024-08-26T10:37:40">
	<LinienID>RBO501</LinienID>
	<RichtungsID>1</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>RBO2812_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-26</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>true</Komplettfahrt>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14625:7501:0:4</HaltID>
		<Abfahrtszeit>2024-08-26T08:04:00</Abfahrtszeit>
		<IstAbfahrtPrognose>2024-08-26T08:04:00</IstAbfahrtPrognose>
		<AbfahrtssteigText>4</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7500:3:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:05:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:05:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:05:40</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:05:40</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7553:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:08:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:08:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:10:16</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:08:00</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7596:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:10:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:10:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:12:14</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:12:14</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7545:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:14:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:14:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:16:05</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:16:05</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7540:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:16:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:16:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:18:05</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:18:05</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7543:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:19:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:19:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:20:31</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:20:31</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7853:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:22:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:22:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:23:11</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:23:11</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7855:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:24:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:24:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:24:31</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:24:31</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7857:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:27:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:27:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:26:47</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:27:00</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7831:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:29:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:29:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:29:26</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:29:26</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7832:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:30:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:30:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:30:26</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:30:26</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7833:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:32:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:32:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:32:02</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:32:02</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7834:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:33:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:33:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:33:39</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:33:39</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7839:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:35:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:35:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:36:23</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:35:00</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Mücka Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8595:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:39:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:39:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:40:34</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:40:34</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Mücka Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8592:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:40:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:40:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:41:22</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:41:22</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Mücka Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8591:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:42:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:42:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:43:34</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:43:34</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Mücka Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8461:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:44:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:44:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:45:34</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:45:34</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Mücka Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8463:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:47:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:47:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:47:31</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:47:31</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Mücka Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8466:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:50:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:50:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:50:31</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:50:31</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Mücka Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8467:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:52:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:52:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:52:31</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:52:31</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Mücka Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8468:5:2</HaltID>
		<Ankunftszeit>2024-08-26T08:54:00</Ankunftszeit>
		<IstAnkunftPrognose>2024-08-26T08:54:31</IstAnkunftPrognose>
		<AnkunftssteigText>2</AnkunftssteigText>
		<RichtungsText>Mücka Bahnhof</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>501</LinienText>
	<ProduktID>RBO501</ProduktID>
	<RichtungsText>Mücka über Kleinsaubernitz</RichtungsText>
	<PrognoseMoeglich>true</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

// Requires the matching to tolerate time discrepancies of up to 3 minutes
TEST(vdv_update, match_despite_time_discrepancy_2) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, rbo501_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 26});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_rbo501);
  u.update(rtt, doc);

  auto const fr = rt::frun{tt,
                           &rtt,
                           {{transport_idx_t{0U}, day_idx_t{30U}},
                            {stop_idx_t{0U}, stop_idx_t{23U}}}};

  EXPECT_TRUE(fr.is_rt());
}

namespace {

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

// The reference time of line 712 in the update is from the year-long
// timetable. However, the GTFS timetable contains different times due to the
// line being redirected starting from June 29
// the time differences is up to 4 minutes
TEST(vdv_update, match_despite_unknown_route_diversion_1) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, smd712_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 20});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_smd712);
  u.update(rtt, doc);

  auto fr = rt::frun{
      tt,
      &rtt,
      {{transport_idx_t{0}, day_idx_t{24}}, {stop_idx_t{0}, stop_idx_t{20}}}};

  EXPECT_TRUE(fr.is_rt());
}

namespace {

mem_dir vgm456_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:24-456_3",1180,2593431074,"Radeburg Grundschule","",0,,49243,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:24-456_3",8196,"456","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8196,"VGM-Busverkehr","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593431074,10:36:00,10:36:00,"de:14627:5021:3:4",0,0,0,""
2593431074,10:38:00,10:38:00,"de:14627:5001:0:4",1,0,0,""
2593431074,10:40:00,10:40:00,"de:14627:5029:1:1",2,0,0,""
2593431074,10:42:00,10:42:00,"de:14627:5030:0:1",3,0,0,""
2593431074,10:43:00,10:43:00,"de:14627:5047:0:1",4,0,0,""
2593431074,10:44:00,10:44:00,"de:14627:5031:0:1",5,0,0,""
2593431074,10:45:00,10:45:00,"de:14627:5032:0:1",6,0,0,""
2593431074,10:47:00,10:47:00,"de:14627:5200:0:1",7,0,0,""
2593431074,10:51:00,10:51:00,"de:14627:5430:0:1",8,0,0,""
2593431074,10:52:00,10:52:00,"de:14627:5431:0:1",9,0,0,""
2593431074,10:54:00,10:54:00,"de:14627:5445:0:1",10,0,0,""
2593431074,10:56:00,10:56:00,"de:14627:5441:0:1",11,0,0,""
2593431074,10:58:00,10:58:00,"de:14627:5440:3:1",12,0,0,""
2593431074,11:02:00,11:02:00,"de:14627:5451:0:1",13,0,0,""
2593431074,11:04:00,11:04:00,"de:14627:5450:0:1",14,0,0,""
2593431074,11:07:00,11:07:00,"de:14627:5481:0:1",15,0,0,""
2593431074,11:09:00,11:09:00,"de:14627:5480:0:1",16,0,0,""
2593431074,11:10:00,11:10:00,"de:14627:5486:0:1",17,0,0,""
2593431074,11:11:00,11:11:00,"de:14627:5485:0:1",18,0,0,""
2593431074,11:15:00,11:15:00,"de:14627:5497:0:1",19,0,0,""
2593431074,11:19:00,11:19:00,"de:14627:5492:0:1",20,0,0,""
2593431074,11:20:00,11:20:00,"de:14627:5491:0:1",21,0,0,""
2593431074,11:22:00,11:22:00,"de:14627:5490:0:1",22,0,0,""
2593431074,11:24:00,11:24:00,"de:14627:5475:0:1",23,0,0,""
2593431074,11:26:00,11:26:00,"de:14627:5476:0:1",24,0,0,""
2593431074,11:28:00,11:28:00,"de:14627:5477:0:1",25,0,0,""
2593431074,11:31:00,11:31:00,"de:14627:5474:0:1",26,0,0,""
2593431074,11:34:00,11:34:00,"de:14627:5470:0:1",27,0,0,""
2593431074,11:37:00,11:37:00,"de:14627:5473:0:1",28,0,0,""
2593431074,11:38:00,11:38:00,"de:14627:5465:0:1",29,0,0,""
2593431074,11:40:00,11:40:00,"de:14627:5466:0:1",30,0,0,""
2593431074,11:42:00,11:42:00,"de:14627:5460:0:1",31,0,0,""
2593431074,11:44:00,11:44:00,"de:14627:5461:0:1",32,0,0,""
2593431074,11:48:00,11:48:00,"de:14627:5252:0:1",33,0,0,""
2593431074,11:52:00,11:52:00,"de:14627:4360:0:2",34,0,0,""
2593431074,11:54:00,11:54:00,"de:14627:4367:0:1",35,0,0,""
2593431074,11:55:00,11:55:00,"de:14627:4378:0:1",36,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14627:4367:0:1","","Radeburg Meißner Berg",,"51.213615000000","13.715478000000",0,,0,"",""
"de:14627:5252:0:1","","Rödern Königsbrücker Straße",,"51.232316000000","13.729465000000",0,,0,"",""
"de:14627:5466:0:1","","Dobra","Haltestelle","51.258914000000","13.773725000000",0,,0,"1","2"
"de:14627:5474:0:1","","Kleinnaundorf Zum Springbach",,"51.245234000000","13.789688000000",0,,0,"",""
"de:14627:5477:0:1","","Tauscha Anbau",,"51.257632000000","13.803531000000",0,,0,"",""
"de:14627:5476:0:1","","Tauscha Dorf",,"51.265418000000","13.804240000000",0,,0,"",""
"de:14627:5475:0:1","","Tauscha Abzw Dobra",,"51.268059000000","13.791933000000",0,,0,"",""
"de:14627:5473:0:1","","Kleinnaundorf Am Eichberg",,"51.244728000000","13.785169000000",0,,0,"",""
"de:14627:5491:0:1","","Sacka Großenhainer Straße",,"51.283985000000","13.792553000000",0,,0,"",""
"de:14627:5497:0:1","","Stölpchen",,"51.309122000000","13.784235000000",0,,0,"",""
"de:14627:5481:0:1","","Thiendorf Autobahn","Haltestelle","51.294822000000","13.729707000000",0,,0,"1","2"
"de:14627:5440:3:1","","Lampertswalde Bahnhof","Bus","51.309228000000","13.678279000000",0,,0,"1","2"
"de:14627:5465:0:1","","Kleinnaundorf Feldmühle",,"51.249058000000","13.780498000000",0,,0,"",""
"de:14627:5490:0:1","","Sacka Kirche",,"51.277197000000","13.792275000000",0,,0,"",""
"de:14627:5450:0:1","","Schönfeld Schloss",,"51.303612000000","13.708246000000",0,,0,"",""
"de:14627:5451:0:1","","Schönfeld Straße der MTS",,"51.301838000000","13.699533000000",0,,0,"",""
"de:14627:5441:0:1","","Lampertswalde Großenhainer Str","Haltestelle","51.315203000000","13.673194000000",0,,0,"1","2"
"de:14627:5021:3:4","","Großenhain Cottbuser Bahnhof","Busbahnhof","51.291177000000","13.524253000000",0,,0,"4","2"
"de:14627:4378:0:1","","Radeburg Grundschule",,"51.214549000000","13.716637000000",0,,0,"",""
"de:14627:5486:0:1","","Welxande Neubau",,"51.297035000000","13.747835000000",0,,0,"",""
"de:14627:5445:0:1","","Lampertswalde Gewerbegebiet",,"51.305719000000","13.670858000000",0,,0,"",""
"de:14627:5480:0:1","","Thiendorf Welxander Straße","Thiendorf Welxander Straße","51.293620000000","13.740900000000",0,,0,"1","2"
"de:14627:5430:0:1","","Quersa Abzw Brockwitz",,"51.302944000000","13.636094000000",0,,0,"",""
"de:14627:5031:0:1","","Großenhain Bornweg",,"51.295631000000","13.554778000000",0,,0,"",""
"de:14627:5492:0:1","","Sacka Zum Oberdorf",,"51.289424000000","13.791601000000",0,,0,"",""
"de:14627:5200:0:1","","Folbern",,"51.299776000000","13.588276000000",0,,0,"",""
"de:14627:5461:0:1","","Zschorna Zur Teichwirtschaft",,"51.254371000000","13.739957000000",0,,0,"",""
"de:14627:5460:0:1","","Zschorna Freibad",,"51.252983000000","13.752731000000",0,,0,"",""
"de:14627:5030:0:1","","Großenhain Radeburger Platz","Haltestelle","51.294294000000","13.535554000000",0,,0,"1","2"
"de:14627:5431:0:1","","Quersa Mühlbacher Weg",,"51.304000000000","13.645535000000",0,,0,"",""
"de:14627:5029:1:1","","Großenhain Fr-Schubert-Allee","Grundschule","51.293895000000","13.530982000000",0,,0,"1","2"
"de:14627:5001:0:4","","Großenhain Mozartallee",,"51.291924000000","13.527721000000",0,,0,"",""
"de:14627:5485:0:1","","Welxande Stölpchener Straße","Haltestelle","51.298805000000","13.755004000000",0,,0,"1","2"
"de:14627:5470:0:1","","Würschnitz",,"51.235522000000","13.793829000000",0,,0,"",""
"de:14627:5047:0:1","","Großenhain An der Turnhalle","Standard","51.295131000000","13.545040000000",0,,0,"1","2"
"de:14627:4360:0:2","","Radeburg Busbahnhof","Radeburg Busbahnhof","51.215156000000","13.720841000000",0,,0,"2","2"
"de:14627:5032:0:1","","Großenhain Friedrich-Ebert-Str",,"51.296142000000","13.560159000000",0,,0,"",""

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
1180,1,1,1,1,1,0,0,20240805,20241214

# calendar_dates.txt
"service_id","date","exception_type"
1180,20240805,2
1180,20240812,2
1180,20241007,2
1180,20241014,2
1180,20240806,2
1180,20240813,2
1180,20241008,2
1180,20241015,2
1180,20240807,2
1180,20241009,2
1180,20241016,2
1180,20241120,2
1180,20240808,2
1180,20241003,2
1180,20241010,2
1180,20241017,2
1180,20241031,2
1180,20240809,2
1180,20241011,2
1180,20241018,2
)__");
}

constexpr auto const update_vgm456 = R"(
<IstFahrt Zst="2024-08-26T11:25:27">
	<LinienID>VGM456</LinienID>
	<RichtungsID>1</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>VGM45601301_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-26</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>true</Komplettfahrt>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14627:5021:3:4</HaltID>
		<Abfahrtszeit>2024-08-26T08:36:00</Abfahrtszeit>
		<IstAbfahrtPrognose>2024-08-26T08:36:00</IstAbfahrtPrognose>
		<AbfahrtssteigText>4</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5001:0:4</HaltID>
		<Abfahrtszeit>2024-08-26T08:38:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:38:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:40:25</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:40:25</IstAnkunftPrognose>
		<AbfahrtssteigText>4</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5029:1:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:40:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:40:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:42:25</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:42:25</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5030:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:42:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:42:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:43:06</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:43:06</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5047:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:43:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:43:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:44:06</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:44:06</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5031:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:44:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:44:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:45:40</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:45:40</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5032:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:45:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:45:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:46:40</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:46:40</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5200:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:47:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:47:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:48:09</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:48:09</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5430:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:51:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:51:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:52:03</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:52:03</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5431:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:52:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:52:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:52:46</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:52:46</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5445:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:54:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:54:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:54:20</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:54:20</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5441:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:56:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:56:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:56:55</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:56:55</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5440:3:1</HaltID>
		<Abfahrtszeit>2024-08-26T08:58:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T08:58:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T08:58:18</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T08:58:18</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5451:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:02:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:02:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:01:21</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:01:21</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5450:0:97</HaltID>
		<Abfahrtszeit>2024-08-26T09:04:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:04:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:01:59</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:04:00</IstAnkunftPrognose>
		<AbfahrtssteigText>97</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5482:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T09:14:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:14:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:08:56</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:08:56</IstAnkunftPrognose>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5480:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:16:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:16:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:09:50</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:09:50</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5486:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:17:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:17:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:10:49</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:10:49</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5485:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:18:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:18:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:11:49</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:11:49</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5497:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:22:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:22:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:21:45</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:21:45</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5492:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:26:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:26:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:25:45</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:25:45</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5491:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:27:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:27:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:25:49</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:25:49</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5490:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:29:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:29:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:27:49</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:27:49</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5475:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:31:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:31:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:29:49</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:29:49</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5476:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:33:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:33:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:32:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:32:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5477:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:35:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:35:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:34:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:34:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5474:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:38:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:38:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:37:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:37:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5470:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:41:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:41:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:40:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:40:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5473:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:44:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:44:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:43:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:43:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5465:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:45:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:45:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:44:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:44:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5466:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:47:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:47:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:46:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:46:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5460:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:49:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:49:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:48:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:48:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5461:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:51:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:51:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:50:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:50:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:5252:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T09:55:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:55:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:54:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:54:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4360:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T09:59:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T09:59:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T09:58:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T09:58:27</IstAnkunftPrognose>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4367:0:1</HaltID>
		<Abfahrtszeit>2024-08-26T10:01:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T10:01:00</Ankunftszeit>
		<IstAbfahrtPrognose>2024-08-26T10:00:27</IstAbfahrtPrognose>
		<IstAnkunftPrognose>2024-08-26T10:00:27</IstAnkunftPrognose>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14627:4378:0:1</HaltID>
		<Ankunftszeit>2024-08-26T10:02:00</Ankunftszeit>
		<IstAnkunftPrognose>2024-08-26T10:01:27</IstAnkunftPrognose>
		<AnkunftssteigText>1</AnkunftssteigText>
		<RichtungsText>Radeburg Grundschule</RichtungsText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>456</LinienText>
	<ProduktID>VGM456</ProduktID>
	<RichtungsText>Radeburg ü. Sacka</RichtungsText>
	<PrognoseMoeglich>true</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, match_despite_unknown_route_diversion_2) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, vgm456_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 26});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_vgm456);
  u.update(rtt, doc);

  auto const fr = rt::frun{tt,
                           &rtt,
                           {{transport_idx_t{0U}, day_idx_t{30U}},
                            {stop_idx_t{0U}, stop_idx_t{27U}}}};

  EXPECT_TRUE(fr.is_rt());
}

namespace {

mem_dir ovo65_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:von:26-65_3",1180,2593401825,"Melaune Ort","",0,84112,89747,0,0
"de:von:26-65_3",1194,2593401832,"Weißenberg (Kr BZ) Markt","",0,,89745,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:von:26-65_3",14177,"65","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
14177,"Omnibus Verkehr Oberlausitz","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593401825,13:18:00,13:18:00,"de:14626:8691:0:2",0,0,0,""
2593401825,13:23:00,13:23:00,"de:14626:8730:0:2",1,0,0,""
2593401825,13:25:00,13:25:00,"de:14626:8731:0:2",2,0,0,""
2593401825,13:26:00,13:26:00,"de:14626:8732:0:2",3,0,0,""
2593401825,13:27:00,13:27:00,"de:14626:8734:0:2",4,0,0,""
2593401825,13:28:00,13:28:00,"de:14626:8736:0:2",5,0,0,""
2593401825,13:29:00,13:29:00,"de:14626:8737:0:1",6,0,0,""
2593401825,13:33:00,13:33:00,"de:14626:8740:0:2",7,0,0,""
2593401825,13:35:00,13:35:00,"de:14626:8741:0:2",8,0,0,""
2593401825,13:37:00,13:37:00,"de:14626:8745:0:1",9,0,0,""
2593401825,13:39:00,13:39:00,"de:14626:8746:0:1",10,0,0,""
2593401825,13:42:00,13:42:00,"de:14626:8747:0:3",11,0,0,""
2593401825,13:43:00,13:43:00,"de:14626:8761:0:1",12,0,0,""
2593401825,13:44:00,13:44:00,"de:14626:8753:0:2",13,0,0,""
2593401825,13:45:00,13:45:00,"de:14626:8751:0:2",14,0,0,""
2593401832,13:05:00,13:05:00,"de:14626:8880:0:3",0,0,0,""
2593401832,13:07:00,13:07:00,"de:14626:8877:0:2",1,0,0,""
2593401832,13:09:00,13:09:00,"de:14626:8898:0:2",2,0,0,""
2593401832,13:10:00,13:10:00,"de:14626:8881:1:1",3,0,0,""
2593401832,13:11:00,13:11:00,"de:14626:8890:3:1",4,0,0,""
2593401832,13:12:00,13:12:00,"de:14626:8891:1:1",5,0,0,""
2593401832,13:13:00,13:13:00,"de:14626:8892:1:1",6,0,0,""
2593401832,13:14:00,13:14:00,"de:14626:8930:0:1",7,0,0,""
2593401832,13:15:00,13:15:00,"de:14626:8914:0:1",8,0,0,""
2593401832,13:16:00,13:16:00,"de:14626:8934:0:2",9,0,0,""
2593401832,13:18:00,13:18:00,"de:14626:8691:0:2",10,0,0,""
2593401832,13:23:00,13:23:00,"de:14626:8730:0:2",11,0,0,""
2593401832,13:25:00,13:25:00,"de:14626:8731:0:2",12,0,0,""
2593401832,13:26:00,13:26:00,"de:14626:8732:0:2",13,0,0,""
2593401832,13:27:00,13:27:00,"de:14626:8734:0:2",14,0,0,""
2593401832,13:28:00,13:28:00,"de:14626:8736:0:2",15,0,0,""
2593401832,13:29:00,13:29:00,"de:14626:8737:0:1",16,0,0,""
2593401832,13:33:00,13:33:00,"de:14626:8740:0:2",17,0,0,""
2593401832,13:35:00,13:35:00,"de:14626:8741:0:2",18,0,0,""
2593401832,13:37:00,13:37:00,"de:14626:8745:0:1",19,0,0,""
2593401832,13:39:00,13:39:00,"de:14626:8746:0:1",20,0,0,""
2593401832,13:42:00,13:42:00,"de:14626:8747:0:3",21,0,0,""
2593401832,13:43:00,13:43:00,"de:14626:8761:0:1",22,0,0,""
2593401832,13:44:00,13:44:00,"de:14626:8753:0:2",23,0,0,""
2593401832,13:45:00,13:45:00,"de:14626:8751:0:2",24,0,0,""
2593401832,13:49:00,13:49:00,"de:14626:8754:0:2",25,0,0,""
2593401832,13:50:00,13:50:00,"de:14626:8755:0:2",26,0,0,""
2593401832,13:52:00,13:52:00,"de:14626:8756:0:2",27,0,0,""
2593401832,13:53:00,13:53:00,"de:14626:8739:0:2",28,0,0,""
2593401832,13:58:00,13:58:00,"de:14625:7904:0:2",29,0,0,""
2593401832,14:00:00,14:00:00,"de:14625:7903:0:2",30,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14626:8761:0:1","","Döbschütz Abzw Melaune",,"51.195154000000","14.755844000000",0,,0,"",""
"de:14626:8747:0:3","","Arnsdorf (OL) Holzmühle","Arnsdorf (OL) Holzmühle","51.201638000000","14.760290000000",0,,0,"3","2"
"de:14626:8741:0:2","","Hilbersdorf (OL) Kreuzung",,"51.188854000000","14.789917000000",0,,0,"",""
"de:14626:8753:0:2","","Melaune Hort","Melaune Hort","51.192643000000","14.752996000000",0,,0,"2","2"
"de:14626:8746:0:1","","Arnsdorf Abzw Thiemendorf",,"51.198931000000","14.774708000000",0,,0,"",""
"de:14626:8740:0:2","","Hilbersdorf (OL) Schäferei",,"51.185751000000","14.805251000000",0,,0,"",""
"de:14626:8751:0:2","","Melaune Ort","Melaune Ort","51.189997000000","14.747938000000",0,,0,"2","2"
"de:14626:8737:0:1","","Königshain Oberdorf",,"51.185802000000","14.849691000000",0,,0,"",""
"de:14626:8734:0:2","","Königshain Gemeindeverwaltung",,"51.181123000000","14.866884000000",0,,0,"",""
"de:14626:8745:0:1","","Arnsdorf (OL) Trodler","Arnsdorf (OL) Trodler","51.195593000000","14.785434000000",0,,0,"1","2"
"de:14626:8736:0:2","","Königshain Hennig",,"51.182390000000","14.857174000000",0,,0,"",""
"de:14626:8691:0:2","","Girbigsdorf Sandschänke","Girbigsdorf Sandschänke","51.166474000000","14.934060000000",0,,0,"2","2"
"de:14626:8732:0:2","","Königshain Teich",,"51.181821000000","14.875131000000",0,,0,"",""
"de:14626:8731:0:2","","Königshain Gut",,"51.179614000000","14.880126000000",0,,0,"",""
"de:14626:8730:0:2","","Königshain Ortseingang",,"51.177992000000","14.892558000000",0,,0,"",""
"de:14626:8739:0:2","","Buchholz b Görlitz, Eichberg",,"51.198959000000","14.696492000000",0,,0,"",""
"de:14626:8756:0:2","","Buchholz Ortseingang",,"51.196054000000","14.702124000000",0,,0,"",""
"de:14626:8755:0:2","","Tetta Gasthaus",,"51.192147000000","14.714279000000",0,,0,"",""
"de:14625:7903:0:2","","Weißenberg (Kr BZ) Markt","Weißenberg (Kr BZ) Markt","51.196837000000","14.658574000000",0,,0,"2","2"
"de:14626:8754:0:2","","Tetta Ortseingang",,"51.193172000000","14.721223000000",0,,0,"",""
"de:14626:8881:1:1","","Görlitz Jägerkaserne","Bus Steige 1+2","51.155640000000","14.984896000000",0,,0,"1","2"
"de:14626:8930:0:1","","Görlitz Klinikum",,"51.162288000000","14.971403000000",0,,0,"",""
"de:14626:8914:0:1","","Görlitz Stadion Junge Welt",,"51.163961000000","14.961405000000",0,,0,"",""
"de:14626:8934:0:2","","Görlitz Flugplatz",,"51.163978000000","14.945810000000",0,,0,"",""
"de:14626:8890:3:1","","Görlitz Heiliges Grab","Bus","51.158502000000","14.983495000000",0,,0,"1","2"
"de:14626:8880:0:3","","Görlitz Busbahnhof","Görlitz Busbahnhof","51.148253000000","14.976901000000",0,,0,"3","2"
"de:14626:8898:0:2","","Görlitz Theater","Görlitz Theater","51.153595000000","14.985210000000",0,,0,"2","2"
"de:14626:8892:1:1","","Görlitz Zeppelinstraße","Bus","51.161218000000","14.974880000000",0,,0,"1","2"
"de:14626:8877:0:2","","Görlitz Lutherplatz",,"51.151352000000","14.979309000000",0,,0,"",""
"de:14625:7904:0:2","","Weißenberg Schützenhaus","Weißenberg Schützenhaus","51.196026000000","14.664242000000",0,,0,"2","2"
"de:14626:8891:1:1","","Görlitz Kummerau","Bus","51.159443000000","14.978904000000",0,,0,"1","2"
"de:14626:8691:0:1","","Girbigsdorf Sandschänke","Girbigsdorf Sandschänke","51.166530000000","14.933971000000",0,,0,"1","2"
"de:14626:8737:0:2","","Königshain Oberdorf",,"51.185644000000","14.850086000000",0,,0,"",""
"de:14626:8761:0:2","","Döbschütz Abzw Melaune",,"51.195103000000","14.756068000000",0,,0,"",""
"de:14626:8751:0:1","","Melaune Ort","Melaune Ort","51.190374000000","14.748567000000",0,,0,"1","2"

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
1180,1,1,1,1,1,0,0,20240805,20241214
1194,1,1,1,1,1,1,0,20240805,20241214

# calendar_dates.txt
"service_id","date","exception_type"
1180,20240805,2
1180,20240812,2
1180,20241007,2
1180,20241014,2
1180,20240806,2
1180,20240813,2
1180,20241008,2
1180,20241015,2
1180,20240807,2
1180,20241009,2
1180,20241016,2
1180,20241120,2
1180,20240808,2
1180,20241003,2
1180,20241010,2
1180,20241017,2
1180,20241031,2
1180,20240809,2
1180,20241011,2
1180,20241018,2
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

constexpr auto const update_ovo65 = R"(
<IstFahrt Zst="2024-08-23T10:37:13">
	<LinienID>OVO65</LinienID>
	<RichtungsID>1</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>OVO16065030_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-23</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>true</Komplettfahrt>
	<UmlaufID>1808</UmlaufID>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14626:8691:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T11:18:00</Abfahrtszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8730:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:23:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:23:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8731:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:25:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:25:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8732:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:26:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:26:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8734:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:27:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:27:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8736:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:28:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:28:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8737:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:29:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:29:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8740:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:33:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:33:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8741:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:35:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:35:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8745:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T11:37:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:37:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8746:0:1</HaltID>
		<Abfahrtszeit>2024-08-23T11:39:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:39:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8747:0:3</HaltID>
		<Abfahrtszeit>2024-08-23T11:42:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:42:00</Ankunftszeit>
		<AbfahrtssteigText>3</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8761:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:43:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:43:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8753:0:2</HaltID>
		<Abfahrtszeit>2024-08-23T11:44:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-23T11:44:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14626:8751:0:1</HaltID>
		<Ankunftszeit>2024-08-23T11:45:00</Ankunftszeit>
		<AnkunftssteigText>1</AnkunftssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>65</LinienText>
	<ProduktID>OVO65</ProduktID>
	<RichtungsText>Melaune</RichtungsText>
	<PrognoseMoeglich>true</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, match_shorter_transport_1) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, ovo65_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 23});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_ovo65);
  u.update(rtt, doc);

  // the shorter transport is a better match as it has a better matching ratio
  auto const fr0 = rt::frun{
      tt,
      &rtt,
      {{transport_idx_t{0U}, day_idx_t{27}}, {stop_idx_t{0}, stop_idx_t{31U}}}};
  EXPECT_FALSE(fr0.is_rt());

  auto const fr1 = rt::frun{
      tt,
      &rtt,
      {{transport_idx_t{1U}, day_idx_t{27}}, {stop_idx_t{0}, stop_idx_t{15U}}}};
  EXPECT_TRUE(fr1.is_rt());
}

namespace {

mem_dir rbo512_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"6750606_3",268,2593443891,"Bautzen August-Bebel-Pl (ZOB)","",1,,49668,0,0
"de:von:27-512_3",268,2593399553,"Bautzen August-Bebel-Pl (ZOB)","",1,,89076,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"6750606_3",8197,"512","",3,"","",""
"de:von:27-512_3",7874,"512/267","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8197,"RBO-Busverkehr","https://www.delfi.de","Europe/Berlin","",""
7874,"RBO-Bus","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593443891,16:02:00,16:02:00,"de:14628:3880:3:5",0,0,0,""
2593443891,16:04:00,16:04:00,"de:14628:3889:0:1",1,0,0,""
2593443891,16:06:00,16:06:00,"de:14628:3881:0:2",2,0,0,""
2593443891,16:08:00,16:08:00,"de:14628:3852:0:1",3,0,0,""
2593443891,16:16:00,16:16:00,"de:14628:3863:0:1",4,0,0,""
2593443891,16:20:00,16:20:00,"de:14628:3864:0:1",5,0,0,""
2593443891,16:21:00,16:21:00,"de:14628:3865:0:1",6,0,0,""
2593443891,16:23:00,16:23:00,"de:14625:7254:0:2",7,0,0,""
2593443891,16:26:00,16:26:00,"de:14625:7251:0:2",8,0,0,""
2593443891,16:27:00,16:27:00,"de:14625:7250:0:2",9,0,0,""
2593443891,16:29:00,16:29:00,"de:14625:7241:0:2",10,0,0,""
2593443891,16:30:00,16:30:00,"de:14625:7240:0:2",11,0,0,""
2593443891,16:32:00,16:32:00,"de:14625:7230:3:2",12,0,0,""
2593443891,16:34:00,16:34:00,"de:14625:7231:0:2",13,0,0,""
2593443891,16:35:00,16:35:00,"de:14625:7232:0:2",14,0,0,""
2593443891,16:37:00,16:37:00,"de:14625:7449:0:2",15,0,0,""
2593443891,16:38:00,16:38:00,"de:14625:7442:0:2",16,0,0,""
2593443891,16:40:00,16:40:00,"de:14625:7455:0:2",17,0,0,""
2593443891,16:41:00,16:41:00,"de:14625:7420:0:2",18,0,0,""
2593443891,16:48:00,16:48:00,"de:14625:7468:0:2",19,0,0,""
2593443891,16:50:00,16:50:00,"de:14625:7513:0:2",20,0,0,""
2593443891,16:51:00,16:51:00,"de:14625:7512:0:2",21,0,0,""
2593443891,16:52:00,16:52:00,"de:14625:7504:0:2",22,0,0,""
2593443891,16:53:00,16:53:00,"de:14625:7509:0:2",23,0,0,""
2593443891,16:54:00,16:54:00,"de:14625:7500:3:2",24,0,0,""
2593443891,16:56:00,16:56:00,"de:14625:7501:0:11",25,0,0,""
2593399553,16:26:00,16:26:00,"de:14625:7251:0:2",0,0,0,""
2593399553,16:27:00,16:27:00,"de:14625:7250:0:2_G",1,0,0,""
2593399553,16:29:00,16:29:00,"de:14625:7241:0:2_G",2,0,0,""
2593399553,16:30:00,16:30:00,"de:14625:7240:0:2_G",3,0,0,""
2593399553,16:32:00,16:32:00,"de:14625:7230:3:2",4,0,0,""
2593399553,16:34:00,16:34:00,"de:14625:7231:0:2",5,0,0,""
2593399553,16:35:00,16:35:00,"de:14625:7232:0:2_G",6,0,0,""
2593399553,16:37:00,16:37:00,"de:14625:7449:0:2_G",7,0,0,""
2593399553,16:38:00,16:38:00,"de:14625:7442:0:2_G",8,0,0,""
2593399553,16:40:00,16:40:00,"de:14625:7455:0:2_G",9,0,0,""
2593399553,16:41:00,16:41:00,"de:14625:7420:0:2_G",10,0,0,""
2593399553,16:48:00,16:48:00,"de:14625:7468:0:2_G",11,0,0,""
2593399553,16:50:00,16:50:00,"de:14625:7513:0:2_G",12,0,0,""
2593399553,16:51:00,16:51:00,"de:14625:7512:0:2",13,0,0,""
2593399553,16:52:00,16:52:00,"de:14625:7504:0:2",14,0,0,""
2593399553,16:53:00,16:53:00,"de:14625:7509:0:2",15,0,0,""
2593399553,16:54:00,16:54:00,"de:14625:7500:3:2",16,0,0,""
2593399553,16:56:00,16:56:00,"de:14625:7501:0:11_G",17,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14625:7468:0:2","","Preuschwitz",,"51.158480000000","14.408214000000",0,,0,"",""
"de:14625:7420:0:2","","Neu-Drauschkowitz",,"51.144900000000","14.349203000000",0,,0,"",""
"de:14625:7455:0:2","","Weißnaußlitz",,"51.138317000000","14.347658000000",0,,0,"",""
"de:14625:7442:0:2","","Dretschen Zur Postschänke",,"51.125560000000","14.341208000000",0,,0,"",""
"de:14625:7449:0:2","","Neu-Diehmen",,"51.112619000000","14.331156000000",0,,0,"",""
"de:14625:7230:3:2","","Neukirch (Lausitz) Bf Ost","Bus","51.091088000000","14.321275000000",0,,0,"2","2"
"de:14625:7240:0:2","","Ringenhain Siedlerstraße",,"51.081309000000","14.328596000000",0,,0,"",""
"de:14625:7232:0:2","","Neukirch/Lausitz Sandhübel",,"51.100803000000","14.327365000000",0,,0,"",""
"de:14625:7501:0:11","","Bautzen August-Bebel-Pl (ZOB)",,"51.177006000000","14.433079000000",0,,0,"",""
"de:14628:3864:0:1","","Hohwald Hohwaldschänke",,"51.052993000000","14.279404000000",0,,0,"",""
"de:14628:3863:0:1","","Hohwaldklinik",,"51.047516000000","14.290400000000",0,,0,"",""
"de:14625:7504:0:2","","Bautzen Neusalzaer Straße","Bautzen Neusalzaer Straße","51.171504000000","14.424311000000",0,,0,"2","2"
"de:14625:7241:0:2","","Ringenhain Erbgericht",,"51.077534000000","14.337301000000",0,,0,"",""
"de:14625:7251:0:2","","Steinigtwolmsdorf Weifaer Straße","Steinigtwolmsdorf Weifaer Straße","51.064839000000","14.345619000000",0,,0,"2","2"
"de:14625:7254:0:2","","Steinigtwolmsdorf Waldhaus",,"51.060113000000","14.318948000000",0,,0,"",""
"de:14628:3880:3:5","","Neustadt Bahnhof","Bus","51.021705000000","14.212983000000",0,,0,"5","2"
"de:14625:7250:0:2","","Steinigtwolmsdorf Niederdorf",,"51.069761000000","14.341002000000",0,,0,"",""
"de:14625:7509:0:2","","Bautzen Packhofstraße","Bautzen Packhofstraße","51.173717000000","14.423817000000",0,,0,"2","2"
"de:14625:7513:0:2","","Bautzen Gewerbepark Wilthener Straße",,"51.166913000000","14.422254000000",0,,0,"",""
"de:14628:3852:0:1","","Langburkersdorf A-Schubert-Str","Langburkersdorf A-Schubert-Str","51.025056000000","14.232575000000",0,,0,"1","2"
"de:14625:7512:0:2","","Bautzen Zeppelinstraße","Bautzen Zeppelinstraße","51.169673000000","14.425255000000",0,,0,"2","2"
"de:14628:3865:0:1","","Hohwald Steinwerke",,"51.057934000000","14.294244000000",0,,0,"",""
"de:14625:7231:0:2","","Neukirch/Lausitz Bautzener Straße","Neukirch/Lausitz Bautzener Straße","51.093458000000","14.320754000000",0,,0,"2","2"
"de:14628:3881:0:2","","Neustadt Wilhelm-Kaulisch-Str.",,"51.026282000000","14.219316000000",0,,0,"",""
"de:14625:7500:3:2","","Bautzen Bahnhof","Bus","51.174061000000","14.428444000000",0,,0,"2","2"
"de:14628:3889:0:1","","Neustadt Neustadthalle",,"51.024672000000","14.209821000000",0,,0,"",""
"de:14625:7501:0:11_G","","Bautzen August-Bebel-Pl (ZOB)",,"51.176995000000","14.433052000000",0,,0,"",""
"de:14625:7513:0:2_G","","Bautzen Gewerbepark Wilthener Straße",,"51.166806000000","14.422227000000",0,,0,"",""
"de:14625:7442:0:2_G","","Dretschen Zur Postschänke",,"51.125442000000","14.341100000000",0,,0,"",""
"de:14625:7420:0:2_G","","Neu-Drauschkowitz",,"51.144793000000","14.349167000000",0,,0,"",""
"de:14625:7449:0:2_G","","Neu-Diehmen",,"51.112535000000","14.331057000000",0,,0,"",""
"de:14625:7232:0:2_G","","Neukirch/Lausitz Sandhübel",,"51.100696000000","14.327266000000",0,,0,"",""
"de:14625:7240:0:2_G","","Ringenhain Siedlerstraße",,"51.081287000000","14.328650000000",0,,0,"",""
"de:14625:7455:0:2_G","","Weißnaußlitz",,"51.138255000000","14.347667000000",0,,0,"",""
"de:14625:7250:0:2_G","","Steinigtwolmsdorf Niederdorf",,"51.069722000000","14.341029000000",0,,0,"",""
"de:14625:7241:0:2_G","","Ringenhain Erbgericht",,"51.077579000000","14.337094000000",0,,0,"",""
"de:14625:7468:0:2_G","","Preuschwitz",,"51.158525000000","14.408079000000",0,,0,"",""


# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
268,1,1,1,1,1,1,1,20240805,20241214

# calendar_dates.txt
"service_id","date","exception_type"
268,20240805,2
268,20240812,2
268,20240806,2
268,20240813,2
268,20240807,2
268,20240808,2
268,20240809,2
268,20240810,2
268,20240811,2

)__");
}

constexpr auto const update_rbo512 = R"(
<IstFahrt Zst="2024-08-26T15:29:21">
	<LinienID>RBO512</LinienID>
	<RichtungsID>2</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>RBO5557_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-26</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>true</Komplettfahrt>
	<UmlaufID>65021</UmlaufID>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14625:7251:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:26:00</Abfahrtszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7250:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:27:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:27:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7241:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:29:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:29:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7240:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:30:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:30:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7230:3:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:32:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:32:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7231:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:34:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:34:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7232:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:35:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:35:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7449:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:37:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:37:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7442:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:38:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:38:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7455:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:40:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:40:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7420:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:41:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:41:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7468:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:48:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:48:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7513:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:50:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:50:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7512:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:51:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:51:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7504:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:52:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:52:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7509:0:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:53:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:53:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7500:3:2</HaltID>
		<Abfahrtszeit>2024-08-26T14:54:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-26T14:54:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14625:7501:0:11</HaltID>
		<Ankunftszeit>2024-08-26T14:56:00</Ankunftszeit>
		<AnkunftssteigText>11</AnkunftssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>512</LinienText>
	<ProduktID>RBO512</ProduktID>
	<RichtungsText>Bautzen A.-Bebel-Platz</RichtungsText>
	<PrognoseMoeglich>false</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, match_shorter_transport_2) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, rbo512_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 26});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_rbo512);
  u.update(rtt, doc);

  auto const fr0 = rt::frun{tt,
                            &rtt,
                            {{transport_idx_t{0U}, day_idx_t{30U}},
                             {stop_idx_t{0U}, stop_idx_t{26U}}}};
  EXPECT_FALSE(fr0.is_rt());

  auto const fr1 = rt::frun{tt,
                            &rtt,
                            {{transport_idx_t{1U}, day_idx_t{30U}},
                             {stop_idx_t{0U}, stop_idx_t{18U}}}};
  EXPECT_TRUE(fr1.is_rt());
}

namespace {

mem_dir ev11_rvs261_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:11-11:EV11_3",1171,2593425252,"A.-/Leipziger Str.","",0,,47995,0,0
"de:vvo:15-261-b_3",1171,2593414015,"Dresden Hauptbahnhof","",0,,47609,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:11-11:EV11_3",8194,"EV 11","",3,"","",""
"de:vvo:15-261-b_3",8193,"261","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8194,"DVB-Bus","https://www.delfi.de","Europe/Berlin","",""
8193,"Busübernahme OVPS","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593425252,17:24:00,17:24:00,"de:14612:302:1:5",0,0,0,""
2593425252,17:25:00,17:25:00,"de:14612:302:1:2",1,0,0,""
2593425252,17:26:00,17:26:00,"de:14612:301:1:1",2,0,0,""
2593425252,17:27:00,17:27:00,"de:14612:305:1:1",3,0,0,""
2593425252,17:28:00,17:28:00,"de:14612:300:2:1",4,0,0,""
2593425252,17:29:00,17:29:00,"de:14612:299:1:1",5,0,0,""
2593425252,17:31:00,17:31:00,"de:14612:298:1:1",6,0,0,""
2593425252,17:33:00,17:33:00,"de:14612:297:1:1",7,0,0,""
2593425252,17:35:00,17:35:00,"de:14612:296:1:1",8,0,0,""
2593425252,17:36:00,17:36:00,"de:14612:295:1:1",9,0,0,""
2593425252,17:37:00,17:37:00,"de:14612:294:1:1",10,0,0,""
2593425252,17:38:00,17:38:00,"de:14612:293:1:1",11,0,0,""
2593425252,17:39:00,17:39:00,"de:14612:292:2:91",12,0,0,""
2593425252,17:41:00,17:41:00,"de:14612:291:1:1",13,0,0,""
2593425252,17:43:00,17:43:00,"de:14612:290:1:1",14,0,0,""
2593425252,17:45:00,17:45:00,"de:14612:12:1:1",15,0,0,""
2593425252,17:47:00,17:47:00,"de:14612:13:2:3",16,0,0,""
2593425252,17:49:00,17:49:00,"de:14612:16:6:92",17,0,0,""
2593425252,17:52:00,17:52:00,"de:14612:17:2:9",18,0,0,""
2593414015,16:15:00,16:15:00,"de:14628:3921:0:4",0,0,0,""
2593414015,16:16:00,16:16:00,"de:14628:3919:0:1",1,0,0,""
2593414015,16:17:00,16:17:00,"de:14628:3935:0:2",2,0,0,""
2593414015,16:19:00,16:19:00,"de:14628:3934:1:1",3,0,0,""
2593414015,16:20:00,16:20:00,"de:14628:3925:0:1",4,0,0,""
2593414015,16:23:00,16:23:00,"de:14628:3860:0:1",5,0,0,""
2593414015,16:25:00,16:25:00,"de:14628:3851:0:1",6,0,0,""
2593414015,16:27:00,16:27:00,"de:14628:3850:0:1",7,0,0,""
2593414015,16:30:00,16:35:00,"de:14628:3881:0:3",8,0,0,""
2593414015,16:36:00,16:36:00,"de:14628:3886:0:2",9,0,0,""
2593414015,16:38:00,16:38:00,"de:14628:3880:3:2",10,0,0,""
2593414015,16:40:00,16:40:00,"de:14628:3889:0:1",11,0,0,""
2593414015,16:42:00,16:42:00,"de:14628:3882:0:1",12,0,0,""
2593414015,16:44:00,16:44:00,"de:14628:3917:0:1",13,0,0,""
2593414015,16:45:00,16:45:00,"de:14628:3916:0:1",14,0,0,""
2593414015,16:46:00,16:46:00,"de:14628:3915:0:1",15,0,0,""
2593414015,16:48:00,16:48:00,"de:14628:3823:0:1",16,0,0,""
2593414015,16:49:00,16:49:00,"de:14628:3824:3:1",17,0,0,""
2593414015,16:51:00,16:51:00,"de:14628:3825:0:1",18,0,0,""
2593414015,16:53:00,16:53:00,"de:14628:3826:0:1",19,0,0,""
2593414015,16:55:00,16:55:00,"de:14628:3827:0:1",20,0,0,""
2593414015,16:57:00,16:58:00,"de:14628:3816:0:1",21,0,0,""
2593414015,16:59:00,16:59:00,"de:14628:3817:0:1",22,0,0,""
2593414015,17:02:00,17:02:00,"de:14628:3836:0:1",23,0,0,""
2593414015,17:06:00,17:06:00,"de:14628:3810:0:1",24,0,0,""
2593414015,17:09:00,17:09:00,"de:14625:6791:0:1",25,0,0,""
2593414015,17:13:00,17:13:00,"de:14612:6765:0:1",26,0,0,""
2593414015,17:16:00,17:16:00,"de:14612:6764:0:2",27,0,0,""
2593414015,17:17:00,17:17:00,"de:14612:6769:0:1",28,0,0,""
2593414015,17:19:00,17:19:00,"de:14612:3771:0:1",29,0,0,""
2593414015,17:22:00,17:22:00,"de:14612:3775:1:1",30,0,0,""
2593414015,17:23:00,17:23:00,"de:14612:3774:1:1",31,0,0,""
2593414015,17:26:00,17:26:00,"de:14612:302:5:12",32,0,0,""
2593414015,17:28:00,17:28:00,"de:14612:300:2:1",33,0,0,""
2593414015,17:31:00,17:31:00,"de:14612:298:1:1",34,0,0,""
2593414015,17:36:00,17:36:00,"de:14612:294:1:1",35,0,0,""
2593414015,17:37:00,17:37:00,"de:14612:293:1:1",36,0,0,""
2593414015,17:44:00,17:44:00,"de:14612:13:3:6",37,0,0,""
2593414015,17:48:00,17:48:00,"de:14612:5:3:5",38,0,0,""
2593414015,17:50:00,17:50:00,"de:14612:29:2:4",39,0,0,""
2593414015,17:53:00,17:53:00,"de:14612:28:2:6",40,0,0,""

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
"de:14612:29:2:4","","Dresden Walpurgisstraße","Walpurgis Bus","51.043574000000","13.737648000000",0,,0,"4","2"
"de:14612:293:1:1","","Dresden Waldschlößchen","Haltestelle","51.067453000000","13.776940000000",0,,0,"1","2"
"de:14612:298:1:1","","Dresden Plattleite","Haltestelle","51.063761000000","13.821497000000",0,,0,"1","2"
"de:14612:300:2:1","","Dresden Schwimmhalle Bühlau","Haltestelle Btf","51.062259000000","13.838727000000",0,,0,"1","2"
"de:14612:3775:1:1","","Weißig Am Weißiger Bach","Haltestelle","51.062197000000","13.886158000000",0,,0,"1","2"
"de:14612:6769:0:1","","Rossendorf Schänkhübel",,"51.063676000000","13.934846000000",0,,0,"",""
"de:14612:6764:0:2","","Rossendorf Siedlung B 6",,"51.063772000000","13.940308000000",0,,0,"",""
"de:14628:3810:0:1","","Wilschdorf Gasthaus",,"51.061802000000","14.012928000000",0,,0,"",""
"de:14628:3817:0:1","","Stolpen Schützenhausstraße",,"51.050384000000","14.079394000000",0,,0,"",""
"de:14628:3850:0:1","","Langburkersdorf Sebnitzer Str.","Langburkersdorf Sebnitzer Str.","51.024276000000","14.231955000000",0,,0,"1","2"
"de:14628:3827:0:1","","Langenwolmsdorf Niederdorf","Langenwolmsdorf Niederdorf","51.045082000000","14.100963000000",0,,0,"1","2"
"de:14628:3823:0:1","","Langenwolmsdorf Bahnübergang",,"51.036581000000","14.160422000000",0,,0,"",""
"de:14628:3824:3:1","","Langenwolmsdorf Bahnhof","Bus","51.036666000000","14.153371000000",0,,0,"1","2"
"de:14628:3917:0:1","","Polenz Erbgericht",,"51.024225000000","14.185566000000",0,,0,"",""
"de:14628:3889:0:1","","Neustadt Neustadthalle",,"51.024672000000","14.209821000000",0,,0,"",""
"de:14612:6765:0:1","","Rossendorf Forschungszentrum",,"51.063151000000","13.950109000000",0,,0,"",""
"de:14628:3921:0:4","","Sebnitz Busbahnhof","Sebnitz Busbahnhof","50.967195000000","14.270646000000",0,,0,"4","2"
"de:14628:3915:0:1","","Polenz Am Polenztal",,"51.028203000000","14.172433000000",0,,0,"",""
"de:14628:3916:0:1","","Polenz Schmiede",,"51.025440000000","14.179889000000",0,,0,"",""
"de:14628:3880:3:2","","Neustadt Bahnhof","Bus","51.021649000000","14.212731000000",0,,0,"2","2"
"de:14612:28:2:6","","Dresden Hauptbahnhof","Hbf Am Hbf","51.039727000000","13.733767000000",0,,0,"6","2"
"de:14612:3774:1:1","","Weißig Am Steinkreuz","Hst. Ri Bühlau","51.060023000000","13.878226000000",0,,0,"1","2"
"de:14628:3816:0:1","","Stolpen Ärztehaus",,"51.050701000000","14.087066000000",0,,0,"",""
"de:14625:6791:0:1","","Wilschdorf Abzw. Wilschdorf B6",,"51.063845000000","13.981667000000",0,,0,"",""
"de:14628:3836:0:1","","Rennersdorf Kammergut",,"51.057810000000","14.063108000000",0,,0,"",""
"de:14628:3925:0:1","","Sebnitz Dr.-Steudner-Straße",,"50.982852000000","14.262148000000",0,,0,"",""
"de:14628:3881:0:3","","Neustadt Wilhelm-Kaulisch-Str.",,"51.026271000000","14.219289000000",0,,0,"",""
"de:14628:3825:0:1","","Langenwolmsdorf Oberdorf",,"51.039394000000","14.132449000000",0,,0,"",""
"de:14628:3860:0:1","","Rugiswalde Abzweig",,"50.999391000000","14.244046000000",0,,0,"",""
"de:14628:3851:0:1","","Langburkersdorf Tännicht",,"51.012720000000","14.232153000000",0,,0,"",""
"de:14612:302:5:12","","Dresden Bühlau Ullersdorfer Pl","Bus_B6","51.062123000000","13.853055000000",0,,0,"12","2"
"de:14628:3935:0:2","","Sebnitz Brückenschänke","Schandauer","50.968728000000","14.257377000000",0,,0,"2","2"
"de:14628:3934:1:1","","Sebnitz Schönbacher Weg","Nord","50.977021000000","14.261339000000",0,,0,"1","2"
"de:14612:13:3:6","","Dresden Albertplatz","Albertstraße","51.061610000000","13.746003000000",0,,0,"6","2"
"de:14628:3826:0:1","","Langenwolmsdorf Ortszentrum",,"51.042049000000","14.119118000000",0,,0,"",""
"de:14612:5:3:5","","Dresden Pirnaischer Platz","Petersburger Str","51.048888000000","13.744484000000",0,,0,"5","2"
"de:14612:3771:0:1","","Weißig Fußweg zum Napoleonstein",,"51.060763000000","13.919027000000",0,,0,"",""
"de:14628:3919:0:1","","Sebnitz Schandauer Straße",,"50.968830000000","14.269837000000",0,,0,"",""
"de:14628:3882:0:1","","Neustadt Abzweig Polenz",,"51.029999000000","14.201933000000",0,,0,"",""
"de:14612:294:1:1","","Dresden Angelikastraße","Haltestelle","51.067035000000","13.782959000000",0,,0,"1","2"
"de:14628:3886:0:2","","Neustadt Friedenseck",,"51.022474000000","14.219478000000",0,,0,"",""

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

TEST(vdv_update, match_261_not_11) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, ev11_rvs261_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 19});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_rvs261);
  u.update(rtt, doc);

  auto fr_ev11 = rt::frun{
      tt,
      &rtt,
      {{transport_idx_t{0}, day_idx_t{23}}, {stop_idx_t{0}, stop_idx_t{19}}}};
  EXPECT_FALSE(fr_ev11.is_rt());

  auto fr_rvs261 = rt::frun{
      tt,
      &rtt,
      {{transport_idx_t{1}, day_idx_t{23}}, {stop_idx_t{0}, stop_idx_t{41}}}};
  EXPECT_TRUE(fr_rvs261.is_rt());
}

namespace {

mem_dir rvs360_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:23-360_3",1171,2593428077,"Bannewitz Windbergstraße","",0,85739,48341,0,0
"de:vvo:21-66_3",1171,2593419356,"Freital-Deuben","",0,85071,47843,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:23-360_3",8195,"360","",3,"","",""
"de:vvo:21-66_3",8194,"66","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8195,"RVD-Busverkehr","https://www.delfi.de","Europe/Berlin","",""
8194,"DVB-Bus","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593428077,16:54:00,16:54:00,"de:14612:28:2:6",0,0,0,""
2593428077,16:56:00,16:56:00,"de:14612:131:2:4",1,0,0,""
2593428077,16:58:00,16:58:00,"de:14612:727:2:4",2,0,0,""
2593428077,16:59:00,16:59:00,"de:14612:732:1:2",3,0,0,""
2593428077,17:01:00,17:01:00,"de:14612:733:1:2",4,0,0,""
2593428077,17:05:00,17:05:00,"de:14628:1072:0:1",5,0,0,""
2593428077,17:07:00,17:07:00,"de:14628:1070:0:1",6,0,0,""
2593428077,17:09:00,17:09:00,"de:14628:1076:0:1",7,0,0,""
2593419356,16:29:00,16:29:00,"de:14612:935:1:3",0,0,0,""
2593419356,16:30:00,16:30:00,"de:14612:935:1:2",1,0,0,""
2593419356,16:31:00,16:31:00,"de:14612:938:1:1",2,0,0,""
2593419356,16:32:00,16:32:00,"de:14612:934:1:1",3,0,0,""
2593419356,16:33:00,16:33:00,"de:14612:792:1:1",4,0,0,""
2593419356,16:34:00,16:34:00,"de:14612:324:1:1",5,0,0,""
2593419356,16:36:00,16:36:00,"de:14612:790:1:1",6,0,0,""
2593419356,16:37:00,16:37:00,"de:14612:789:1:1",7,0,0,""
2593419356,16:39:00,16:39:00,"de:14612:788:1:1",8,0,0,""
2593419356,16:40:00,16:40:00,"de:14612:787:1:1",9,0,0,""
2593419356,16:42:00,16:42:00,"de:14612:786:1:1",10,0,0,""
2593419356,16:44:00,16:44:00,"de:14612:785:1:1",11,0,0,""
2593419356,16:46:00,16:46:00,"de:14612:784:1:1",12,0,0,""
2593419356,16:47:00,16:47:00,"de:14612:309:1:2",13,0,0,""
2593419356,16:49:00,16:49:00,"de:14612:311:2:4",14,0,0,""
2593419356,16:50:00,16:50:00,"de:14612:782:1:2",15,0,0,""
2593419356,16:52:00,16:52:00,"de:14612:30:1:2",16,0,0,""
2593419356,16:54:00,16:54:00,"de:14612:28:2:6",17,0,0,""
2593419356,16:56:00,16:56:00,"de:14612:131:2:4",18,0,0,""
2593419356,16:58:00,16:58:00,"de:14612:727:2:4",19,0,0,""
2593419356,16:59:00,16:59:00,"de:14612:732:1:2",20,0,0,""
2593419356,17:01:00,17:01:00,"de:14612:733:1:2",21,0,0,""
2593419356,17:02:00,17:02:00,"de:14612:743:1:2",22,0,0,""
2593419356,17:03:00,17:03:00,"de:14612:740:1:2",23,0,0,""
2593419356,17:05:00,17:05:00,"de:14612:741:1:2",24,0,0,""
2593419356,17:06:00,17:06:00,"de:14612:164:1:2",25,0,0,""
2593419356,17:07:00,17:07:00,"de:14612:165:1:5",26,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14628:1070:0:1","","Bannewitz Boderitzer Straße","Bannewitz Boderitzer Straße","50.998763000000","13.720257000000",0,,0,"1","2"
"de:14628:1072:0:1","","Bannewitz Nöthnitz",,"51.003959000000","13.727120000000",0,,0,"",""
"de:14612:733:1:2","","Dresden Südhöhe","Bergstraße","51.019846000000","13.728854000000",0,,0,"2","2"
"de:14612:732:1:2","","Dresden Mommsenstraße","Haltestelle","51.027480000000","13.731270000000",0,,0,"2","2"
"de:14612:131:2:4","","Dresden Reichenbachstraße","Fritz-Löffler Bus","51.034705000000","13.731144000000",0,,0,"4","2"
"de:14612:727:2:4","","Dresden Technische Universität","Ri. Südhöhe","51.029570000000","13.730713000000",0,,0,"4","2"
"de:14628:1076:0:1","","Bannewitz Windbergstraße","Bannewitz Windbergstraße","50.993833000000","13.717616000000",0,,0,"1","2"
"de:14612:28:2:6","","Dresden Hauptbahnhof","Hbf Am Hbf","51.039727000000","13.733767000000",0,,0,"6","2"
"de:14612:741:1:2","","Dresden Cunnersdorfer Straße","Haltestelle","51.018105000000","13.707096000000",0,,0,"2","2"
"de:14612:740:1:2","","Dresden Dorfhainer Straße","Haltestelle","51.018812000000","13.718981000000",0,,0,"2","2"
"de:14612:743:1:2","","Dresden Höckendorfer Weg","Haltestelle","51.019433000000","13.725485000000",0,,0,"2","2"
"de:14612:164:1:2","","Dresden Achtbeeteweg","Haltestelle","51.017648000000","13.703216000000",0,,0,"2","2"
"de:14612:934:1:1","","Dresden Theilestraße","Haltestelle","50.990633000000","13.802390000000",0,,0,"1","2"
"de:14612:784:1:1","","DD C-D-Friedrich-Straße","Haltestelle","51.026864000000","13.755201000000",0,,0,"1","2"
"de:14612:938:1:1","","Dresden Hänichenweg","Haltestelle","50.988298000000","13.802623000000",0,,0,"1","2"
"de:14612:30:1:2","","Dresden Gret-Palucca-Straße","Zentralhst.","51.039010000000","13.739346000000",0,,0,"2","2"
"de:14612:785:1:1","","Dresden Corinthstraße","Ri. Hbf","51.021394000000","13.764157000000",0,,0,"1","2"
"de:14612:782:1:2","","Dresden Uhlandstraße","Haltestelle","51.036920000000","13.739481000000",0,,0,"2","2"
"de:14612:787:1:1","","Dresden Marie-Wittich-Straße","Haltestelle","51.013127000000","13.777282000000",0,,0,"1","2"
"de:14612:311:2:4","","Dresden Strehlener Platz","Strehlener Str.","51.034152000000","13.748060000000",0,,0,"4","2"
"de:14612:309:1:2","","Dresden Weberplatz","Haltestelle","51.031237000000","13.751060000000",0,,0,"2","2"
"de:14612:788:1:1","","Dresden Tornaer Straße","Haltestelle","51.009849000000","13.783651000000",0,,0,"1","2"
"de:14612:789:1:1","","Dresden Gamigstraße","Haltestelle","51.005581000000","13.788439000000",0,,0,"1","2"
"de:14612:790:1:1","","Dresden Fritz-Meinhardt-Straße","B 172","51.001986000000","13.793568000000",0,,0,"1","2"
"de:14612:792:1:1","","Dresden Erich-Kästner-Straße","Dohnaer stw","50.995829000000","13.803405000000",0,,0,"1","2"
"de:14612:324:1:1","","DD Prohlis Kaufpark Nickern","Haltestelle","50.998334000000","13.800099000000",0,,0,"1","2"
"de:14612:165:1:5","","Dresden Coschütz","Haltestelle","51.015935000000","13.699353000000",0,,0,"2","2"
"de:14612:935:1:2","","Dresden Lockwitz","Am Plan","50.987269000000","13.806872000000",0,,0,"2","2"
"de:14612:786:1:1","","Dresden Spitzwegstraße","Haltestelle","51.015568000000","13.771757000000",0,,0,"1","2"
"de:14612:935:1:3","","Dresden Lockwitz","Am Plan","50.987467000000","13.806540000000",0,,0,"3","2"

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

constexpr auto const update_rvs360 = R"(
<IstFahrt Zst="2024-08-27T16:34:15">
	<LinienID>RVS360</LinienID>
	<RichtungsID>1</RichtungsID>
	<FahrtRef>
		<FahrtID>
			<FahrtBezeichner>RVS24204_vvorbl</FahrtBezeichner>
			<Betriebstag>2024-08-27</Betriebstag>
		</FahrtID>
	</FahrtRef>
	<Komplettfahrt>true</Komplettfahrt>
	<UmlaufID>33301</UmlaufID>
	<BetreiberID>vvorbl</BetreiberID>
	<IstHalt>
		<HaltID>de:14612:28:2:6</HaltID>
		<Abfahrtszeit>2024-08-27T14:54:00</Abfahrtszeit>
		<AbfahrtssteigText>6</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:131:2:4</HaltID>
		<Abfahrtszeit>2024-08-27T14:56:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-27T14:56:00</Ankunftszeit>
		<AbfahrtssteigText>4</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:727:2:4</HaltID>
		<Abfahrtszeit>2024-08-27T14:58:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-27T14:58:00</Ankunftszeit>
		<AbfahrtssteigText>4</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:732:1:2</HaltID>
		<Abfahrtszeit>2024-08-27T14:59:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-27T14:59:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14612:733:1:2</HaltID>
		<Abfahrtszeit>2024-08-27T15:01:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-27T15:01:00</Ankunftszeit>
		<AbfahrtssteigText>2</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14628:1072:0:1</HaltID>
		<Abfahrtszeit>2024-08-27T15:05:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-27T15:05:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14628:1070:0:1</HaltID>
		<Abfahrtszeit>2024-08-27T15:07:00</Abfahrtszeit>
		<Ankunftszeit>2024-08-27T15:07:00</Ankunftszeit>
		<AbfahrtssteigText>1</AbfahrtssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<IstHalt>
		<HaltID>de:14628:1076:0:1</HaltID>
		<Ankunftszeit>2024-08-27T15:09:00</Ankunftszeit>
		<AnkunftssteigText>1</AnkunftssteigText>
		<Besetztgrad>Unbekannt</Besetztgrad>
	</IstHalt>
	<LinienText>360</LinienText>
	<ProduktID>RVS360</ProduktID>
	<RichtungsText>Bannewitz - 164 Deuben</RichtungsText>
	<PrognoseMoeglich>true</PrognoseMoeglich>
	<FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, match_360_not_66) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, rvs360_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 27});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(update_rvs360);
  u.update(rtt, doc);

  // Should only match line 360
  auto const fr0 = rt::frun{tt,
                            &rtt,
                            {{transport_idx_t{0U}, day_idx_t{31U}},
                             {stop_idx_t{0U}, stop_idx_t{8U}}}};
  EXPECT_TRUE(fr0.is_rt());

  // should not match line 66
  auto const fr1 = rt::frun{tt,
                            &rtt,
                            {{transport_idx_t{1U}, day_idx_t{31U}},
                             {stop_idx_t{0U}, stop_idx_t{27U}}}};
  EXPECT_FALSE(fr1.is_rt());
}

namespace {

mem_dir vgm418_files() {
  return mem_dir::read(R"__(
# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"de:vvo:24-418_3",1171,2593430133,"Meißen Busbahnhof","",1,,48900,0,0

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"de:vvo:24-418_3",8196,"418","",3,"","",""

# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
8196,"VGM-Busverkehr","https://www.delfi.de","Europe/Berlin","",""

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2593430133,17:37:00,17:37:00,"de:14627:4850:0:1",0,0,0,""
2593430133,17:40:00,17:40:00,"de:14522:80004:0:BS_G",1,0,0,""
2593430133,17:41:00,17:41:00,"de:14522:98055:0:BO",2,0,0,""
2593430133,17:43:00,17:43:00,"de:14627:4848:0:2",3,0,0,""
2593430133,17:45:00,17:45:00,"de:14627:4856:0:2",4,0,0,""
2593430133,17:47:00,17:47:00,"de:14627:4857:0:2",5,0,0,""
2593430133,17:49:00,17:49:00,"de:14627:4872:0:2",6,0,0,""
2593430133,17:51:00,17:51:00,"de:14627:4870:0:2",7,0,0,""
2593430133,17:53:00,17:53:00,"de:14627:4809:0:2",8,0,0,""
2593430133,17:55:00,17:55:00,"de:14627:4800:3:1",9,0,0,""
2593430133,17:56:00,17:56:00,"de:14627:4793:0:1_G",10,0,0,""
2593430133,17:57:00,17:57:00,"de:14627:4797:0:1_G",11,0,0,""
2593430133,17:59:00,18:03:00,"de:14627:4801:0:1",12,0,0,""
2593430133,18:05:00,18:05:00,"de:14627:4804:0:2_G",13,0,0,""
2593430133,18:06:00,18:06:00,"de:14627:4805:0:2_G",14,0,0,""
2593430133,18:07:00,18:07:00,"de:14627:4806:0:2_G",15,0,0,""
2593430133,18:09:00,18:09:00,"de:14627:4810:1:1",16,0,0,""
2593430133,18:10:00,18:10:00,"de:14627:4816:0:2",17,0,0,""
2593430133,18:11:00,18:11:00,"de:14627:4811:0:2",18,0,0,""
2593430133,18:12:00,18:12:00,"de:14627:4817:0:2",19,0,0,""
2593430133,18:13:00,18:13:00,"de:14627:4813:0:2",20,0,0,""
2593430133,18:14:00,18:14:00,"de:14627:4814:0:2",21,0,0,""
2593430133,18:15:00,18:15:00,"de:14627:4824:0:2",22,0,0,""
2593430133,18:16:00,18:16:00,"de:14627:4825:0:2",23,0,0,""
2593430133,18:17:00,18:17:00,"de:14627:4826:0:2",24,0,0,""
2593430133,18:18:00,18:18:00,"de:14627:4819:0:2",25,0,0,""
2593430133,18:19:00,18:19:00,"de:14627:4837:0:2",26,0,0,""
2593430133,18:20:00,18:20:00,"de:14627:4836:0:2",27,0,0,""
2593430133,18:21:00,18:21:00,"de:14627:4838:0:2",28,0,0,""
2593430133,18:22:00,18:22:00,"de:14627:4693:0:2",29,0,0,""
2593430133,18:23:00,18:23:00,"de:14627:4692:0:2",30,0,0,""
2593430133,18:24:00,18:24:00,"de:14627:4691:3:2",31,0,0,""
2593430133,18:25:00,18:25:00,"de:14627:4690:0:2",32,0,0,""
2593430133,18:28:00,18:28:00,"de:14627:4688:0:2",33,0,0,""
2593430133,18:29:00,18:29:00,"de:14627:4689:0:2",34,0,0,""
2593430133,18:31:00,18:31:00,"de:14627:4685:0:2",35,0,0,""
2593430133,18:32:00,18:32:00,"de:14627:4684:0:2",36,0,0,""
2593430133,18:34:00,18:34:00,"de:14627:4088:0:2",37,0,0,""
2593430133,18:35:00,18:35:00,"de:14627:4083:0:2",38,0,0,""
2593430133,18:36:00,18:36:00,"de:14627:4082:1:2",39,0,0,""
2593430133,18:37:00,18:37:00,"de:14627:4081:1:2",40,0,0,""
2593430133,18:38:00,18:38:00,"de:14627:4080:1:2",41,0,0,""
2593430133,18:39:00,18:39:00,"de:14627:4077:1:2",42,0,0,""
2593430133,18:40:00,18:40:00,"de:14627:4076:1:2",43,0,0,""
2593430133,18:41:00,18:41:00,"de:14627:4018:1:2",44,0,0,""
2593430133,18:42:00,18:42:00,"de:14627:4075:1:2",45,0,0,""
2593430133,18:44:00,18:44:00,"de:14627:4013:1:3",46,0,0,""
2593430133,18:45:00,18:45:00,"de:14627:4055:3:2",47,0,0,""
2593430133,18:46:00,18:46:00,"de:14627:4048:1:2",48,0,0,""
2593430133,18:48:00,18:48:00,"de:14627:4010:1:4",49,0,0,""
2593430133,18:50:00,18:50:00,"de:14627:4007:1:1",50,0,0,""

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:14627:4007:1:1","","Meißen Busbahnhof","Busbahnhof","51.163736000000","13.483308000000",0,,0,"1","2"
"de:14627:4010:1:4","","Meißen Bahnhofstraße","SüdOst Ri Busbf","51.164040000000","13.478421000000",0,,0,"2","2"
"de:14627:4048:1:2","","Meißen Uferstraße","Haltestelle","51.163049000000","13.475798000000",0,,0,"2","2"
"de:14627:4055:3:2","","Meißen S-Bahnhof Altstadt","3","51.160615000000","13.473265000000",0,,0,"2","2"
"de:14627:4684:0:2","","Garsebach Mittelmühle",,"51.129625000000","13.436964000000",0,,0,"",""
"de:14627:4013:1:3","","Meißen Talbad","Kerstingstraße","51.157376000000","13.468513000000",0,,0,"3","2"
"de:14627:4018:1:2","","Meißen W.-Walkhoff-Platz","Haltestelle","51.152722000000","13.459835000000",0,,0,"2","2"
"de:14627:4076:1:2","","Meißen Schützestraße","Haltestelle","51.150558000000","13.458865000000",0,,0,"2","2"
"de:14627:4077:1:2","","Meißen Hohe Eifer","Haltestelle","51.144973000000","13.459718000000",0,,0,"2","2"
"de:14627:4685:0:2","","Garsebach Wendeplatz",,"51.126051000000","13.431350000000",0,,0,"",""
"de:14627:4811:0:2","","Deutschenbora Abzweig Meißen",,"51.053727000000","13.358649000000",0,,0,"",""
"de:14627:4856:0:2","","Abzw Starbach",,"51.089988000000","13.267039000000",0,,0,"",""
"de:14627:4693:0:2","","Miltitz Bad",,"51.095449000000","13.414848000000",0,,0,"",""
"de:14627:4826:0:2","","Rothschönberg Abzw Kottewitz",,"51.071726000000","13.398094000000",0,,0,"",""
"de:14627:4836:0:2","","Munzig Abzw Burkhardswalde","Munzig Abzw Burkhardswalde","51.085000000000","13.411227000000",0,,0,"2","2"
"de:14627:4805:0:2_G","","Nossen Siedlung Eula",,"51.057629000000","13.316141000000",0,,0,"",""
"de:14627:4837:0:2","","Munzig Abzw Heynitz",,"51.079531000000","13.404256000000",0,,0,"",""
"de:14627:4080:1:2","","Meißen Kühnestraße","Haltestelle","51.140899000000","13.458676000000",0,,0,"2","2"
"de:14627:4819:0:2","","Rothschönberg Appenhof",,"51.074802000000","13.401840000000",0,,0,"",""
"de:14627:4690:0:2","","Miltitz OT Roitzschen",,"51.110228000000","13.416680000000",0,,0,"",""
"de:14627:4838:0:2","","Munzig Kulturhaus",,"51.090208000000","13.413015000000",0,,0,"",""
"de:14627:4825:0:2","","Rothschönberg Weg zum Gasthof",,"51.069338000000","13.392515000000",0,,0,"",""
"de:14627:4689:0:2","","Robschütz",,"51.122848000000","13.422852000000",0,,0,"",""
"de:14627:4806:0:2_G","","Nossen Eula Neuer Weg",,"51.056856000000","13.328115000000",0,,0,"",""
"de:14627:4824:0:2","","Rothschönberg Rote Mühle",,"51.067114000000","13.381053000000",0,,0,"",""
"de:14627:4816:0:2","","Deutschenbora Bahnhofstraße","Deutschenbora Bahnhofstraße","51.053840000000","13.350618000000",0,,0,"2","2"
"de:14627:4081:1:2","","Meißen Zuckerhut","Haltestelle","51.136987000000","13.459206000000",0,,0,"2","2"
"de:14627:4813:0:2","","Deutschenbora Elgersdorfer Str.",,"51.060554000000","13.362395000000",0,,0,"",""
"de:14627:4088:0:2","","Meißen Steinbruch",,"51.134383000000","13.447825000000",0,,0,"",""
"de:14627:4075:1:2","","Meißen Niesnerstraße","Haltestelle","51.154119000000","13.461506000000",0,,0,"2","2"
"de:14627:4817:0:2","","Deutschenbora Am Fußweg",,"51.058070000000","13.359017000000",0,,0,"",""
"de:14627:4797:0:1_G","","Nossen Grundschule",,"51.057375000000","13.295677000000",0,,0,"",""
"de:14627:4793:0:1_G","","Nossen August-Bebel-Straße",,"51.059233000000","13.298237000000",0,,0,"",""
"de:14627:4809:0:2","","Nossen Brücke",,"51.063857000000","13.286901000000",0,,0,"",""
"de:14627:4688:0:2","","Robschütz Abzweig Luga",,"51.119696000000","13.418252000000",0,,0,"",""
"de:14627:4810:1:1","","Deutschenbora Hirschfelder Str","Hauptstr. (Eula)","51.054382000000","13.344132000000",0,,0,"1","2"
"de:14627:4801:0:1","","Nossen Markt","Markt","51.057573000000","13.298911000000",0,,0,"1","2"
"de:14627:4800:3:1","","Nossen Bahnhof","Bus","51.060435000000","13.293494000000",0,,0,"1","2"
"de:14627:4691:3:2","","Miltitz Bahnhof","Bus","51.105614000000","13.418594000000",0,,0,"2","2"
"de:14627:4804:0:2_G","","Nossen Bahnübergang",,"51.058126000000","13.306367000000",0,,0,"",""
"de:14627:4082:1:2","","Meißen Buschbad","Haltestelle","51.134834000000","13.458353000000",0,,0,"2","2"
"de:14522:98055:0:BO","","Choren, Dorfeingang",,"51.098823000000","13.248974000000",0,,0,"",""
"de:14627:4872:0:2","","Neubodenbach",,"51.080248000000","13.278789000000",0,,0,"",""
"de:14627:4814:0:2","","Deutschenbora Klotz-Mühle",,"51.062462000000","13.369644000000",0,,0,"",""
"de:14627:4857:0:2","","Starbach Logistikzentrum",,"51.087477000000","13.269204000000",0,,0,"",""
"de:14627:4848:0:2","","Starbach Dorfplatz",,"51.092894000000","13.269518000000",0,,0,"",""
"de:14627:4870:0:2","","Rhäsa",,"51.069999000000","13.285059000000",0,,0,"",""
"de:14522:80004:0:BS_G","","Choren, Wendestelle",,"51.101260000000","13.242012000000",0,,0,"",""
"de:14627:4083:0:2","","Meißen Abzweig Dobritz",,"51.136299000000","13.451714000000",0,,0,"",""
"de:14627:4692:0:2","","Miltitz Mühle",,"51.101700000000","13.414461000000",0,,0,"",""
"de:14627:4850:0:1","","Rüsseina Wendeplatz",,"51.111221000000","13.261379000000",0,,0,"",""
"de:14522:80004:0:BS","","Choren, Wendestelle",,"51.101260000000","13.241994000000",0,,0,"",""
"de:14627:4793:0:1","","Nossen August-Bebel-Straße",,"51.059204000000","13.298219000000",0,,0,"",""
"de:14627:4797:0:1","","Nossen Grundschule",,"51.057420000000","13.295578000000",0,,0,"",""
"de:14627:4804:0:2","","Nossen Bahnübergang",,"51.058103000000","13.306313000000",0,,0,"",""
"de:14627:4805:0:2","","Nossen Siedlung Eula",,"51.057595000000","13.315970000000",0,,0,"",""
"de:14627:4806:0:2","","Nossen Eula Neuer Weg",,"51.056850000000","13.328160000000",0,,0,"",""
"de:14627:4810:1:2","","Deutschenbora Hirschfelder Str","Hauptstr. (Eula)","51.054309000000","13.344312000000",0,,0,"2","2"
"de:14627:4010:1:2","","Meißen Bahnhofstraße","Nordwest Ri Brücke","51.164598000000","13.477784000000",0,,0,"1","2"

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

// prognosis: arrival at second stop before departure at first
constexpr auto const update_vgm418_0 = R"(
<IstFahrt Zst="2024-08-26T17:49:13">
  <LinienID>VGM418</LinienID>
  <RichtungsID>2</RichtungsID>
  <FahrtRef>
    <FahrtID>
      <FahrtBezeichner>VGM41802401_vvorbl</FahrtBezeichner>
      <Betriebstag>2024-08-26</Betriebstag>
    </FahrtID>
  </FahrtRef>
  <Komplettfahrt>true</Komplettfahrt>
  <BetreiberID>vvorbl</BetreiberID>
  <IstHalt>
    <HaltID>de:14627:4850:0:1</HaltID>
    <Abfahrtszeit>2024-08-26T15:37:00</Abfahrtszeit>
    <IstAbfahrtPrognose>2024-08-26T15:58:04</IstAbfahrtPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14522:80004:0:BS</HaltID>
    <Abfahrtszeit>2024-08-26T15:40:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:40:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T15:46:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T15:46:07</IstAnkunftPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14522:98055:0:BO</HaltID>
    <Abfahrtszeit>2024-08-26T15:41:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:41:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T15:47:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T15:47:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4848:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T15:43:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:43:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T15:49:23</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T15:49:23</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4856:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T15:45:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:45:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T15:51:23</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T15:51:23</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4857:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T15:47:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:47:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T15:52:25</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T15:52:25</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4872:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T15:49:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:49:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T15:54:25</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T15:54:25</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4870:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T15:51:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:51:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T15:56:25</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T15:56:25</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4809:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T15:53:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:53:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T15:58:25</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T15:58:25</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4800:3:1</HaltID>
    <Abfahrtszeit>2024-08-26T15:55:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:55:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:00:25</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:00:25</IstAnkunftPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4793:0:1</HaltID>
    <Abfahrtszeit>2024-08-26T15:56:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:56:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:01:25</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:01:25</IstAnkunftPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4797:0:1</HaltID>
    <Abfahrtszeit>2024-08-26T15:57:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:57:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:02:25</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:02:25</IstAnkunftPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4801:0:1</HaltID>
    <Abfahrtszeit>2024-08-26T16:03:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T15:59:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:04:25</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:04:25</IstAnkunftPrognose>
    <AbfahrtssteigText>1</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4804:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:05:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:05:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:07:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:07:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4805:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:06:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:06:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:08:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:08:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4806:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:07:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:07:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:09:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:09:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4810:1:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:09:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:09:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:11:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:11:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4816:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:10:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:10:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:12:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:12:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4811:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:11:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:11:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:13:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:13:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4817:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:12:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:12:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:14:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:14:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4813:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:13:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:13:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:15:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:15:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4814:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:14:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:14:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:16:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:16:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4824:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:15:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:15:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:17:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:17:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4825:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:16:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:16:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:18:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:18:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4826:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:17:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:17:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:19:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:19:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4819:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:18:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:18:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:20:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:20:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4837:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:19:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:19:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:21:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:21:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4836:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:20:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:20:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:22:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:22:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4838:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:21:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:21:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:23:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:23:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4693:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:22:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:22:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:24:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:24:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4692:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:23:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:23:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:25:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:25:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen ü. Miltitz</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4691:3:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:24:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:24:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:26:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:26:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4690:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:25:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:25:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:27:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:27:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4688:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:28:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:28:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:30:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:30:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4689:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:29:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:29:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:31:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:31:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4685:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:31:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:31:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:33:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:33:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4684:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:32:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:32:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:34:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:34:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4088:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:34:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:34:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:36:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:36:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
)";
constexpr auto const update_vgm418_1 = R"(
  <IstHalt>
    <HaltID>de:14627:4083:0:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:35:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:35:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:37:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:37:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4082:1:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:36:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:36:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:38:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:38:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4081:1:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:37:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:37:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:39:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:39:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4080:1:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:38:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:38:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:40:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:40:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4077:1:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:39:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:39:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:41:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:41:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4076:1:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:40:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:40:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:42:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:42:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4018:1:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:41:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:41:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:43:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:43:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4075:1:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:42:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:42:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:44:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:44:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4013:1:3</HaltID>
    <Abfahrtszeit>2024-08-26T16:44:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:44:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:46:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:46:07</IstAnkunftPrognose>
    <AbfahrtssteigText>3</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4055:3:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:45:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:45:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:47:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:47:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4048:1:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:46:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:46:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:48:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:48:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4010:1:2</HaltID>
    <Abfahrtszeit>2024-08-26T16:48:00</Abfahrtszeit>
    <Ankunftszeit>2024-08-26T16:48:00</Ankunftszeit>
    <IstAbfahrtPrognose>2024-08-26T16:50:07</IstAbfahrtPrognose>
    <IstAnkunftPrognose>2024-08-26T16:50:07</IstAnkunftPrognose>
    <AbfahrtssteigText>2</AbfahrtssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <IstHalt>
    <HaltID>de:14627:4007:1:1</HaltID>
    <Ankunftszeit>2024-08-26T16:50:00</Ankunftszeit>
    <IstAnkunftPrognose>2024-08-26T16:52:07</IstAnkunftPrognose>
    <AnkunftssteigText>1</AnkunftssteigText>
    <RichtungsText>Meißen Busbahnhof</RichtungsText>
    <Besetztgrad>Unbekannt</Besetztgrad>
  </IstHalt>
  <LinienText>418</LinienText>
  <ProduktID>VGM418</ProduktID>
  <RichtungsText>Meißen ü. Nossen - Miltitz</RichtungsText>
  <PrognoseMoeglich>true</PrognoseMoeglich>
  <FaelltAus>false</FaelltAus>
</IstFahrt>
)";

}  // namespace

TEST(vdv_update, monotonize) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / August / 1},
                    date::sys_days{2024_y / August / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, vgm418_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / August / 26});

  auto u = rt::vdv::updater{tt, src_idx};

  auto doc = pugi::xml_document{};
  doc.load_string(std::string{update_vgm418_0}.append(update_vgm418_1).c_str());
  u.update(rtt, doc);

  auto const fr = rt::frun{tt,
                           &rtt,
                           {{transport_idx_t{0U}, day_idx_t{30U}},
                            {stop_idx_t{0U}, stop_idx_t{51U}}}};

  EXPECT_TRUE(fr.is_rt());

  for (auto i = stop_idx_t{0}; i != fr.size() - 1; ++i) {
    auto const rs = fr[i];
    if (i != 0) {
      EXPECT_LE(rs.time(nigiri::event_type::kArr),
                rs.time(nigiri::event_type::kDep));
    }
    auto const rs_next = fr[i + 1];
    EXPECT_LE(rs.time(nigiri::event_type::kDep),
              rs_next.time(nigiri::event_type::kArr));
  }
}