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
"(
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

TEST(vdv_update, time_and_delay) {
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
