#include "gtest/gtest.h"

#include "pugixml.hpp"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/vdv_aus.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::rt;
using namespace date;
using namespace std::chrono_literals;

namespace {

mem_dir siri_test_files() {
  return mem_dir::read(R"__(
# agency.txt
"agency_id","agency_name","agency_url","agency_timezone","agency_lang","agency_phone"
10446,"DB Regio AG Bayern","https://www.delfi.de","Europe/Berlin","",""

# calendar.txt
"service_id","monday","tuesday","wednesday","thursday","friday","saturday","sunday","start_date","end_date"
9782,1,1,1,1,1,0,0,20250614,20251213

# calendar_dates.txt
"service_id","date","exception_type"
9782,20250804,2
9782,20250811,2
9782,20250818,2
9782,20250825,2
9782,20250901,2
9782,20250908,2
9782,20250805,2
9782,20250812,2
9782,20250819,2
9782,20250826,2
9782,20250902,2
9782,20250909,2
9782,20250806,2
9782,20250813,2
9782,20250820,2
9782,20250827,2
9782,20250903,2
9782,20250910,2
9782,20250619,2
9782,20250807,2
9782,20250814,2
9782,20250821,2
9782,20250828,2
9782,20250904,2
9782,20250911,2
9782,20250808,2
9782,20250815,2
9782,20250822,2
9782,20250829,2
9782,20250905,2
9782,20250912,2
9782,20251003,2

# stops.txt
"stop_id","stop_code","stop_name","stop_desc","stop_lat","stop_lon","location_type","parent_station","wheelchair_boarding","platform_code","level_id"
"de:09565:4410","","Schwabach-Limbach",,"49.338212000000","11.044580000000",1,,0,"",""
"de:09565:4411","","Schwabach",,"49.326200000000","11.035354000000",1,,0,"",""
"de:09574:7040","","Lauf West",,"49.501127000000","11.277998000000",1,,0,"",""
"de:09565:4410:2:2","","Schwabach-Limbach","Zug","49.338013000000","11.044490000000",0,"de:09565:4410",0,"1","2"
"de:09564:510","","Nürnberg Hbf",,"49.445615000000","11.082992000000",1,,0,"",""
"de:09564:1431","","Nürnberg-Mögeldorf",,"49.459210000000","11.133863000000",1,,0,"",""
"de:09564:1912","","Nürnberg-Reichelsdorf",,"49.382116000000","11.038858000000",1,,0,"",""
"de:09564:1913","","Nürnberg-Eibach",,"49.409606000000","11.047751000000",1,,0,"",""
"de:09564:427:3:N","","Nürnberg-Dürrenhof","Bus","49.448185000000","11.097356000000",0,,0,"1","2"
"de:09564:451:3:1","","Nürnberg-Ostring","Bus","49.454562000000","11.119679000000",0,,0,"1","2"
"de:09564:620:3:4","","Nürnberg-Steinbühl","Bus","49.442952000000","11.068304000000",0,,0,"2","1"
"de:09564:1912:2:2","","Nürnberg-Reichelsdorf","Zug","49.382221000000","11.038956000000",0,"de:09564:1912",0,"2","2"
"de:09574:7700:3:3","","Röthenbach (Pegnitz)","Bus","49.479928000000","11.230369000000",0,,0,"3","2"
"de:09574:7850:3:2","","Schwaig","Bus","49.471716000000","11.199126000000",0,,0,"2","2"
"de:09564:510:2:3","","Nürnberg Hbf","Gleis 2+3","49.445691000000","11.082156000000",0,"de:09564:510",0,"3","5"
"de:09564:1810:2:1","","Katzwang","Zug","49.355500000000","11.042577000000",0,"de:09564:1810",0,"1","5"
"de:09574:7000:3:2","","Lauf a.d.Pegnitz Bahnhof links der Pegnitz","Bus Eckertstr.","49.507130000000","11.285283000000",0,,0,"2","2"
"de:09574:7040:2:2","","Lauf West","Zug->LAU","49.501507000000","11.278160000000",0,"de:09574:7040",0,"2","2"
"de:09564:1913:2:2","","Nürnberg-Eibach","Zug->Nürnberg","49.409314000000","11.047625000000",0,"de:09564:1913",0,"2","5"
"de:09564:1412:2:2","","Nürnberg-Laufamholz","Zug->Lauf","49.462894000000","11.168413000000",0,"de:09564:1412",0,"2","2"
"de:09564:1431:2:2","","Nürnberg-Mögeldorf","Zug","49.459169000000","11.133989000000",0,"de:09564:1431",0,"2","2"
"de:09564:611:2:1","","Nürnberg-Sandreuth","Zug","49.429574000000","11.054803000000",0,"de:09564:611",0,"1","2"
"de:09564:1411:2:2","","Nürnberg-Rehhof","Zug->Lauf","49.462632000000","11.154614000000",0,"de:09564:1411",0,"2","2"
"de:09564:1911:2:1","","Reichelsdorfer Keller","Zug","49.368400000000","11.039971000000",0,"de:09564:1911",0,"1","5"
"de:09574:7720:2:2","","Röthenbach-Steinberg","Zug->Hartmannshof","49.483185000000","11.245928000000",0,"de:09574:7720",0,"2","2"
"de:09574:7710:2:2","","Röthenbach-Seespitze","Zug->Hartrmannsh.","49.491553000000","11.261613000000",0,"de:09574:7710",0,"2","5"
"de:09565:4411:56:6","","Schwabach","Gleis 5+6 (S-B)","49.326586000000","11.035660000000",0,"de:09565:4411",0,"6","2"
"de:09564:1810","","Katzwang",,"49.355254000000","11.042720000000",1,,0,"",""
"de:09564:1412","","Nürnberg-Laufamholz",,"49.462947000000","11.168475000000",1,,0,"",""
"de:09564:611","","Nürnberg-Sandreuth",,"49.429586000000","11.054650000000",1,,0,"",""
"de:09564:1911","","Reichelsdorfer Keller",,"49.368137000000","11.040214000000",1,,0,"",""
"de:09574:7710","","Röthenbach-Seespitze",,"49.491524000000","11.261487000000",1,,0,"",""
"de:09574:7720","","Röthenbach-Steinberg",,"49.483366000000","11.246162000000",1,,0,"",""
"de:09564:1411","","Nürnberg-Rehhof",,"49.462620000000","11.154758000000",1,,0,"",""
8005439,,,,"49.326200000000","11.035354000000"
8005440,,,,"49.338212000000","11.044580000000"
8003214,,,,"49.355254000000","11.042720000000"
8004994,,,,"49.368137000000","11.040214000000"
8004483,,,,"49.382221000000","11.038956000000"
8004477,,,,"49.409606000000","11.047751000000"
8004484,,,,"49.429574000000","11.054803000000"
8004487,,,,"49.442952000000","11.068304000000"
8000284,,,,"49.445691000000","11.082156000000"
8004442,,,,"49.448185000000","11.097356000000"
8004470,,,,"49.454562000000","11.119679000000"
8004481,,,,"49.459210000000","11.133863000000"
8004491,,,,"49.462620000000","11.154758000000"
8004480,,,,"49.462894000000","11.168413000000"
8005451,,,,"49.471716000000","11.199126000000"
8005140,,,,"49.479928000000","11.230369000000"
8005141,,,,"49.483185000000","11.245928000000"
8005142,,,,"49.491553000000","11.261613000000"
8003587,,,,"49.501127000000","11.277998000000"
8003580,,,,"49.507130000000","11.285283000000"

# routes.txt
"route_id","agency_id","route_short_name","route_long_name","route_type","route_color","route_text_color","route_desc"
"162465_109",10446,"S2","",109,"","",""

# trips.txt
"route_id","service_id","trip_id","trip_headsign","trip_short_name","direction_id","block_id","shape_id","wheelchair_accessible","bikes_allowed"
"162465_109",9782,2867631759,"Lauf a.d.Pegnitz Bahnhof links der Pegnitz","39611",0,,40080,0,0

# stop_times.txt
"trip_id","arrival_time","departure_time","stop_id","stop_sequence","pickup_type","drop_off_type","stop_headsign"
2867631759,10:39:00,10:39:00,"de:09565:4411:56:6",0,0,0,""
2867631759,10:41:00,10:41:00,"de:09565:4410:2:2",1,0,0,""
2867631759,10:43:00,10:44:00,"de:09564:1810:2:1",2,0,0,""
2867631759,10:45:00,10:46:00,"de:09564:1911:2:1",3,0,0,""
2867631759,10:48:00,10:48:00,"de:09564:1912:2:2",4,0,0,""
2867631759,10:50:00,10:51:00,"de:09564:1913:2:2",5,0,0,""
2867631759,10:53:00,10:53:00,"de:09564:611:2:1",6,0,0,""
2867631759,10:55:00,10:56:00,"de:09564:620:3:4",7,0,0,""
2867631759,10:58:00,11:03:00,"de:09564:510:2:3",8,0,0,""
2867631759,11:04:00,11:05:00,"de:09564:427:3:N",9,0,0,""
2867631759,11:07:00,11:07:00,"de:09564:451:3:1",10,0,0,""
2867631759,11:09:00,11:09:00,"de:09564:1431:2:2",11,0,0,""
2867631759,11:11:00,11:11:00,"de:09564:1411:2:2",12,0,0,""
2867631759,11:12:00,11:13:00,"de:09564:1412:2:2",13,0,0,""
2867631759,11:15:00,11:15:00,"de:09574:7850:3:2",14,0,0,""
2867631759,11:17:00,11:18:00,"de:09574:7700:3:3",15,0,0,""
2867631759,11:19:00,11:20:00,"de:09574:7720:2:2",16,0,0,""
2867631759,11:21:00,11:22:00,"de:09574:7710:2:2",17,0,0,""
2867631759,11:23:00,11:24:00,"de:09574:7040:2:2",18,0,0,""
2867631759,11:26:00,11:26:00,"de:09574:7000:3:2",19,0,0,""


)__");
}

constexpr auto const kMsg =
    R"(<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Siri xmlns:datex="http://datex2.eu/schema/2_0RC1/2_0" xmlns="http://www.siri.org.uk/siri" xmlns:acsb="http://www.ifopt.org.uk/acsb" xmlns:ifopt="http://www.ifopt.org.uk/ifopt" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xs="http://www.w3.org/2001/XMLSchema" version="2.0">
  <ServiceDelivery>
    <ResponseTimestamp>2025-07-03T08:39:50.880Z</ResponseTimestamp>
    <ProducerRef>rcsued-siri-spnv</ProducerRef>
    <RequestMessageRef>fa3138de-6e2f-435b-884c-92d029740a51</RequestMessageRef>
    <EstimatedTimetableDelivery version="2.0">
      <ResponseTimestamp>2025-07-03T08:39:50.880Z</ResponseTimestamp>
      <RequestMessageRef>fa3138de-6e2f-435b-884c-92d029740a51</RequestMessageRef>
      <EstimatedJourneyVersionFrame>
        <RecordedAtTime>2025-07-03T08:39:50.880Z</RecordedAtTime>
        <EstimatedVehicleJourney>
          <LineRef>2</LineRef>
          <DirectionRef>Schwabach#Lauf(links Pegnitz)</DirectionRef>
          <FramedVehicleJourneyRef>
            <DataFrameRef>2025-07-03</DataFrameRef>
            <DatedVehicleJourneyRef>39611-800721-8005439-103900_x35__x33_ADD_x33__x35_DB-s</DatedVehicleJourneyRef>
          </FramedVehicleJourneyRef>
          <PublishedLineName>S2</PublishedLineName>
          <DirectionName>Lauf(links Pegnitz)</DirectionName>
          <OriginRef>8005439</OriginRef>
          <OriginName>Schwabach</OriginName>
          <DestinationRef>8003580</DestinationRef>
          <OperatorRef>800721</OperatorRef>
          <ProductCategoryRef>S</ProductCategoryRef>
          <Monitored>true</Monitored>
          <PredictionInaccurate>false</PredictionInaccurate>
          <VehicleJourneyRef>39611-800721-8005439-103900#!ADD!#DB-s</VehicleJourneyRef>
          <RecordedCalls>
            <RecordedCall>
              <StopPointRef>8005439</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedDepartureTime>2025-07-03T10:39:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T10:39:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>6</DeparturePlatformName>
            </RecordedCall>
          </RecordedCalls>
          <EstimatedCalls>
            <EstimatedCall>
              <StopPointRef>8005440</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T10:41:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T10:40:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>1</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T10:41:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T10:41:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>1</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8003214</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T10:43:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T10:43:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>1</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T10:44:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T10:44:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>1</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8004994</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T10:45:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T10:45:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>1</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T10:46:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T10:46:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>1</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8004483</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T10:48:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T10:47:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T10:48:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T10:48:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8004477</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T10:50:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T10:50:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T10:51:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T10:51:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8004484</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T10:53:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T10:53:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>1</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T10:53:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T10:53:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>1</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8004487</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T10:55:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T10:55:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T10:56:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T10:56:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8000284</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T10:58:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T10:57:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>3</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:03:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:03:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>3</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8004442</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:04:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:05:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>1</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:05:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:05:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>1</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8004470</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:07:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:07:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>1</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:07:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:08:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>1</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8004481</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:09:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:09:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:09:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:09:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8004491</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:11:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:11:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:11:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:11:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8004480</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:12:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:13:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:13:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:13:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8005451</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:15:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:15:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:15:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:16:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8005140</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:17:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:18:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>3</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:18:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:18:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>3</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8005141</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:19:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:19:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:20:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:20:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8005142</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:21:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:21:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:22:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:22:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8003587</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:23:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:23:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2025-07-03T11:24:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2025-07-03T11:25:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>8003580</StopPointRef>
              <PredictionInaccurate>false</PredictionInaccurate>
              <AimedArrivalTime>2025-07-03T11:26:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2025-07-03T11:26:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
            </EstimatedCall>
          </EstimatedCalls>
          <IsCompleteStopSequence>true</IsCompleteStopSequence>
        </EstimatedVehicleJourney>
      </EstimatedJourneyVersionFrame>
    </EstimatedTimetableDelivery>
  </ServiceDelivery>
</Siri>)";

}  // namespace

TEST(siri_update, matching) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2025_y / July / 1},
                    date::sys_days{2025_y / July / 31}};
  auto const src_idx = source_idx_t{0};
  load_timetable({}, src_idx, siri_test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2025_y / July / 3});

  auto u = rt::vdv_aus::updater{tt, src_idx,
                                rt::vdv_aus::updater::xml_format::kSiri};

  auto doc = pugi::xml_document{};
  doc.load_string(kMsg);
  u.update(rtt, doc);

  auto td = transit_realtime::TripDescriptor{};
  td.set_trip_id("2867631759");
  td.set_start_date("20250703");
  td.set_start_time("10:39:00");
  auto const [r, trip] =
      gtfsrt_resolve_run(date::sys_days{2025_y / July / 3}, tt, &rtt, {}, td);
  ASSERT_TRUE(r.valid());
  ASSERT_TRUE(r.is_rt());

  constexpr auto const kExpected =
      R"(   0: de:09565:4411:56:6 Schwabach.......................................                                                             d: 03.07 08:39 [03.07 10:39]  RT 03.07 08:39 [03.07 10:39]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
   1: de:09565:4410:2:2 Schwabach-Limbach............................... a: 03.07 08:41 [03.07 10:41]  RT 03.07 08:40 [03.07 10:40]  d: 03.07 08:41 [03.07 10:41]  RT 03.07 08:41 [03.07 10:41]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
   2: de:09564:1810:2:1 Katzwang........................................ a: 03.07 08:43 [03.07 10:43]  RT 03.07 08:43 [03.07 10:43]  d: 03.07 08:44 [03.07 10:44]  RT 03.07 08:44 [03.07 10:44]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
   3: de:09564:1911:2:1 Reichelsdorfer Keller........................... a: 03.07 08:45 [03.07 10:45]  RT 03.07 08:45 [03.07 10:45]  d: 03.07 08:46 [03.07 10:46]  RT 03.07 08:46 [03.07 10:46]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
   4: de:09564:1912:2:2 Nürnberg-Reichelsdorf........................... a: 03.07 08:48 [03.07 10:48]  RT 03.07 08:47 [03.07 10:47]  d: 03.07 08:48 [03.07 10:48]  RT 03.07 08:48 [03.07 10:48]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
   5: de:09564:1913:2:2 Nürnberg-Eibach................................. a: 03.07 08:50 [03.07 10:50]  RT 03.07 08:50 [03.07 10:50]  d: 03.07 08:51 [03.07 10:51]  RT 03.07 08:51 [03.07 10:51]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
   6: de:09564:611:2:1 Nürnberg-Sandreuth.............................. a: 03.07 08:53 [03.07 10:53]  RT 03.07 08:53 [03.07 10:53]  d: 03.07 08:53 [03.07 10:53]  RT 03.07 08:53 [03.07 10:53]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
   7: de:09564:620:3:4 Nürnberg-Steinbühl.............................. a: 03.07 08:55 [03.07 10:55]  RT 03.07 08:55 [03.07 10:55]  d: 03.07 08:56 [03.07 10:56]  RT 03.07 08:56 [03.07 10:56]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
   8: de:09564:510:2:3 Nürnberg Hbf.................................... a: 03.07 08:58 [03.07 10:58]  RT 03.07 08:57 [03.07 10:57]  d: 03.07 09:03 [03.07 11:03]  RT 03.07 09:03 [03.07 11:03]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
   9: de:09564:427:3:N Nürnberg-Dürrenhof.............................. a: 03.07 09:04 [03.07 11:04]  RT 03.07 09:05 [03.07 11:05]  d: 03.07 09:05 [03.07 11:05]  RT 03.07 09:05 [03.07 11:05]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
  10: de:09564:451:3:1 Nürnberg-Ostring................................ a: 03.07 09:07 [03.07 11:07]  RT 03.07 09:07 [03.07 11:07]  d: 03.07 09:07 [03.07 11:07]  RT 03.07 09:08 [03.07 11:08]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
  11: de:09564:1431:2:2 Nürnberg-Mögeldorf.............................. a: 03.07 09:09 [03.07 11:09]  RT 03.07 09:09 [03.07 11:09]  d: 03.07 09:09 [03.07 11:09]  RT 03.07 09:09 [03.07 11:09]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
  12: de:09564:1411:2:2 Nürnberg-Rehhof................................. a: 03.07 09:11 [03.07 11:11]  RT 03.07 09:11 [03.07 11:11]  d: 03.07 09:11 [03.07 11:11]  RT 03.07 09:11 [03.07 11:11]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
  13: de:09564:1412:2:2 Nürnberg-Laufamholz............................. a: 03.07 09:12 [03.07 11:12]  RT 03.07 09:13 [03.07 11:13]  d: 03.07 09:13 [03.07 11:13]  RT 03.07 09:13 [03.07 11:13]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
  14: de:09574:7850:3:2 Schwaig......................................... a: 03.07 09:15 [03.07 11:15]  RT 03.07 09:15 [03.07 11:15]  d: 03.07 09:15 [03.07 11:15]  RT 03.07 09:16 [03.07 11:16]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
  15: de:09574:7700:3:3 Röthenbach (Pegnitz)............................ a: 03.07 09:17 [03.07 11:17]  RT 03.07 09:18 [03.07 11:18]  d: 03.07 09:18 [03.07 11:18]  RT 03.07 09:18 [03.07 11:18]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
  16: de:09574:7720:2:2 Röthenbach-Steinberg............................ a: 03.07 09:19 [03.07 11:19]  RT 03.07 09:19 [03.07 11:19]  d: 03.07 09:20 [03.07 11:20]  RT 03.07 09:20 [03.07 11:20]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
  17: de:09574:7710:2:2 Röthenbach-Seespitze............................ a: 03.07 09:21 [03.07 11:21]  RT 03.07 09:21 [03.07 11:21]  d: 03.07 09:22 [03.07 11:22]  RT 03.07 09:22 [03.07 11:22]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
  18: de:09574:7040:2:2 Lauf West....................................... a: 03.07 09:23 [03.07 11:23]  RT 03.07 09:23 [03.07 11:23]  d: 03.07 09:24 [03.07 11:24]  RT 03.07 09:25 [03.07 11:25]  [{name=S2, day=2025-07-03, id=2867631759, src=0}]
  19: de:09574:7000:3:2 Lauf a.d.Pegnitz Bahnhof links der Pegnitz...... a: 03.07 09:26 [03.07 11:26]  RT 03.07 09:26 [03.07 11:26]
)";
  EXPECT_EQ(kExpected, (std::stringstream{} << rt::frun{tt, &rtt, r}).str());
}
