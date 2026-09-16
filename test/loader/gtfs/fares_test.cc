#include "gtest/gtest.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "../../raptor_search.h"
#include "../../routing/results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

constexpr auto const kBasicTimetable = R"(
# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
weekday,1,1,1,1,1,0,0,20220101,20221231
weekend,0,0,0,0,0,1,1,20220101,20221231
everyday,1,1,1,1,1,1,1,20220101,20221231

# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,0.02,1.03,,
C,C,,0.04,1.05,,
D,D,,0.06,1.07,,
E,E,,0.08,1.09,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
line1,DB,Line 1,,,3
line2,DB,Line 2,,,3
line3,DB,Line 3,,,3
air,DB,Line 4,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
line1,everyday,T1,T1,
line2,everyday,T2,T2,
line3,everyday,T3,T3,
air,everyday,T4,T4,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,00:00:00,00:00:00,A,1,0,0
T1,00:30:00,00:30:00,B,2,0,0
T2,00:35:00,00:35:00,B,1,0,0
T2,01:00:00,01:00:00,C,2,0,0
T3,01:05:00,01:05:00,C,1,0,0
T3,02:00:00,02:00:00,D,2,0,0
T4,02:05:00,02:05:00,D,1,0,0
T4,02:30:00,02:30:00,E,2,0,0

# frequencies.txt
trip_id,start_time,end_time,headway_secs
T1,00:00:00,24:00:00,3600
T2,00:35:00,24:35:00,3600
T3,01:05:00,25:05:00,3600
T4,02:05:00,26:05:00,3600
)";

constexpr auto const kBasicFares = R"(
# timeframes.txt
timeframe_group_id,start_time,end_time,service_id
offpeak,00:00:00,05:00:00,weekday
peak,5:00:00,9:30:00,weekday
offpeak,9:30:00,15:00:00,weekday
peak,15:00:00,19:00:00,weekday
offpeak,19:00:00,24:00:00,weekday
offpeak,,,weekend

# rider_categories.txt
rider_category_id,rider_category_name,is_default_fare_category,eligibility_url
adult,Adult,1,
reduced,Students and seniors,0,

# fare_media.txt
fare_media_id,fare_media_name,fare_media_type
paper-card,Paper Card,1
app,App,4
)";

constexpr auto const kAreaSets = R"(
# fare_media.txt
fare_media_id,fare_media_name,fare_media_type
HSL_ticket,HSL Ticket,1
HSL_card,HSL Card,2
HSL_app,HSL App,4
contactless,Contactless CEMV,3

# rider_categories.txt
rider_category_id,rider_category_name,is_default_fare_category,eligibility_url
adult,Adult,1,
seventy_plus,70+,,
student,Student,,
pensioner,Pensioner with Kela,,"https://www.hsl.fi/en/tickets-and-fares/discounted-travel#:~:text=Read%20more-,Pensioners%20(Kela),-If%20you%20receive"
reduced_mobility,Reduced Mobility,,https://www.hsl.fi/en/tickets-and-fares/discounted-travel/people-with-reduced-mobility
children,Children 7 to 17,,https://www.hsl.fi/en/tickets-and-fares/discounted-travel

# fare_products.txt
fare_product_id,fare_product_name,fare_media_id,rider_category_id,currency,amount
one_two_zone_single_ticket,One-Two-zone Ticket,HSL_ticket,,EUR,3.20
three_zone_single_ticket,Three Zone Ticket,HSL_ticket,,EUR,4.40
four_zone_single_ticket,Four Zone Ticket,HSL_ticket,,EUR,4.80
one_two_zone_single_ticket,One-Two-zone Ticket,HSL_card,,EUR,3.20
three_zone_single_ticket,Three Zone Ticket,HSL_card,,EUR,4.40
four_zone_single_ticket,Four Zone Ticket,HSL_card,,EUR,4.80
one_two_zone_single_ticket,One-Two-zone Ticket,HSL_app,,EUR,3.20
three_zone_single_ticket,Three Zone Ticket,HSL_app,,EUR,4.40
four_zone_single_ticket,Four Zone Ticket,HSL_app,,EUR,4.80
one_two_zone_single_ticket,One-Two-zone Contactless,contactless,,EUR,3.40
three_zone_single_ticket,Three Zone Contactless,contactless,,EUR,4.70
four_zone_single_ticket,Four Zone Contactless,contactless,,EUR,5.10
one_two_zone_single_ticket,Pensioner One-Two-zone Ticket,HSL_card,pensioner,EUR,1.60
three_zone_single_ticket,Pensioner Three Zone Ticket,HSL_card,pensioner,EUR,2.20
four_zone_single_ticket,Pensioner Four Zone Ticket,HSL_card,pensioner,EUR,2.40
one_two_zone_single_ticket,Reduced Mobility One-Two-zone Ticket,HSL_card,reduced_mobility,EUR,1.60
three_zone_single_ticket,Reduced Mobility Three Zone Ticket,HSL_card,reduced_mobility,EUR,2.20
four_zone_single_ticket,Reduced Mobility Four Zone Ticket,HSL_card,reduced_mobility,EUR,2.40
one_two_zone_single_ticket,Children One-Two-zone Ticket,HSL_card,children,EUR,1.60
three_zone_single_ticket,Children Three Zone Ticket,HSL_card,children,EUR,2.20
four_zone_single_ticket,Children Four Zone Ticket,HSL_card,children,EUR,2.40
one_two_zone_single_ticket,Children One-Two-zone Ticket,HSL_app,children,EUR,1.60
three_zone_single_ticket,Children Three Zone Ticket,HSL_app,children,EUR,2.20
four_zone_single_ticket,Children Four Zone Ticket,HSL_app,children,EUR,2.40

# areas.txt
area_id,area_name
A,A
B,B
C,C
D,D

# stop_areas.txt
area_id,stop_id
A,A
B,B
C,C
D,D

# area_set_elements.txt
area_set_id,area_id
A_set,A
B_set,B
C_set,C
D_set,D
AB_set,A
AB_set,B
BC_set,B
BC_set,C
CD_set,C
CD_set,D
ABC_set,A
ABC_set,B
ABC_set,C
BCD_set,B
BCD_set,C
BCD_set,D
ABCD_set,A
ABCD_set,B
ABCD_set,C
ABCD_set,D

# fare_leg_rules.txt
leg_group_id,network_id,from_area_id,to_area_id,fare_product_id,contains_exactly_area_set_id
A_leg,HSL,,,one_two_zone_single_ticket,A_set
B_leg,HSL,,,one_two_zone_single_ticket,B_set
C_leg,HSL,,,one_two_zone_single_ticket,C_set
D_leg,HSL,,,one_two_zone_single_ticket,D_set
AB_leg,HSL,,,one_two_zone_single_ticket,AB_set
BC_leg,HSL,,,one_two_zone_single_ticket,BC_set
CD_leg,HSL,,,one_two_zone_single_ticket,CD_set
ABC_leg,HSL,,,three_zone_single_ticket,ABC_set
BCD_leg,HSL,,,three_zone_single_ticket,BCD_set
ABCD_leg,HSL,,,four_zone_single_ticket,ABCD_set

# networks.txt
network_id,network_name
HSL,Helsinki Regional Transport Authority

# fare_leg_join_rules.txt
from_network_id	,to_network_id,from_stop_id	,to_stop_id
HSL,HSL,,

# route_networks.txt
network_id,route_id
HSL,line1
HSL,line2
HSL,line3
HSL,air
)";

constexpr auto const kFareTimetable1 = R"(
# fare_products.txt
fare_product_id,fare_product_name,fare_media_id,amount,currency,rider_category_id
offpeak-pink-oneway-paper-card,Pink Paper Card,paper-card,3.00,EUR,adult
offpeak-pink-oneway-app,Pink App,app,2.50,EUR,adult
offpeak-pink-day-app,Pink Daypass,app,4.00,EUR,adult
offpeak-pink-week-app,Pink Weekpass,app,10.00,EUR,adult
offpeak-blue-oneway-paper-card,Blue Paper Card,paper-card,3.00,EUR,adult
offpeak-blue-oneway-app,Blue App,app,2.50,EUR,adult
offpeak-blue-day-app,Blue Daypass,app,4.00,EUR,adult
offpeak-blue-week-app,Blue Weekpass,app,10.00,EUR,adult
offpeak-full-oneway-paper-card,Full Paper Card,paper-card,4.00,EUR,adult
offpeak-full-oneway-app,Full App,app,3.50,EUR,adult
offpeak-full-day-app,Full Daypass,app,5.00,EUR,adult
offpeak-full-week-app,Full Weekpass,app,12.00,EUR,adult
peak-pink-oneway-paper-card,Peak Pink Paper Card,paper-card,6.00,EUR,adult
peak-pink-oneway-app,Peak Pink App,app,5.00,EUR,adult
peak-pink-day-app,Peak Pink Daypass,app,8.00,EUR,adult
peak-pink-week-app,Peak Pink Weekpass,app,20.00,EUR,adult
peak-blue-oneway-paper-card,Peak Blue Paper Card,paper-card,6.00,EUR,adult
peak-blue-oneway-app,Peak Blue App,app,5.00,EUR,adult
peak-blue-day-app,Peak Blue Daypass,app,8.00,EUR,adult
peak-blue-week-app,Peak Blue Weekpass,app,20.00,EUR,adult
peak-full-oneway-paper-card,Peak Full Paper Card,paper-card,8.00,EUR,adult
peak-full-oneway-app,Peak Full App,app,7.00,EUR,adult
peak-full-day-app,Peak Full Daypass,app,10.00,EUR,adult
peak-full-week-app,Peak Full Weekpass,app,24.00,EUR,adult
airport-card,Airport Card,paper-card,10.00,EUR,adult
peak-full-airport-ext,Full Airport Extension Card,paper-card,6.00,EUR,adult
reduced-offpeak-pink-oneway-paper-card,Pink Paper Card,paper-card,2.00,EUR,reduced
reduced-offpeak-pink-oneway-app,Pink App,app,1.50,EUR,reduced
reduced-offpeak-pink-day-app,Pink Daypass,app,2.50,EUR,reduced
reduced-offpeak-pink-week-app,Pink Weekpass,app,6.00,EUR,reduced
reduced-offpeak-blue-oneway-paper-card,Blue Paper Card,paper-card,2.00,EUR,reduced
reduced-offpeak-blue-oneway-app,Blue App,app,1.50,EUR,reduced
reduced-offpeak-blue-day-app,Blue Daypass,app,2.50,EUR,reduced
reduced-offpeak-blue-week-app,Blue Weekpass,app,6.00,EUR,reduced
reduced-offpeak-full-oneway-paper-card,Full Paper Card,paper-card,3.00,EUR,reduced
reduced-offpeak-full-oneway-app,Full App,app,2.50,EUR,reduced
reduced-offpeak-full-day-app,Full Daypass,app,3.50,EUR,reduced
reduced-offpeak-full-week-app,Full Weekpass,app,8.00,EUR,reduced
reduced-peak-pink-oneway-paper-card,Peak Pink Paper Card,paper-card,4.00,EUR,reduced
reduced-peak-pink-oneway-app,Peak Pink App,app,3.50,EUR,reduced
reduced-peak-pink-day-app,Peak Pink Daypass,app,5.00,EUR,reduced
reduced-peak-pink-week-app,Peak Pink Weekpass,app,12.00,EUR,reduced
reduced-peak-blue-oneway-paper-card,Peak Blue Paper Card,paper-card,4.00,EUR,reduced
reduced-peak-blue-oneway-app,Peak Blue App,app,3.50,EUR,reduced
reduced-peak-blue-day-app,Peak Blue Daypass,app,5.00,EUR,reduced
reduced-peak-blue-week-app,Peak Blue Weekpass,app,12.00,EUR,reduced
reduced-peak-full-oneway-paper-card,Peak Full Paper Card,paper-card,6.00,EUR,reduced
reduced-peak-full-oneway-app,Peak Full App,app,5.50,EUR,reduced
reduced-peak-full-day-app,Peak Full Daypass,app,7.00,EUR,reduced
reduced-peak-full-week-app,Peak Full Weekpass,app,16.00,EUR,reduced

# areas.txt
area_id,area_name
1,Area 1
2,Area 2
3,Area Air

# stop_areas.txt
area_id,stop_id
1,A
1,B
1,C
2,C
2,D
3,E

# networks.txt
network_id,network_name
blue,Blue Network
pink,Pink Network
air,Airport Shuttle

# fare_leg_join_rules.txt
from_network_id,to_network_id,from_stop_id,to_stop_id
pink,pink,,
pink,blue,,

# fare_leg_rules.txt
leg_group_id,network_id,from_area_id,to_area_id,from_timeframe_group_id,fare_product_id,rule_priority
core,,1,1,offpeak,offpeak-pink-oneway-paper-card,1
core,,1,1,offpeak,offpeak-pink-oneway-app,1
core,,1,1,offpeak,offpeak-pink-day-app,1
core,,1,1,offpeak,offpeak-pink-week-app,1
core,,1,1,peak,peak-pink-oneway-paper-card,1
core,,1,1,peak,peak-pink-oneway-app,1
core,,1,1,peak,peak-pink-day-app,1
core,,1,1,peak,peak-pink-week-app,1
core,,2,2,offpeak,offpeak-blue-oneway-paper-card,1
core,,2,2,offpeak,offpeak-blue-oneway-app,1
core,,2,2,offpeak,offpeak-blue-day-app,1
core,,2,2,offpeak,offpeak-blue-week-app,1
core,,2,2,peak,peak-blue-oneway-paper-card,1
core,,2,2,peak,peak-blue-oneway-app,1
core,,2,2,peak,peak-blue-day-app,1
core,,2,2,peak,peak-blue-week-app,1
core,,1,2,offpeak,offpeak-full-oneway-paper-card,
core,,1,2,offpeak,offpeak-full-oneway-app,
core,,1,2,offpeak,offpeak-full-day-app,
core,,1,2,offpeak,offpeak-full-week-app,
core,,1,2,peak,peak-full-oneway-paper-card,
core,,1,2,peak,peak-full-oneway-app,
core,,1,2,peak,peak-full-day-app,
core,,1,2,peak,peak-full-week-app,
core,,1,1,offpeak,reduced-offpeak-pink-oneway-paper-card,1
core,,1,1,offpeak,reduced-offpeak-pink-oneway-app,1
core,,1,1,offpeak,reduced-offpeak-pink-day-app,1
core,,1,1,offpeak,reduced-offpeak-pink-week-app,1
core,,1,1,peak,reduced-peak-pink-oneway-paper-card,1
core,,1,1,peak,reduced-peak-pink-oneway-app,1
core,,1,1,peak,reduced-peak-pink-day-app,1
core,,1,1,peak,reduced-peak-pink-week-app,1
core,,2,2,offpeak,reduced-offpeak-blue-oneway-paper-card,1
core,,2,2,offpeak,reduced-offpeak-blue-oneway-app,1
core,,2,2,offpeak,reduced-offpeak-blue-day-app,1
core,,2,2,offpeak,reduced-offpeak-blue-week-app,1
core,,2,2,peak,reduced-peak-blue-oneway-paper-card,1
core,,2,2,peak,reduced-peak-blue-oneway-app,1
core,,2,2,peak,reduced-peak-blue-day-app,1
core,,2,2,peak,reduced-peak-blue-week-app,1
core,,1,2,offpeak,reduced-offpeak-full-oneway-paper-card,
core,,1,2,offpeak,reduced-offpeak-full-oneway-app,
core,,1,2,offpeak,reduced-offpeak-full-day-app,
core,,1,2,offpeak,reduced-offpeak-full-week-app,
core,,1,2,peak,reduced-peak-full-oneway-paper-card,
core,,1,2,peak,reduced-peak-full-oneway-app,
core,,1,2,peak,reduced-peak-full-day-app,
core,,1,2,peak,reduced-peak-full-week-app,
air,air,2,3,,airport-card,

# fare_transfer_rules.txt
from_leg_group_id,to_leg_group_id,transfer_count,duration_limit,duration_limit_type,fare_transfer_type,fare_product_id
core,air,1,10800,1,0,peak-full-airport-ext

# route_networks.txt
network_id,route_id
pink,line1
pink,line2
blue,line3
air,air
)";

constexpr auto const kTimetableWithoutFares = R"(
# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
weekday,1,1,1,1,1,0,0,20220101,20221231
weekend,0,0,0,0,0,1,1,20220101,20221231
everyday,1,1,1,1,1,1,1,20220101,20221231

# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
E,E,,8.0,9.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
line1,DB,Line 1,,,3
line2,DB,Line 2,,,3
line3,DB,Line 3,,,3
air,DB,Line 4,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
line1,everyday,T1,T1,
line2,everyday,T2,T2,
line3,everyday,T3,T3,
air,everyday,T4,T4,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,00:00:00,00:00:00,A,1,0,0
T1,00:30:00,00:30:00,B,2,0,0
T2,00:35:00,00:35:00,B,1,0,0
T2,01:00:00,01:00:00,C,2,0,0
T3,01:05:00,01:05:00,C,1,0,0
T3,02:00:00,02:00:00,D,2,0,0
T4,02:05:00,02:05:00,D,1,0,0
T4,02:30:00,02:30:00,E,2,0,0

# frequencies.txt
trip_id,start_time,end_time,headway_secs
T1,00:00:00,24:00:00,3600
T2,00:35:00,24:35:00,3600
T3,01:05:00,25:05:00,3600
T4,02:05:00,26:05:00,3600
)";

}  // namespace

std::string to_string(timetable const& tt,
                      rt_timetable const* rtt,
                      std::vector<fare_transfer> const& fare_transfers,
                      bool const main_only = true) {
  auto ss = std::stringstream{};

  for (auto const& transfer : fare_transfers) {
    if (main_only && !transfer.main_) {
      continue;
    }
    if (transfer.rule_.has_value()) {
      ss << "FARE TRANSFER START\n";
      auto const f = tt.fares_[transfer.legs_.front().src_];
      auto const product =
          transfer.rule_->fare_product_ == fare_product_idx_t::invalid()
              ? ""
              : tt.strings_.get(f.fare_products_[transfer.rule_->fare_product_]
                                    .front()
                                    .name_);
      ss << "TRANSFER PRODUCT: " << product << "\n";
      ss << "RULE: " << transfer.rule_->fare_transfer_type_ << "\n";
    }
    for (auto const& l : transfer.legs_) {
      ss << "FARE LEG:\n";
      auto first = true;
      for (auto const& jl : l.joined_leg_) {
        if (first) {
          first = false;
        } else {
          ss << "** JOINED WITH\n";
        }
        jl->print(ss, tt, rtt);
      }
      ss << "PRODUCTS\n";
      for (auto const& r : l.rule_) {
        auto const& f = tt.fares_[l.src_];
        auto const products = f.fare_products_[r.fare_product_];
        for (auto const& p : products) {
          auto const id = f.fare_product_id_[r.fare_product_];
          auto const& m = f.fare_media_[p.media_];
          if (products.size() != 1U) {
            ss << "id=" << tt.strings_.get(id) << ", name=";
          }
          ss << tt.strings_.get(p.name_) << " [priority=" << r.rule_priority_
             << "]: " << p.amount_ << " " << tt.strings_.get(p.currency_code_)
             << ", fare_media_name=" << tt.strings_.get(m.name_)
             << ", fare_type=" << m.type_ << ", ride_category="
             << (p.rider_category_ == rider_category_idx_t::invalid()
                     ? "??"
                     : tt.strings_.get(
                           f.rider_categories_[p.rider_category_].name_))
             << "\n";
        }
      }
      ss << "\n\n";
    }
    if (transfer.rule_.has_value()) {
      ss << "FARE TRANSFER END\n";
    }
  }
  return ss.str();
}

TEST(fares, area_sets) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2022_y / January / 1},
                    date::sys_days{2022_y / December / 1}};
  load_timetable({}, source_idx_t{0},
                 mem_dir::read(fmt::format("{}{}", kBasicTimetable, kAreaSets)),
                 tt);
  finalize(tt);

  auto const results = raptor_search(tt, nullptr, "A", "C",
                                     unixtime_t{sys_days{2022_y / March / 30}});
  ASSERT_EQ(1U, results.size());
  auto const fare_legs = get_fares(tt, nullptr, *results.begin());

  constexpr auto const kExpected = R"(FARE LEG:
   0: A       A...............................................                               d: 30.03 00:00 [30.03 02:00]  [{name=Line 1, day=2022-03-30, id=T1, src=0}]
   1: B       B............................................... a: 30.03 00:30 [30.03 02:30]
** JOINED WITH
   0: B       B...............................................                               d: 30.03 00:35 [30.03 02:35]  [{name=Line 2, day=2022-03-30, id=T2, src=0}]
   1: C       C............................................... a: 30.03 01:00 [30.03 03:00]
PRODUCTS
id=three_zone_single_ticket, name=Three Zone Ticket [priority=0]: 4.4 EUR, fare_media_name=HSL Ticket, fare_type=PAPER, ride_category=??
id=three_zone_single_ticket, name=Three Zone Ticket [priority=0]: 4.4 EUR, fare_media_name=HSL Card, fare_type=CARD, ride_category=??
id=three_zone_single_ticket, name=Three Zone Ticket [priority=0]: 4.4 EUR, fare_media_name=HSL App, fare_type=APP, ride_category=??
id=three_zone_single_ticket, name=Three Zone Contactless [priority=0]: 4.7 EUR, fare_media_name=Contactless CEMV, fare_type=CONTACTLESS, ride_category=??
id=three_zone_single_ticket, name=Pensioner Three Zone Ticket [priority=0]: 2.2 EUR, fare_media_name=HSL Card, fare_type=CARD, ride_category=Pensioner with Kela
id=three_zone_single_ticket, name=Reduced Mobility Three Zone Ticket [priority=0]: 2.2 EUR, fare_media_name=HSL Card, fare_type=CARD, ride_category=Reduced Mobility
id=three_zone_single_ticket, name=Children Three Zone Ticket [priority=0]: 2.2 EUR, fare_media_name=HSL Card, fare_type=CARD, ride_category=Children 7 to 17
id=three_zone_single_ticket, name=Children Three Zone Ticket [priority=0]: 2.2 EUR, fare_media_name=HSL App, fare_type=APP, ride_category=Children 7 to 17


)";
  EXPECT_EQ(kExpected, to_string(tt, nullptr, fare_legs));
}

TEST(fares, simple_fares) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2022_y / January / 1},
                    date::sys_days{2022_y / December / 1}};
  load_timetable({}, source_idx_t{0},
                 mem_dir::read(fmt::format("{}{}{}", kBasicTimetable,
                                           kBasicFares, kFareTimetable1)),
                 tt);
  finalize(tt);

  {  // OFFPEAK TEST
    auto const results = raptor_search(
        tt, nullptr, "A", "E", unixtime_t{sys_days{2022_y / March / 30}});
    ASSERT_EQ(1U, results.size());
    auto const fare_legs = get_fares(tt, nullptr, *results.begin());
    constexpr auto const kExpected = R"(FARE TRANSFER START
TRANSFER PRODUCT: Full Airport Extension Card
RULE: A+AB
FARE LEG:
   0: A       A...............................................                               d: 30.03 00:00 [30.03 02:00]  [{name=Line 1, day=2022-03-30, id=T1, src=0}]
   1: B       B............................................... a: 30.03 00:30 [30.03 02:30]
** JOINED WITH
   0: B       B...............................................                               d: 30.03 00:35 [30.03 02:35]  [{name=Line 2, day=2022-03-30, id=T2, src=0}]
   1: C       C............................................... a: 30.03 01:00 [30.03 03:00]
** JOINED WITH
   0: C       C...............................................                               d: 30.03 01:05 [30.03 03:05]  [{name=Line 3, day=2022-03-30, id=T3, src=0}]
   1: D       D............................................... a: 30.03 02:00 [30.03 04:00]
PRODUCTS
Full App [priority=0]: 3.5 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Full Paper Card [priority=0]: 4 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Adult
Full Daypass [priority=0]: 5 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Full Weekpass [priority=0]: 12 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Full App [priority=0]: 2.5 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors
Full Paper Card [priority=0]: 3 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Students and seniors
Full Daypass [priority=0]: 3.5 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors
Full Weekpass [priority=0]: 8 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors


FARE LEG:
   0: D       D...............................................                               d: 30.03 02:05 [30.03 04:05]  [{name=Line 4, day=2022-03-30, id=T4, src=0}]
   1: E       E............................................... a: 30.03 02:30 [30.03 04:30]
PRODUCTS
Airport Card [priority=0]: 10 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Adult


FARE TRANSFER END
)";
    EXPECT_EQ(kExpected, to_string(tt, nullptr, fare_legs));
  }

  {  // PEAK TEST
    auto const results = raptor_search(
        tt, nullptr, "A", "E", unixtime_t{sys_days{2022_y / March / 30} + 6h});
    ASSERT_EQ(1U, results.size());
    auto const fare_legs = get_fares(tt, nullptr, *results.begin());
    constexpr auto const kExpected = R"(FARE TRANSFER START
TRANSFER PRODUCT: Full Airport Extension Card
RULE: A+AB
FARE LEG:
   0: A       A...............................................                               d: 30.03 06:00 [30.03 08:00]  [{name=Line 1, day=2022-03-30, id=T1, src=0}]
   1: B       B............................................... a: 30.03 06:30 [30.03 08:30]
** JOINED WITH
   0: B       B...............................................                               d: 30.03 06:35 [30.03 08:35]  [{name=Line 2, day=2022-03-30, id=T2, src=0}]
   1: C       C............................................... a: 30.03 07:00 [30.03 09:00]
** JOINED WITH
   0: C       C...............................................                               d: 30.03 07:05 [30.03 09:05]  [{name=Line 3, day=2022-03-30, id=T3, src=0}]
   1: D       D............................................... a: 30.03 08:00 [30.03 10:00]
PRODUCTS
Peak Full App [priority=0]: 7 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Peak Full Paper Card [priority=0]: 8 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Adult
Peak Full Daypass [priority=0]: 10 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Peak Full Weekpass [priority=0]: 24 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Peak Full App [priority=0]: 5.5 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors
Peak Full Paper Card [priority=0]: 6 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Students and seniors
Peak Full Daypass [priority=0]: 7 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors
Peak Full Weekpass [priority=0]: 16 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors


FARE LEG:
   0: D       D...............................................                               d: 30.03 08:05 [30.03 10:05]  [{name=Line 4, day=2022-03-30, id=T4, src=0}]
   1: E       E............................................... a: 30.03 08:30 [30.03 10:30]
PRODUCTS
Airport Card [priority=0]: 10 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Adult


FARE TRANSFER END
)";
    EXPECT_EQ(kExpected, to_string(tt, nullptr, fare_legs));
  }

  {  // OFFPEAK TEST
    auto const results = raptor_search(
        tt, nullptr, "C", "E", unixtime_t{sys_days{2022_y / March / 30}});
    ASSERT_EQ(1U, results.size());
    auto const fare_legs = get_fares(tt, nullptr, *results.begin());
    constexpr auto const kExpected = R"(FARE TRANSFER START
TRANSFER PRODUCT: Full Airport Extension Card
RULE: A+AB
FARE LEG:
   0: C       C...............................................                               d: 30.03 00:05 [30.03 02:05]  [{name=Line 3, day=2022-03-30, id=T3, src=0}]
   1: D       D............................................... a: 30.03 01:00 [30.03 03:00]
PRODUCTS
Blue App [priority=1]: 2.5 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Blue Paper Card [priority=1]: 3 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Adult
Blue Daypass [priority=1]: 4 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Blue Weekpass [priority=1]: 10 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Blue App [priority=1]: 1.5 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors
Blue Paper Card [priority=1]: 2 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Students and seniors
Blue Daypass [priority=1]: 2.5 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors
Blue Weekpass [priority=1]: 6 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors


FARE LEG:
   0: D       D...............................................                               d: 30.03 01:05 [30.03 03:05]  [{name=Line 4, day=2022-03-30, id=T4, src=0}]
   1: E       E............................................... a: 30.03 01:30 [30.03 03:30]
PRODUCTS
Airport Card [priority=0]: 10 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Adult


FARE TRANSFER END
)";
    EXPECT_EQ(kExpected, to_string(tt, nullptr, fare_legs));
  }
}

TEST(fares, rt_added_fares) {
  auto const to_unix = [](auto&& x) {
    return std::chrono::time_point_cast<std::chrono::seconds>(x)
        .time_since_epoch()
        .count();
  };

  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2022_y / January / 1},
                    date::sys_days{2022_y / December / 1}};
  load_timetable({}, source_idx_t{0},
                 mem_dir::read(fmt::format("{}{}{}", kBasicTimetable,
                                           kBasicFares, kFareTimetable1)),
                 tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2022_y / March / 30});

  transit_realtime::FeedMessage msg;

  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(date::sys_days{2022_y / March / 30} + 9h));

  auto const e = msg.add_entity();
  e->set_id("1");
  e->set_is_deleted(false);

  auto const tu = e->mutable_trip_update();
  auto const td = tu->mutable_trip();
  td->set_start_time("02:35:00");
  td->set_start_date("20220330");
  td->set_trip_id("NEW");
  td->set_schedule_relationship(
      transit_realtime::TripDescriptor_ScheduleRelationship_NEW);
  tu->mutable_trip_properties()->set_trip_short_name("Additional");

  {
    auto const stop_update = tu->add_stop_time_update();
    stop_update->set_stop_sequence(1U);
    stop_update->set_stop_id("B");
    stop_update->mutable_departure()->set_time(1648600500);
  }
  {
    auto const stop_update = tu->add_stop_time_update();
    stop_update->set_stop_sequence(2U);
    stop_update->set_stop_id("E");
    stop_update->mutable_arrival()->set_time(1648600800);
  }

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg);
  EXPECT_EQ(stats.total_entities_success_, 1U);

  {
    auto const results = raptor_search(
        tt, &rtt, "A", "E", unixtime_t{sys_days{2022_y / March / 30}});
    ASSERT_EQ(1U, results.size());
    auto const fare_legs = get_fares(tt, &rtt, *results.begin());
    constexpr auto const kExpected = R"(FARE LEG:
   0: A       A...............................................                               d: 30.03 00:00 [30.03 02:00]  [{name=Line 1, day=2022-03-30, id=T1, src=0}]
   1: B       B............................................... a: 30.03 00:30 [30.03 02:30]
PRODUCTS
Pink App [priority=1]: 2.5 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Pink Paper Card [priority=1]: 3 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Adult
Pink Daypass [priority=1]: 4 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Pink Weekpass [priority=1]: 10 EUR, fare_media_name=App, fare_type=APP, ride_category=Adult
Pink App [priority=1]: 1.5 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors
Pink Paper Card [priority=1]: 2 EUR, fare_media_name=Paper Card, fare_type=PAPER, ride_category=Students and seniors
Pink Daypass [priority=1]: 2.5 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors
Pink Weekpass [priority=1]: 6 EUR, fare_media_name=App, fare_type=APP, ride_category=Students and seniors


FARE LEG:
   0: B       B...............................................                                                             d: 30.03 00:35 [30.03 02:35]  RT 30.03 00:35 [30.03 02:35]
   1: E       E............................................... a: 30.03 00:40 [30.03 02:40]  RT 30.03 00:40 [30.03 02:40]
PRODUCTS


)";
    EXPECT_EQ(kExpected, to_string(tt, &rtt, fare_legs));
  }
}

TEST(fares, fares_without_fares) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2022_y / January / 1},
                    date::sys_days{2022_y / December / 1}};
  load_timetable({}, source_idx_t{0}, mem_dir::read(kTimetableWithoutFares),
                 tt);
  finalize(tt);

  {
    auto const results = raptor_search(
        tt, nullptr, "A", "E", unixtime_t{sys_days{2022_y / March / 30}});
    ASSERT_EQ(1U, results.size());
    auto const fare_legs = get_fares(tt, nullptr, *results.begin());
    constexpr auto const kExpected = R"(FARE LEG:
   0: A       A...............................................                               d: 30.03 00:00 [30.03 02:00]  [{name=Line 1, day=2022-03-30, id=T1, src=0}]
   1: B       B............................................... a: 30.03 00:30 [30.03 02:30]
PRODUCTS


FARE LEG:
   0: B       B...............................................                               d: 30.03 00:35 [30.03 02:35]  [{name=Line 2, day=2022-03-30, id=T2, src=0}]
   1: C       C............................................... a: 30.03 01:00 [30.03 03:00]
PRODUCTS


FARE LEG:
   0: C       C...............................................                               d: 30.03 01:05 [30.03 03:05]  [{name=Line 3, day=2022-03-30, id=T3, src=0}]
   1: D       D............................................... a: 30.03 02:00 [30.03 04:00]
PRODUCTS


FARE LEG:
   0: D       D...............................................                               d: 30.03 02:05 [30.03 04:05]  [{name=Line 4, day=2022-03-30, id=T4, src=0}]
   1: E       E............................................... a: 30.03 02:30 [30.03 04:30]
PRODUCTS


)";
    EXPECT_EQ(kExpected, to_string(tt, nullptr, fare_legs));
  }
}

namespace {

// Networks defined only via routes.txt network_id (no networks.txt, no
// route_networks.txt) as allowed by the GTFS spec, see
// https://github.com/motis-project/motis/issues/905
constexpr auto const kRouteNetworkIdTimetable = R"(
# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
everyday,1,1,1,1,1,1,1,20220101,20221231

# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MBTA,MBTA,https://mbta.com,America/New_York

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,0.02,1.03,,
C,C,,0.04,1.05,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type,network_id
Red,MBTA,Red,Red Line,,1,rapid_transit
1,MBTA,1,Bus 1,,3,local_bus

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
Red,everyday,T1,T1,
1,everyday,T2,T2,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:30:00,10:30:00,B,2,0,0
T2,10:35:00,10:35:00,B,1,0,0
T2,11:00:00,11:00:00,C,2,0,0

# fare_media.txt
fare_media_id,fare_media_name,fare_media_type
cash,Cash,0

# fare_products.txt
fare_product_id,fare_product_name,fare_media_id,amount,currency
prod_rapid_transit,Rapid Transit Ticket,cash,2.40,USD
prod_local_bus,Local Bus Ticket,cash,1.70,USD

# fare_leg_rules.txt
leg_group_id,network_id,fare_product_id
leg_rapid_transit,rapid_transit,prod_rapid_transit
leg_local_bus,local_bus,prod_local_bus
)";

}  // namespace

TEST(fares, network_id_in_routes_txt) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2022_y / January / 1},
                    date::sys_days{2022_y / December / 1}};
  load_timetable({}, source_idx_t{0}, mem_dir::read(kRouteNetworkIdTimetable),
                 tt);
  finalize(tt);

  ASSERT_EQ(1U, tt.fares_.size());
  EXPECT_EQ(2U, tt.fares_[source_idx_t{0}].networks_.size());
  EXPECT_EQ(2U, tt.fares_[source_idx_t{0}].route_networks_.size());

  auto const results = raptor_search(
      tt, nullptr, "A", "C", unixtime_t{sys_days{2022_y / March / 30} + 14h});
  ASSERT_EQ(1U, results.size());
  auto const fare_legs = get_fares(tt, nullptr, *results.begin());

  // Each leg gets exactly the product of its own network.
  constexpr auto const kExpected = R"(FARE LEG:
   0: A       A...............................................                               d: 30.03 14:00 [30.03 10:00]  [{name=Red, day=2022-03-30, id=T1, src=0}]
   1: B       B............................................... a: 30.03 14:30 [30.03 10:30]
PRODUCTS
Rapid Transit Ticket [priority=0]: 2.4 USD, fare_media_name=Cash, fare_type=NONE, ride_category=??


FARE LEG:
   0: B       B...............................................                               d: 30.03 14:35 [30.03 10:35]  [{name=1, day=2022-03-30, id=T2, src=0}]
   1: C       C............................................... a: 30.03 15:00 [30.03 11:00]
PRODUCTS
Local Bus Ticket [priority=0]: 1.7 USD, fare_media_name=Cash, fare_type=NONE, ride_category=??


)";
  EXPECT_EQ(kExpected, to_string(tt, nullptr, fare_legs));
}

namespace {

// Two products (cash, card) per network, free transfer only for card holders.
// The cash product sorts first (cheaper), so the transfer rule has to be
// matched against all leg rules, not just the first one. Only the card
// products take part in the transfer (A+AB without transfer product = the
// following leg is included).
constexpr auto const kTransferThinningTimetable = R"(
# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
everyday,1,1,1,1,1,1,1,20220101,20221231

# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MBTA,MBTA,https://mbta.com,America/New_York

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,0.02,1.03,,
C,C,,0.04,1.05,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type,network_id
Red,MBTA,Red,Red Line,,1,rapid_transit
1,MBTA,1,Bus 1,,3,local_bus

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
Red,everyday,T1,T1,
1,everyday,T2,T2,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:30:00,10:30:00,B,2,0,0
T2,10:35:00,10:35:00,B,1,0,0
T2,11:00:00,11:00:00,C,2,0,0

# fare_media.txt
fare_media_id,fare_media_name,fare_media_type
cash,Cash,0
card,CharlieCard,2

# fare_products.txt
fare_product_id,fare_product_name,fare_media_id,amount,currency
prod_rapid_transit_cash,Subway Cash,cash,2.40,USD
prod_rapid_transit_card,Subway Card,card,2.90,USD
prod_local_bus_cash,Bus Cash,cash,1.70,USD
prod_local_bus_card,Bus Card,card,1.90,USD

# fare_leg_rules.txt
leg_group_id,network_id,fare_product_id
leg_rapid_transit_cash,rapid_transit,prod_rapid_transit_cash
leg_rapid_transit_card,rapid_transit,prod_rapid_transit_card
leg_local_bus_cash,local_bus,prod_local_bus_cash
leg_local_bus_card,local_bus,prod_local_bus_card

# fare_transfer_rules.txt
from_leg_group_id,to_leg_group_id,transfer_count,duration_limit,duration_limit_type,fare_transfer_type,fare_product_id
leg_rapid_transit_card,leg_local_bus_card,,7200,1,0,
)";

}  // namespace

TEST(fares, transfer_thins_out_uncovered_products) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2022_y / January / 1},
                    date::sys_days{2022_y / December / 1}};
  load_timetable({}, source_idx_t{0}, mem_dir::read(kTransferThinningTimetable),
                 tt);
  finalize(tt);

  auto const results = raptor_search(
      tt, nullptr, "A", "C", unixtime_t{sys_days{2022_y / March / 30} + 14h});
  ASSERT_EQ(1U, results.size());
  auto const fare_legs = get_fares(tt, nullptr, *results.begin());

  constexpr auto const kExpected = R"(FARE TRANSFER START
TRANSFER PRODUCT: 
RULE: A+AB
FARE LEG:
   0: A       A...............................................                               d: 30.03 14:00 [30.03 10:00]  [{name=Red, day=2022-03-30, id=T1, src=0}]
   1: B       B............................................... a: 30.03 14:30 [30.03 10:30]
PRODUCTS
Subway Card [priority=0]: 2.9 USD, fare_media_name=CharlieCard, fare_type=CARD, ride_category=??


FARE LEG:
   0: B       B...............................................                               d: 30.03 14:35 [30.03 10:35]  [{name=1, day=2022-03-30, id=T2, src=0}]
   1: C       C............................................... a: 30.03 15:00 [30.03 11:00]
PRODUCTS
Bus Card [priority=0]: 1.9 USD, fare_media_name=CharlieCard, fare_type=CARD, ride_category=??


FARE TRANSFER END
)";
  EXPECT_EQ(kExpected, to_string(tt, nullptr, fare_legs));
}

namespace {

// Modeled after Naolib (Nantes), see
// https://github.com/motis-project/motis/issues/911: free legs on weekends
// (rule priority) must not be replaced by a transfer with a paid product.
// On weekdays, the cheapest applicable transfer product (1h, not 24h) is used
// and legs outside of its duration limit are charged separately.
constexpr auto const kCheapestTransferFares = R"(
# networks.txt
network_id,network_name
1,Naolib

# route_networks.txt
network_id,route_id
1,line1
1,line2
1,line3
1,air

# fare_media.txt
fare_media_id,fare_media_name,fare_media_type
ticket,Ticket,1
none,No ticket needed,0

# fare_products.txt
fare_product_id,fare_product_name,amount,currency,fare_media_id
1h,1h,1.80,EUR,ticket
24h,24h,6.50,EUR,ticket
gratuit,Gratuit,0.00,EUR,none

# timeframes.txt
timeframe_group_id,start_time,end_time,service_id
weekend-free,0:00:00,24:00:00,weekend

# fare_leg_rules.txt
leg_group_id,network_id,from_timeframe_group_id,to_timeframe_group_id,fare_product_id,rule_priority
general_network,1,,,1h,
general_network,1,,,24h,
general_network,1,weekend-free,weekend-free,gratuit,1

# fare_transfer_rules.txt
from_leg_group_id,to_leg_group_id,transfer_count,duration_limit,duration_limit_type,fare_transfer_type,fare_product_id
general_network,general_network,-1,86400,0,2,24h
general_network,general_network,-1,3600,0,2,1h
)";

}  // namespace

TEST(fares, cheapest_transfer_or_separate_legs) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2022_y / January / 1},
                    date::sys_days{2022_y / December / 1}};
  load_timetable({}, source_idx_t{0},
                 mem_dir::read(fmt::format("{}{}", kBasicTimetable,
                                           kCheapestTransferFares)),
                 tt);
  finalize(tt);

  // variants = cheapest covers per traveler profile (any media, paper
  // ticket, no ticket needed): identical to the main cover in both cases
  auto const n_main = [](std::vector<fare_transfer> const& x) {
    return utl::count_if(x, [](auto&& t) { return t.main_; });
  };

  {  // Saturday: all legs free, no transfer product
    auto const results = raptor_search(
        tt, nullptr, "A", "D", unixtime_t{sys_days{2022_y / April / 2} + 10h});
    ASSERT_EQ(1U, results.size());
    auto const fare_legs = get_fares(tt, nullptr, *results.begin());
    EXPECT_EQ(3U, fare_legs.size());
    EXPECT_EQ(3U, n_main(fare_legs));
    constexpr auto const kExpected = R"(FARE LEG:
   0: A       A...............................................                               d: 02.04 10:00 [02.04 12:00]  [{name=Line 1, day=2022-04-02, id=T1, src=0}]
   1: B       B............................................... a: 02.04 10:30 [02.04 12:30]
PRODUCTS
Gratuit [priority=1]: 0 EUR, fare_media_name=No ticket needed, fare_type=NONE, ride_category=??


FARE LEG:
   0: B       B...............................................                               d: 02.04 10:35 [02.04 12:35]  [{name=Line 2, day=2022-04-02, id=T2, src=0}]
   1: C       C............................................... a: 02.04 11:00 [02.04 13:00]
PRODUCTS
Gratuit [priority=1]: 0 EUR, fare_media_name=No ticket needed, fare_type=NONE, ride_category=??


FARE LEG:
   0: C       C...............................................                               d: 02.04 11:05 [02.04 13:05]  [{name=Line 3, day=2022-04-02, id=T3, src=0}]
   1: D       D............................................... a: 02.04 12:00 [02.04 14:00]
PRODUCTS
Gratuit [priority=1]: 0 EUR, fare_media_name=No ticket needed, fare_type=NONE, ride_category=??


)";
    EXPECT_EQ(kExpected, to_string(tt, nullptr, fare_legs));
  }

  {  // Wednesday: 1h ticket for the first two legs (60min), third leg separate
    auto const results = raptor_search(
        tt, nullptr, "A", "D", unixtime_t{sys_days{2022_y / March / 30} + 10h});
    ASSERT_EQ(1U, results.size());
    auto const fare_legs = get_fares(tt, nullptr, *results.begin());
    EXPECT_EQ(2U, fare_legs.size());
    EXPECT_EQ(2U, n_main(fare_legs));
    constexpr auto const kExpected = R"(FARE TRANSFER START
TRANSFER PRODUCT: 1h
RULE: AB
FARE LEG:
   0: A       A...............................................                               d: 30.03 10:00 [30.03 12:00]  [{name=Line 1, day=2022-03-30, id=T1, src=0}]
   1: B       B............................................... a: 30.03 10:30 [30.03 12:30]
PRODUCTS
1h [priority=0]: 1.8 EUR, fare_media_name=Ticket, fare_type=PAPER, ride_category=??
24h [priority=0]: 6.5 EUR, fare_media_name=Ticket, fare_type=PAPER, ride_category=??


FARE LEG:
   0: B       B...............................................                               d: 30.03 10:35 [30.03 12:35]  [{name=Line 2, day=2022-03-30, id=T2, src=0}]
   1: C       C............................................... a: 30.03 11:00 [30.03 13:00]
PRODUCTS
1h [priority=0]: 1.8 EUR, fare_media_name=Ticket, fare_type=PAPER, ride_category=??
24h [priority=0]: 6.5 EUR, fare_media_name=Ticket, fare_type=PAPER, ride_category=??


FARE TRANSFER END
FARE LEG:
   0: C       C...............................................                               d: 30.03 11:05 [30.03 13:05]  [{name=Line 3, day=2022-03-30, id=T3, src=0}]
   1: D       D............................................... a: 30.03 12:00 [30.03 14:00]
PRODUCTS
1h [priority=0]: 1.8 EUR, fare_media_name=Ticket, fare_type=PAPER, ride_category=??
24h [priority=0]: 6.5 EUR, fare_media_name=Ticket, fare_type=PAPER, ride_category=??


)";
    EXPECT_EQ(kExpected, to_string(tt, nullptr, fare_legs));
  }
}
