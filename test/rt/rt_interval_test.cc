#include "gtest/gtest.h"

#include "pugixml.hpp"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/vdv_aus.h"
#include "nigiri/timetable.h"

#include "./util.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::rt;
using namespace date;
using namespace std::chrono_literals;

// Interval boundaries:
//  - static timetable: date_range_ = [2024-07-01, 2024-08-31[
//    internal_interval() = [2024-06-26, 2024-09-01[
//  - RT timetable: event times are stored as delta_t (int16 minutes) relative
//    to base_day_, so rt_interval() = base_day_ -/+ 22d 18h 07min.
//    Base day 2024-07-01 -> rt_interval() = [06-08 05:53, 07-23 18:07[
//    Base day 2024-08-30 -> rt_interval() = [08-07 05:53, 09-21 18:07[
//    is_in_rt_interval() requires runs to be kRtIntervalBuffer (1 day) inside:
//    Base day 2024-07-01 -> [06-09 05:53, 07-22 18:07[
//    Base day 2024-08-30 -> [08-08 05:53, 09-20 18:07[
//
// Trips (UTC, daily):
//  - DAY:   A 10:00 -> B 10:30/10:31 -> C 11:00
//  - NIGHT: A 23:30 -> B 00:15/00:16 (+1d) -> C 01:00 (+1d)
//  - EDGE:  A 05:50 -> B 12:00/12:01 -> C 18:10
//    (straddles both the 05:53 and the 18:07 boundaries)

namespace {

constexpr auto const kDelay = 5;

mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,49.0,8.0
B,B,49.1,8.1
C,C,49.2,8.2

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
S,1,1,1,1,1,1,1,20240101,20241231

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
R,AG,R,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign
R,S,DAY,C
R,S,NIGHT,C
R,S,EDGE,C

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
DAY,10:00:00,10:00:00,A,1
DAY,10:30:00,10:31:00,B,2
DAY,11:00:00,11:00:00,C,3
NIGHT,23:30:00,23:30:00,A,1
NIGHT,24:15:00,24:16:00,B,2
NIGHT,25:00:00,25:00:00,C,3
EDGE,05:50:00,05:50:00,A,1
EDGE,12:00:00,12:01:00,B,2
EDGE,18:10:00,18:10:00,C,3
)");
}

timetable load() {
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2024_y / July / 1},
                    sys_days{2024_y / August / 31}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);
  return tt;
}

std::string yyyymmdd(sys_days const d) { return date::format("%Y%m%d", d); }

transit_realtime::FeedMessage gtfsrt_msg(
    std::string const& trip_id,
    sys_days const start_date,
    transit_realtime::TripDescriptor_ScheduleRelationship const sr =
        transit_realtime::TripDescriptor_ScheduleRelationship_SCHEDULED) {
  auto msg = transit_realtime::FeedMessage{};
  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(test::to_unix(start_date));

  auto const e = msg.add_entity();
  e->set_id("1");
  auto const td = e->mutable_trip_update()->mutable_trip();
  td->set_trip_id(trip_id);
  td->set_start_date(yyyymmdd(start_date));
  td->set_schedule_relationship(sr);

  if (sr == transit_realtime::TripDescriptor_ScheduleRelationship_SCHEDULED) {
    auto const upd = e->mutable_trip_update()->add_stop_time_update();
    upd->set_stop_sequence(1U);
    upd->mutable_departure()->set_delay(kDelay * 60);
  }
  return msg;
}

transit_realtime::FeedMessage gtfsrt_new_trip_msg(std::string const& trip_id,
                                                  unixtime_t const dep,
                                                  unixtime_t const arr) {
  auto msg = transit_realtime::FeedMessage{};
  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(test::to_unix(dep));

  auto const e = msg.add_entity();
  e->set_id("1");
  auto const td = e->mutable_trip_update()->mutable_trip();
  td->set_trip_id(trip_id);
  td->set_schedule_relationship(
      transit_realtime::TripDescriptor_ScheduleRelationship_NEW);

  auto const first = e->mutable_trip_update()->add_stop_time_update();
  first->set_stop_id("A");
  first->mutable_departure()->set_time(
      static_cast<std::int64_t>(test::to_unix(dep)));

  auto const last_stu = e->mutable_trip_update()->add_stop_time_update();
  last_stu->set_stop_id("C");
  last_stu->mutable_arrival()->set_time(
      static_cast<std::int64_t>(test::to_unix(arr)));

  return msg;
}

rt::run resolve(timetable const& tt,
                rt_timetable const& rtt,
                std::string const& trip_id,
                sys_days const start_date) {
  auto td = transit_realtime::TripDescriptor{};
  td.set_trip_id(trip_id);
  td.set_start_date(yyyymmdd(start_date));
  return rt::gtfsrt_resolve_run(start_date, tt, &rtt, source_idx_t{0}, td)
      .first;
}

rt::statistics gtfsrt_update(timetable const& tt,
                             rt_timetable& rtt,
                             std::string const& trip_id,
                             sys_days const start_date) {
  return rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag",
                               gtfsrt_msg(trip_id, start_date));
}

void expect_delayed(timetable const& tt,
                    rt_timetable const& rtt,
                    std::string const& trip_id,
                    sys_days const start_date) {
  auto const r = resolve(tt, rtt, trip_id, start_date);
  ASSERT_TRUE(r.valid()) << trip_id << " " << start_date;
  ASSERT_TRUE(r.is_rt()) << trip_id << " " << start_date;

  auto const fr = rt::frun{tt, &rtt, r};
  auto const last_stop = static_cast<stop_idx_t>(fr.size() - 1U);
  EXPECT_EQ(fr[0].scheduled_time(event_type::kDep) + kDelay * 1min,
            fr[0].time(event_type::kDep))
      << trip_id << " " << start_date;
  EXPECT_EQ(fr[last_stop].scheduled_time(event_type::kArr) + kDelay * 1min,
            fr[last_stop].time(event_type::kArr))
      << trip_id << " " << start_date;
}

void expect_not_rt(timetable const& tt,
                   rt_timetable const& rtt,
                   std::string const& trip_id,
                   sys_days const start_date) {
  auto const r = resolve(tt, rtt, trip_id, start_date);
  ASSERT_TRUE(r.valid()) << trip_id << " " << start_date;
  EXPECT_FALSE(r.is_rt()) << trip_id << " " << start_date;
}

struct siri_call {
  std::string_view stop_;
  std::optional<unixtime_t> arr_, dep_;
};

std::string siri_msg(std::string_view journey_ref,
                     sys_days const service_day,
                     std::vector<siri_call> const& calls) {
  auto const fmt_time = [](unixtime_t const t) {
    return date::format("%FT%TZ", std::chrono::sys_seconds{t});
  };

  auto calls_str = std::string{};
  for (auto const& c : calls) {
    calls_str +=
        fmt::format("<EstimatedCall><StopPointRef>{}</StopPointRef>", c.stop_);
    if (c.arr_.has_value()) {
      calls_str += fmt::format(
          "<AimedArrivalTime>{}</AimedArrivalTime>"
          "<ExpectedArrivalTime>{}</ExpectedArrivalTime>",
          fmt_time(*c.arr_), fmt_time(*c.arr_ + kDelay * 1min));
    }
    if (c.dep_.has_value()) {
      calls_str += fmt::format(
          "<AimedDepartureTime>{}</AimedDepartureTime>"
          "<ExpectedDepartureTime>{}</ExpectedDepartureTime>",
          fmt_time(*c.dep_), fmt_time(*c.dep_ + kDelay * 1min));
    }
    calls_str += "</EstimatedCall>";
  }

  return fmt::format(
      R"(<?xml version="1.0" encoding="UTF-8"?>
<Siri xmlns="http://www.siri.org.uk/siri" version="2.0">
  <ServiceDelivery>
    <EstimatedTimetableDelivery version="2.0">
      <EstimatedJourneyVersionFrame>
        <EstimatedVehicleJourney>
          <LineRef>R</LineRef>
          <FramedVehicleJourneyRef>
            <DataFrameRef>{}</DataFrameRef>
            <DatedVehicleJourneyRef>{}</DatedVehicleJourneyRef>
          </FramedVehicleJourneyRef>
          <EstimatedCalls>{}</EstimatedCalls>
          <IsCompleteStopSequence>true</IsCompleteStopSequence>
        </EstimatedVehicleJourney>
      </EstimatedJourneyVersionFrame>
    </EstimatedTimetableDelivery>
  </ServiceDelivery>
</Siri>)",
      date::format("%F", service_day), journey_ref, calls_str);
}

std::string siri_day_msg(sys_days const d) {
  return siri_msg(fmt::format("DAY-{}", d), d,
                  {{"A", std::nullopt, d + 10h},
                   {"B", d + 10h + 30min, d + 10h + 31min},
                   {"C", d + 11h, std::nullopt}});
}

std::string siri_night_msg(sys_days const d) {
  return siri_msg(fmt::format("NIGHT-{}", d), d,
                  {{"A", std::nullopt, d + 23h + 30min},
                   {"B", d + 24h + 15min, d + 24h + 16min},
                   {"C", d + 25h, std::nullopt}});
}

std::string siri_edge_msg(sys_days const d) {
  return siri_msg(fmt::format("EDGE-{}", d), d,
                  {{"A", std::nullopt, d + 5h + 50min},
                   {"B", d + 12h, d + 12h + 1min},
                   {"C", d + 18h + 10min, std::nullopt}});
}

vdv_aus::statistics siri_update(rt_timetable& rtt,
                                vdv_aus::updater& u,
                                std::string const& msg) {
  auto doc = pugi::xml_document{};
  EXPECT_TRUE(doc.load_string(msg.c_str()));
  return u.update(rtt, doc);
}

}  // namespace

TEST(rt_interval, rt_interval_bounds) {
  auto const tt = load();

  auto const rtt_start =
      rt::create_rt_timetable(tt, sys_days{2024_y / July / 1});
  EXPECT_EQ((interval<unixtime_t>{sys_days{2024_y / June / 8} + 5h + 53min,
                                  sys_days{2024_y / July / 23} + 18h + 7min}),
            rtt_start.rt_interval());

  auto const rtt_end =
      rt::create_rt_timetable(tt, sys_days{2024_y / August / 30});
  EXPECT_EQ(
      (interval<unixtime_t>{sys_days{2024_y / August / 7} + 5h + 53min,
                            sys_days{2024_y / September / 21} + 18h + 7min}),
      rtt_end.rt_interval());
}

TEST(rt_interval, gtfsrt_static_interval_start) {
  auto const tt = load();
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / July / 1});

  // First day of the static timetable.
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY", sys_days{2024_y / July / 1})
                   .total_entities_success_);
  expect_delayed(tt, rtt, "DAY", sys_days{2024_y / July / 1});

  // Departs before date_range_.from_, arrives within: part of the timetable.
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "NIGHT", sys_days{2024_y / June / 30})
                   .total_entities_success_);
  expect_delayed(tt, rtt, "NIGHT", sys_days{2024_y / June / 30});

  // Completely before date_range_.from_: not part of the timetable.
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY", sys_days{2024_y / June / 30})
                   .trip_resolve_error_);
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "NIGHT", sys_days{2024_y / June / 29})
                   .trip_resolve_error_);

  // Before internal_interval().from_.
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY", sys_days{2024_y / June / 1})
                   .trip_resolve_error_);

  EXPECT_EQ(2U, rtt.n_rt_transports());
}

TEST(rt_interval, gtfsrt_static_interval_end) {
  auto const tt = load();
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / August / 30});

  // Last day of the static timetable.
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY", sys_days{2024_y / August / 30})
                   .total_entities_success_);
  expect_delayed(tt, rtt, "DAY", sys_days{2024_y / August / 30});

  // Arrives after date_range_.to_.
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "NIGHT", sys_days{2024_y / August / 30})
                   .total_entities_success_);
  expect_delayed(tt, rtt, "NIGHT", sys_days{2024_y / August / 30});
  EXPECT_EQ(sys_days{2024_y / August / 31} + 1h + 5min,
            rtt.unix_event_time(rt_transport_idx_t{1U}, 2U, event_type::kArr));

  // date_range_.to_ is exclusive.
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY", sys_days{2024_y / August / 31})
                   .trip_resolve_error_);
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "NIGHT", sys_days{2024_y / August / 31})
                   .trip_resolve_error_);

  // Beyond internal_interval().to_ and beyond the traffic day bitfield.
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY", sys_days{2024_y / October / 1})
                   .trip_resolve_error_);
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY",
                             sys_days{2024_y / June / 26} + days{kMaxDays})
                   .trip_resolve_error_);

  EXPECT_EQ(2U, rtt.n_rt_transports());
}

TEST(rt_interval, gtfsrt_rt_interval_end) {
  auto const tt = load();
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / July / 1});

  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "EDGE", sys_days{2024_y / July / 21})
                   .total_entities_success_);
  expect_delayed(tt, rtt, "EDGE", sys_days{2024_y / July / 21});

  // Arrival 11:00 < 18:07
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY", sys_days{2024_y / July / 22})
                   .total_entities_success_);
  expect_delayed(tt, rtt, "DAY", sys_days{2024_y / July / 22});

  // Departure 05:50 < 18:07, but arrival 18:10 >= 18:07
  auto const edge =
      gtfsrt_update(tt, rtt, "EDGE", sys_days{2024_y / July / 22});
  EXPECT_EQ(0, edge.total_entities_success_);
  EXPECT_EQ(1, edge.total_entities_fail_);
  expect_not_rt(tt, rtt, "EDGE", sys_days{2024_y / July / 22});

  // Departure 23:30 >= 18:07
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "NIGHT", sys_days{2024_y / July / 22})
                   .total_entities_fail_);
  expect_not_rt(tt, rtt, "NIGHT", sys_days{2024_y / July / 22});

  // Still within the static timetable, but far outside the RT interval.
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY", sys_days{2024_y / August / 30})
                   .total_entities_fail_);
  expect_not_rt(tt, rtt, "DAY", sys_days{2024_y / August / 30});

  EXPECT_EQ(2U, rtt.n_rt_transports());
}

TEST(rt_interval, gtfsrt_rt_interval_start) {
  auto const tt = load();
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / August / 30});

  // Departure 05:50 >= 05:53 (previous day).
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "EDGE", sys_days{2024_y / August / 9})
                   .total_entities_success_);
  expect_delayed(tt, rtt, "EDGE", sys_days{2024_y / August / 9});

  // Departure 10:00 >= 05:53
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY", sys_days{2024_y / August / 8})
                   .total_entities_success_);
  expect_delayed(tt, rtt, "DAY", sys_days{2024_y / August / 8});

  // Departure 05:50 < 05:53
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "EDGE", sys_days{2024_y / August / 8})
                   .total_entities_fail_);
  expect_not_rt(tt, rtt, "EDGE", sys_days{2024_y / August / 8});

  // Arrival 01:00 (+1d) < 05:53
  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "NIGHT", sys_days{2024_y / August / 7})
                   .total_entities_fail_);
  expect_not_rt(tt, rtt, "NIGHT", sys_days{2024_y / August / 7});

  EXPECT_EQ(2U, rtt.n_rt_transports());

  // Cancellations only touch the traffic day bitfield: no RT interval needed.
  auto const cancel = rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "tag",
      gtfsrt_msg(
          "EDGE", sys_days{2024_y / August / 8},
          transit_realtime::TripDescriptor_ScheduleRelationship_CANCELED));
  EXPECT_EQ(1, cancel.total_entities_success_);
  auto const cancelled =
      resolve(tt, rtt, "EDGE", sys_days{2024_y / August / 8});
  ASSERT_TRUE(cancelled.valid());
  EXPECT_FALSE(rtt.is_transport_active(cancelled.t_.t_idx_, cancelled.t_.day_));
  EXPECT_EQ(2U, rtt.n_rt_transports());
}

TEST(rt_interval, gtfsrt_rt_interval_buffer) {
  auto const tt = load();
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / July / 1});
  auto const d = sys_days{2024_y / July / 21};

  // Last run accepted at the end can still be delayed by kRtIntervalBuffer.
  auto msg = gtfsrt_msg("EDGE", d);
  msg.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_stop_time_update(0)
      ->mutable_departure()
      ->set_delay(kRtIntervalBuffer.count() * 60);
  EXPECT_EQ(1, rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg)
                   .total_entities_success_);

  auto const fr = rt::frun{tt, &rtt, resolve(tt, rtt, "EDGE", d)};
  ASSERT_TRUE(fr.is_rt());
  EXPECT_EQ(d + 1_days + 5h + 50min, fr[0].time(event_type::kDep));
  EXPECT_EQ(d + 1_days + 18h + 10min, fr[2].time(event_type::kArr));
  EXPECT_EQ(rt_data_state::kPropagated, fr[2].data_state(event_type::kArr));
}

TEST(rt_interval, gtfsrt_new_trip_rt_interval) {
  auto const tt = load();
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / July / 1});

  auto const d = sys_days{2024_y / July / 23};

  // Last arrival 18:06 < 18:07
  EXPECT_EQ(1, rt::gtfsrt_update_msg(
                   tt, rtt, source_idx_t{0}, "tag",
                   gtfsrt_new_trip_msg("NEW_IN", d + 17h, d + 18h + 6min))
                   .total_entities_success_);

  // Last arrival 18:07 is the (exclusive) end of the RT interval
  EXPECT_EQ(0, rt::gtfsrt_update_msg(
                   tt, rtt, source_idx_t{0}, "tag",
                   gtfsrt_new_trip_msg("NEW_OUT", d + 17h, d + 18h + 7min))
                   .total_entities_success_);

  auto const& added = rtt.additional_trips_[source_idx_t{0}];
  ASSERT_TRUE(added.ids_.find("NEW_IN").has_value());
  EXPECT_FALSE(added.ids_.find("NEW_OUT").has_value());
  EXPECT_EQ(1U, rtt.n_rt_transports());
  EXPECT_EQ(d + 18h + 6min,
            rtt.unix_event_time(rt_transport_idx_t{0U}, 1U, event_type::kArr));
}

TEST(rt_interval, siri_static_interval_start) {
  auto const tt = load();
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / July / 1});
  auto u = vdv_aus::updater{tt, source_idx_t{0},
                            vdv_aus::updater::xml_format::kSiri};

  siri_update(rtt, u, siri_day_msg(sys_days{2024_y / July / 1}));
  expect_delayed(tt, rtt, "DAY", sys_days{2024_y / July / 1});

  // Departs before date_range_.from_, arrives within.
  siri_update(rtt, u, siri_night_msg(sys_days{2024_y / June / 30}));
  expect_delayed(tt, rtt, "NIGHT", sys_days{2024_y / June / 30});

  // Not part of the timetable.
  EXPECT_EQ(0U, siri_update(rtt, u, siri_day_msg(sys_days{2024_y / June / 30}))
                    .matched_runs_);

  // First stop before internal_interval().from_, rest within.
  EXPECT_EQ(0U,
            siri_update(rtt, u, siri_night_msg(sys_days{2024_y / June / 25}))
                .matched_runs_);

  // Completely before internal_interval().from_.
  EXPECT_EQ(0U, siri_update(rtt, u, siri_day_msg(sys_days{2024_y / June / 24}))
                    .matched_runs_);
  EXPECT_EQ(0U, siri_update(rtt, u, siri_day_msg(sys_days{2024_y / June / 1}))
                    .matched_runs_);

  EXPECT_EQ(2U, rtt.n_rt_transports());
}

TEST(rt_interval, siri_static_interval_end) {
  auto const tt = load();
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / August / 30});
  auto u = vdv_aus::updater{tt, source_idx_t{0},
                            vdv_aus::updater::xml_format::kSiri};

  siri_update(rtt, u, siri_day_msg(sys_days{2024_y / August / 30}));
  expect_delayed(tt, rtt, "DAY", sys_days{2024_y / August / 30});

  // Arrives after date_range_.to_.
  siri_update(rtt, u, siri_night_msg(sys_days{2024_y / August / 30}));
  expect_delayed(tt, rtt, "NIGHT", sys_days{2024_y / August / 30});

  // date_range_.to_ is exclusive.
  EXPECT_EQ(0U,
            siri_update(rtt, u, siri_day_msg(sys_days{2024_y / August / 31}))
                .matched_runs_);
  EXPECT_EQ(0U,
            siri_update(rtt, u, siri_night_msg(sys_days{2024_y / August / 31}))
                .matched_runs_);

  // Beyond internal_interval().to_
  EXPECT_EQ(0U,
            siri_update(rtt, u, siri_day_msg(sys_days{2024_y / September / 1}))
                .matched_runs_);
  EXPECT_EQ(0U,
            siri_update(rtt, u, siri_day_msg(sys_days{2024_y / October / 1}))
                .matched_runs_);

  EXPECT_EQ(2U, rtt.n_rt_transports());
}

TEST(rt_interval, siri_rt_interval) {
  auto const tt = load();

  {  // End: base day 2024-07-01 -> runs have to end before 07-22 18:07
    auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / July / 1});
    auto u = vdv_aus::updater{tt, source_idx_t{0},
                              vdv_aus::updater::xml_format::kSiri};

    siri_update(rtt, u, siri_edge_msg(sys_days{2024_y / July / 21}));
    expect_delayed(tt, rtt, "EDGE", sys_days{2024_y / July / 21});

    siri_update(rtt, u, siri_day_msg(sys_days{2024_y / July / 22}));
    expect_delayed(tt, rtt, "DAY", sys_days{2024_y / July / 22});

    auto const edge =
        siri_update(rtt, u, siri_edge_msg(sys_days{2024_y / July / 22}));
    EXPECT_EQ(1U, edge.matched_runs_);
    expect_not_rt(tt, rtt, "EDGE", sys_days{2024_y / July / 22});

    EXPECT_EQ(1U,
              siri_update(rtt, u, siri_day_msg(sys_days{2024_y / August / 30}))
                  .matched_runs_);
    expect_not_rt(tt, rtt, "DAY", sys_days{2024_y / August / 30});

    EXPECT_EQ(2U, rtt.n_rt_transports());
  }

  {  // Start: base day 2024-08-30 -> runs have to start at 08-08 05:53
    auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / August / 30});
    auto u = vdv_aus::updater{tt, source_idx_t{0},
                              vdv_aus::updater::xml_format::kSiri};

    siri_update(rtt, u, siri_edge_msg(sys_days{2024_y / August / 9}));
    expect_delayed(tt, rtt, "EDGE", sys_days{2024_y / August / 9});

    EXPECT_EQ(1U,
              siri_update(rtt, u, siri_edge_msg(sys_days{2024_y / August / 8}))
                  .matched_runs_);
    expect_not_rt(tt, rtt, "EDGE", sys_days{2024_y / August / 8});

    EXPECT_EQ(1U, rtt.n_rt_transports());
  }
}

TEST(rt_interval, gtfsrt_event_outside_rt_interval) {
  auto const tt = load();
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / July / 1});
  auto const d = sys_days{2024_y / July / 1};

  // Departure delay of 30 days -> skipped, arrival at C +5min -> applied.
  auto msg = gtfsrt_msg("DAY", d);
  auto const tu = msg.mutable_entity(0)->mutable_trip_update();
  tu->mutable_stop_time_update(0)->mutable_departure()->set_delay(30 * 86400);
  auto const last_stu = tu->add_stop_time_update();
  last_stu->set_stop_sequence(3U);
  last_stu->mutable_arrival()->set_delay(kDelay * 60);
  EXPECT_EQ(1, rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg)
                   .total_entities_success_);

  auto const fr = rt::frun{tt, &rtt, resolve(tt, rtt, "DAY", d)};
  ASSERT_TRUE(fr.is_rt());
  EXPECT_EQ(d + 10h, fr[0].time(event_type::kDep));
  EXPECT_EQ(rt_data_state::kNoRtData, fr[0].data_state(event_type::kDep));
  EXPECT_EQ(d + 10h + 30min, fr[1].time(event_type::kArr));
  EXPECT_EQ(d + 10h + 31min, fr[1].time(event_type::kDep));
  EXPECT_EQ(d + 11h + 5min, fr[2].time(event_type::kArr));
  EXPECT_EQ(rt_data_state::kPredicted, fr[2].data_state(event_type::kArr));

  // Absolute time far outside of the RT interval -> skipped.
  auto abs_msg = gtfsrt_msg("NIGHT", d);
  auto const abs_dep = abs_msg.mutable_entity(0)
                           ->mutable_trip_update()
                           ->mutable_stop_time_update(0)
                           ->mutable_departure();
  abs_dep->clear_delay();
  abs_dep->set_time(
      static_cast<std::int64_t>(test::to_unix(sys_days{2024_y / August / 30})));
  EXPECT_EQ(1, rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", abs_msg)
                   .total_entities_success_);

  auto const night = rt::frun{tt, &rtt, resolve(tt, rtt, "NIGHT", d)};
  ASSERT_TRUE(night.is_rt());
  EXPECT_EQ(d + 23h + 30min, night[0].time(event_type::kDep));
  EXPECT_EQ(d + 25h, night[2].time(event_type::kArr));
}

TEST(rt_interval, siri_event_outside_rt_interval) {
  auto const tt = load();
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / July / 1});
  auto u = vdv_aus::updater{tt, source_idx_t{0},
                            vdv_aus::updater::xml_format::kSiri};
  auto const d = sys_days{2024_y / July / 1};

  // Expected departure at A 60 days late -> skipped, rest -> applied.
  auto msg = siri_day_msg(d);
  auto const expected_dep_a =
      std::string{"<ExpectedDepartureTime>2024-07-01T10:05:00Z"};
  auto const pos = msg.find(expected_dep_a);
  ASSERT_NE(std::string::npos, pos);
  msg.replace(pos, expected_dep_a.size(),
              "<ExpectedDepartureTime>2024-08-30T10:05:00Z");

  auto const stats = siri_update(rtt, u, msg);
  EXPECT_EQ(1U, stats.matched_runs_);

  auto const fr = rt::frun{tt, &rtt, resolve(tt, rtt, "DAY", d)};
  ASSERT_TRUE(fr.is_rt());
  EXPECT_EQ(d + 10h, fr[0].time(event_type::kDep));
  EXPECT_EQ(d + 10h + 35min, fr[1].time(event_type::kArr));
  EXPECT_EQ(d + 10h + 36min, fr[1].time(event_type::kDep));
  EXPECT_EQ(d + 11h + 5min, fr[2].time(event_type::kArr));
}

TEST(rt_interval, base_day_before_timetable) {
  auto const tt = load();

  // Base day before internal_interval_days().from_ = 2024-06-26 -> clamped.
  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / June / 1});
  EXPECT_EQ(sys_days{2024_y / June / 26}, rtt.base_day_);
  EXPECT_EQ(day_idx_t{0U}, rtt.base_day_idx_);

  EXPECT_EQ(1, gtfsrt_update(tt, rtt, "DAY", sys_days{2024_y / July / 1})
                   .total_entities_success_);
  expect_delayed(tt, rtt, "DAY", sys_days{2024_y / July / 1});

  // Base days from internal_interval_days().from_ onwards are kept as is.
  for (auto const d :
       {sys_days{2024_y / June / 26}, sys_days{2024_y / July / 1},
        sys_days{2024_y / September / 30}}) {
    EXPECT_EQ(d, rt::create_rt_timetable(tt, d).base_day_);
  }
}
