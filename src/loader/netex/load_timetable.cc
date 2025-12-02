#include "nigiri/loader/netex/load_timetable.h"

#include "nigiri/loader/get_index.h"

#include <filesystem>
#include <ranges>
#include <string>

#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"
#include "utl/parser/arg_parser.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "wyhash.h"

#include "pugixml.hpp"

#include "nigiri/loader/gtfs/route_key.h"
#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/shape_prepare.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"

#include "utl/progress_tracker.h"

using namespace std::string_view_literals;
namespace fs = std::filesystem;

namespace nigiri::loader::netex {

// =====
// UTILS
// -----
template <typename T>
struct is_unique_ptr : std::false_type {};

template <typename T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type {};

template <typename T>
concept UniquePtr = is_unique_ptr<T>::value;

template <typename T>
concept Tuplable = cista::to_tuple_works_v<T>;

template <Tuplable T>
auto format_as(T const& t);

template <UniquePtr T>
auto format_as(T const& t);

template <UniquePtr T>
auto format_as(T const& t) {
  return format_as(*t);
}

template <Tuplable T>
auto format_as(T const& t) {
  return cista::to_tuple(t);
}

template <typename T>
std::unique_ptr<T> uniq(T&& t) {
  return std::make_unique<T>(std::forward<T>(t));
}

struct none {};

std::string_view ref(pugi::xml_node n, char const* child) {
  auto const str = n.child(child).attribute("ref").as_string();
  return str == nullptr ? std::string_view{} : std::string_view{str};
}

std::string_view val(pugi::xml_node n, char const* child) {
  auto const str = n.child(child).child_value();
  return str == nullptr ? std::string_view{} : std::string_view{str};
}

std::string_view id(pugi::xml_node n) {
  auto const str = n.attribute("id").as_string();
  return str == nullptr ? std::string_view{} : std::string_view{str};
}

geo::latlng get_pos(pugi::xml_node const x) {
  auto const loc = x.child("Location");
  return geo::latlng{utl::parse<double>(val(loc, "Latitude")),
                     utl::parse<double>(val(loc, "Longitude"))};
}

std::uint16_t get_route_type(std::string_view transport_mode,
                             std::string_view rail_sub_mode) {
  switch (cista::hash(transport_mode)) {
    case cista::hash("bus"): return 3;
    // Bus with two overhead wires using spring-loaded trolley poles.
    case cista::hash("trolleyBus"): return 11;

    case cista::hash("tram"): return 0;

    case cista::hash("rail"):
      switch (cista::hash(rail_sub_mode)) {
        case cista::hash("highSpeedRail"): return 101;
        case cista::hash("rackAndPinionRailway"): return 1400;  // ?
        case cista::hash("regionalRail"): return 106;
        case cista::hash("interregionalRail"): return 2;
        case cista::hash("crossCountryRail"):
        case cista::hash("longDistance"):
        case cista::hash("international"): return 102;
        case cista::hash("sleeperRailService"):
        case cista::hash("nightRail"): return 105;
        case cista::hash("carTransportRailService"): return 104;
        case cista::hash("touristRailway"): return 107;
        case cista::hash("airportLinkRail"):
        case cista::hash("railShuttle"):
        case cista::hash("suburbanRailway"): return 404;
        case cista::hash("replacementRailService"): return 714;  // bus?
        case cista::hash("specialTrain"): return 111;
        default: return 2;
      }

    // DEPRECATED. Use mode rail with RailSubmode longDistance.
    case cista::hash("intercityRail"): return 102;

    // DEPRECATED. Use mode metro with MetroSubmode urbanRail (or in some few
    // cases rail with RailSubmode local).
    case cista::hash("urbanRail"):
    case cista::hash("metro"):
      // Within an urban area. For underground and railway.
      return 400;

    // All air related mode. No special distinction is made.
    case cista::hash("air"): return 1100;

    // Most water related modes. The only specialisation is the mode ferry. In
    // Transmodel also ship was used for this mode.
    case cista::hash("water"): return 1200;

    // Can be only two cabines or multiple.
    case cista::hash("cableway"): return 1300;

    // Cable railway on steep slope using two counterbalanced carriages.
    case cista::hash("funicular"): return 1400;

    case cista::hash("taxi"): return 1507;

    // Ferry can be passenger ferries and/or car ferries. The specialisation
    // from water is the detailed schedule and usually the very rigid pattern
    // with only a few stops on the route.
    case cista::hash("ferry"): return 4;

    // General Mode for elevators and for Modes moved by cable. Especially
    // lifts are not only vertical elevators. If a better specialisation
    // applies (like cableway) use that.
    case cista::hash("lift"): return 1303;

    // For all alternative modes where a vehicle is driven by the user.
    case cista::hash("selfDrive"):
    case cista::hash("snowAndIce"):
    default: return 1700;
  }
}

// Parses YYYY-MM-DD
date::sys_days parse_date(utl::cstr s) {
  utl::verify(s.len >= 10, "invalid date {}", s.view());
  auto const date = date::year_month_day{
      date::year{utl::parse_verify<int>(s.substr(0, utl::size(4)))},
      date::month{utl::parse_verify<unsigned>(s.substr(5, utl::size(2)))},
      date::day{utl::parse_verify<unsigned>(s.substr(8, utl::size(2)))}};
  return date::sys_days{date};
}

// Parses HH:MM
duration_t parse_time(utl::cstr s, utl::cstr day_offset) {
  if (s.len < 5) {
    return duration_t{0};
  }
  return duration_t{utl::parse<int>(day_offset) * 1440 +
                    60 * utl::parse_verify<int>(s.substr(0, utl::size(2))) +
                    utl::parse_verify<int>(s.substr(3, utl::size(2)))};
}

// =========
// AUTHORITY
// ---------
struct authority {
  std::string_view name_;
  std::string_view short_name_;
};

using authority_map_t = hash_map<std::string_view, std::unique_ptr<authority>>;

authority_map_t get_authorities(pugi::xml_document const& doc) {
  auto authorities = authority_map_t{};
  authorities.emplace("", uniq(authority{}));
  for (auto const s :
       doc.select_nodes("//ResourceFrame/organisations/Authority")) {
    auto const n = s.node();
    authorities.emplace(id(n),
                        uniq(authority{.name_ = val(n, "Name"),
                                       .short_name_ = val(n, "ShortName")}));
  }
  return authorities;
}

// =========
// OPERATORS
// ---------
struct operätor {
  std::string_view code_;
  std::string_view name_;
  provider_idx_t provider_{provider_idx_t::invalid()};
};

using operator_map_t = hash_map<std::string_view, std::unique_ptr<operätor>>;

operator_map_t get_operators(pugi::xml_document const& doc) {
  auto operators = operator_map_t{};
  operators.emplace("", uniq(operätor{}));
  for (auto const s :
       doc.select_nodes("//ResourceFrame/organisations/Operator")) {
    auto const n = s.node();
    operators.emplace(id(n), uniq(operätor{.code_ = val(n, "PublicCode"),
                                           .name_ = val(n, "Name")}));
  }
  return operators;
}

// =============
// VEHICLE TYPES
// -------------
struct vehicle_type {
  std::string_view name_;
  std::string_view short_name_;
};

using vehicle_type_map_t =
    hash_map<std::string_view, std::unique_ptr<vehicle_type>>;

vehicle_type_map_t get_vehicle_types(pugi::xml_document const& doc) {
  auto vehicle_types = vehicle_type_map_t{};
  vehicle_types.emplace("", uniq(vehicle_type{}));
  for (auto const v :
       doc.select_nodes("//ResourceFrame/vehicleTypes/VehicleType")) {
    auto const n = v.node();
    vehicle_types.emplace(
        id(n), uniq(vehicle_type{.name_ = val(n, "Name"),
                                 .short_name_ = val(n, "ShortName")}));
  }
  return vehicle_types;
}

// =====
// LINES
// -----
struct line {
  hash_map<std::uint16_t /* GTFS route type */, route_id_idx_t> routes_{};
  std::string_view id_;
  std::string_view name_;
  authority const* authority_;
  operätor const* operator_;
};
using line_map_t = hash_map<std::string_view, std::unique_ptr<line>>;
line_map_t get_lines(pugi::xml_document const& doc,
                     authority_map_t const& authorities,
                     operator_map_t const& operators) {
  auto lines = line_map_t{};
  for (auto const l : doc.select_nodes("//ServiceFrame/lines/Line")) {
    auto const n = l.node();
    lines.emplace(
        id(n),
        uniq(line{
            .id_ = id(n),
            .name_ = val(n, "Name"),
            .authority_ = authorities.at(ref(n, "AuthorityRef")).get(),
            .operator_ =
                operators.at(ref(n.child("additionalOperators"), "OperatorRef"))
                    .get()}));
  }
  return lines;
}

// ====================
// DESTINATION DISPLAYS
// --------------------
struct destination_display {
  trip_direction_idx_t trip_direction_{trip_direction_idx_t::invalid()};
  std::string_view direction_;
};
using destination_display_map_t =
    hash_map<std::string_view, std::unique_ptr<destination_display>>;
destination_display_map_t get_destination_displays(
    pugi::xml_document const& doc) {
  auto destination_displays = destination_display_map_t{};
  destination_displays.emplace(std::string_view{},
                               uniq(destination_display{.direction_ = ""}));
  for (auto const display : doc.select_nodes(
           "//ServiceFrame/destinationDisplays/DestinationDisplay")) {
    auto const n = display.node();
    destination_displays.emplace(
        id(n), uniq(destination_display{.direction_ = val(n, "FrontText")}));
  }
  return destination_displays;
}

// =============
// STOPS & QUAYS
// -------------
struct stop {
  location_idx_t location_{location_idx_t::invalid()};
  stop const* parent_{nullptr};
  std::string_view id_;
  std::string_view name_;
  geo::latlng pos_;
};

using stop_map_t = hash_map<std::string_view, std::unique_ptr<stop>>;

stop_map_t get_stops(pugi::xml_document const& doc) {
  auto stops = stop_map_t{};
  auto parents = hash_map<stop*, std::string_view>{};
  for (auto const s : doc.select_nodes("//SiteFrame/stopPlaces/StopPlace | "
                                       "//SiteFrame/stopPlaces/Quay")) {
    auto const n = s.node();

    auto const get_global_id = [](pugi::xml_node const x) {
      return val(
          x.select_node("keyList/KeyValue/Key[text() = 'GlobalID']").parent(),
          "Value");
    };

    auto const stop_id = id(n);
    auto const global_stop_id = get_global_id(n);
    auto const parent =
        stops
            .emplace(stop_id,
                     uniq(stop{.id_ = !global_stop_id.empty() ? global_stop_id
                                                              : stop_id,
                               .name_ = val(n, "Name"),
                               .pos_ = get_pos(n.child("Centroid"))}))
            .first->second.get();

    auto const parent_ref = ref(n, "ParentSiteRef");
    if (!parent_ref.empty()) {
      parents.emplace(parent, parent_ref);
    }

    for (auto const q : n.select_nodes("quays/Quay")) {
      auto const qn = q.node();
      auto const quay_id = id(qn);
      auto const global_quay_id = get_global_id(qn);
      auto const pos = get_pos(qn.child("Centroid"));
      stops.emplace(
          quay_id,
          uniq(stop{.parent_ = parent,
                    .id_ = !global_quay_id.empty() ? global_quay_id : quay_id,
                    .name_ = val(qn, "Name"),
                    .pos_ = pos == geo::latlng{} ? parent->pos_ : pos}));
    }
  }
  for (auto& [stop, parent_ref] : parents) {
    stop->parent_ = stops.at(parent_ref).get();
  }
  return stops;
}

// ================
// STOP ASSIGNMENTS
// ----------------
using stop_assignment_map_t = hash_map<std::string_view, stop const*>;

stop_assignment_map_t get_stop_assignments(pugi::xml_document const& doc,
                                           stop_map_t const& stops) {
  auto stop_assigments = hash_map<std::string_view, stop const*>{};
  for (auto const& a : doc.select_nodes("//ServiceFrame/stopAssignments/"
                                        "PassengerStopAssignment")) {
    auto const n = a.node();
    auto const sstop = ref(n, "ScheduledStopPointRef");
    auto const quay = ref(n, "QuayRef");
    auto const stop_place = ref(n, "StopPlaceRef");
    if (!sstop.empty() && (!quay.empty() || !stop_place.empty())) {
      auto s = end(stops);

      if (!quay.empty()) {
        s = stops.find(quay);
      }
      if (s == end(stops) && !stop_place.empty()) {
        s = stops.find(stop_place);
      }

      if (s != end(stops)) {
        stop_assigments[sstop] = s->second.get();
      }
    }
  }
  return stop_assigments;
}

// =================
// OPERATING PERIODS
// -----------------
using operating_period_map_t =
    hash_map<std::string_view, std::unique_ptr<bitfield>>;

operating_period_map_t get_operating_periods(
    pugi::xml_document const& doc, interval<date::sys_days> const& interval) {
  auto operating_periods = operating_period_map_t{};
  for (auto const s : doc.select_nodes(
           "//ServiceCalendarFrame/ServiceCalendar/operatingPeriods/"
           "UicOperatingPeriod")) {
    auto const sn = s.node();
    auto const from = parse_date(val(sn, "FromDate"));
    auto const to = parse_date(val(sn, "ToDate"));
    auto const bits = val(sn, "ValidDayBits");

    utl::verify((to - from).count() + 1 == static_cast<int>(bits.size()),
                "from={}, to={} != n_bits={}", from, to, bits.size());

    auto bf = bitfield{};
    auto day = std::max(from, interval.from_);
    for (; day <= to && day < interval.to_; day += std::chrono::days{1}) {
      auto const tt_day_idx = (day - interval.from_).count();
      if (tt_day_idx >= 0 && tt_day_idx < static_cast<int>(bf.size())) {
        bf.set(static_cast<unsigned>(tt_day_idx),
               bits.at(static_cast<unsigned>((day - from).count())) != '0');
      }
    }

    operating_periods.emplace(id(sn), uniq(std::move(bf)));
  }
  return operating_periods;
}

// ====================
// DAY TYPE ASSIGNMENTS
// --------------------
using day_type_assignment_map_t = hash_map<std::string_view, bitfield const*>;
day_type_assignment_map_t get_day_type_assignments(
    pugi::xml_document const& doc,
    operating_period_map_t const& operating_periods) {
  auto days = hash_map<std::string_view, bitfield const*>{};
  for (auto const x : doc.select_nodes(
           "//ServiceCalendarFrame/ServiceCalendar/dayTypeAssignments/"
           "DayTypeAssignment")) {
    auto const n = x.node();
    days.emplace(ref(n, "DayTypeRef"),
                 operating_periods.at(ref(n, "OperatingPeriodRef")).get());
  }
  return days;
}

// ===============
// JOURNEY PATTERN
// ---------------
struct journey_pattern {
  struct stop_point {
    std::string_view id_;
    stop const* stop_;
    destination_display const* destination_display_;
    bool in_allowed_;
    bool out_allowed_;
  };
  line* line_;
  direction_id_t direction_;
  std::vector<stop_point> stop_points_;
};

using journey_pattern_map_t =
    hash_map<std::string_view, std::unique_ptr<journey_pattern>>;

journey_pattern_map_t get_journey_patterns(
    pugi::xml_document const& doc,
    stop_assignment_map_t const& stop_assignments,
    destination_display_map_t const& destination_displays,
    line_map_t const& lines,
    stop_map_t& stops) {
  auto journey_patterns = journey_pattern_map_t{};

  auto const get_stop =
      [&](std::string_view stop_point_ref) -> netex::stop const* {
    auto const it = stop_assignments.find(stop_point_ref);
    if (it != end(stop_assignments)) {
      return it->second;
    }

    // Invalid - fall back to information from ScheduledStopPoint
    return utl::get_or_create(
               stops, stop_point_ref,
               [&]() {
                 auto const stop_point = doc.select_node(
                     fmt::format("//ServiceFrame/scheduledStopPoints/"
                                 "ScheduledStopPoint[@id='{}']",
                                 stop_point_ref)
                         .c_str());
                 return uniq(stop{.id_ = stop_point_ref,
                                  .name_ = val(stop_point.node(), "Name"),
                                  .pos_ = get_pos(stop_point.node())});
               })
        .get();
  };

  for (auto const j : doc.select_nodes(
           "//ServiceFrame/journeyPatterns/ServiceJourneyPattern")) {
    auto const n = j.node();

    auto stop_points = std::vector<journey_pattern::stop_point>{};
    for (auto const sp : n.child("pointsInSequence").children()) {
      auto const in = val(sp, "ForBoarding");
      auto const out = val(sp, "ForAlighting");
      stop_points.push_back(journey_pattern::stop_point{
          .id_ = id(sp),
          .stop_ = get_stop(ref(sp, "ScheduledStopPointRef")),
          .destination_display_ =
              destination_displays.at(ref(sp, "DestinationDisplayRef")).get(),
          .in_allowed_ = in.empty() || in == "true"sv,
          .out_allowed_ = out.empty() || out == "true"sv});
    }

    journey_patterns.emplace(
        id(n),
        uniq(journey_pattern{
            .line_ = lines.at(ref(n.child("RouteView"), "LineRef")).get(),
            .direction_ =
                direction_id_t{ref(n, "DirectionRef").ends_with("1::") ? 0 : 1},
            .stop_points_ = stop_points}));
  }
  return journey_patterns;
}

// ===============
// SERVICE_JOURNEY
// ---------------
struct service_journey {
  struct stop_times {
    duration_t arr_, dep_;
  };
  std::string_view id_;
  std::uint32_t trip_nr_;
  std::uint16_t route_type_;
  vehicle_type const* vehicle_type_{};
  journey_pattern const* journey_pattern_{};
  bitfield const* traffic_days_;
  std::vector<stop_times> stop_times_{};
  ptrdiff_t dbg_offset_;
};

gtfs::stop_seq_t stop_seq(service_journey const& sj) {
  auto seq = gtfs::stop_seq_t{};
  for (auto const x : sj.journey_pattern_->stop_points_) {
    seq.push_back(nigiri::stop{x.stop_->location_, x.in_allowed_,
                               x.out_allowed_, x.in_allowed_, x.out_allowed_}
                      .value());
  }
  return seq;
}

std::vector<service_journey> get_service_journeys(
    pugi::xml_document const& doc,
    day_type_assignment_map_t const& day_type_assignments,
    vehicle_type_map_t const& vehicle_types,
    journey_pattern_map_t const& journey_patterns) {
  auto service_journeys = std::vector<service_journey>{};
  for (auto const s :
       doc.select_nodes("//TimetableFrame/vehicleJourneys/ServiceJourney")) {
    auto const n = s.node();

    auto sj = service_journey{
        .id_ = id(n),
        .trip_nr_ = utl::parse<std::uint32_t>(val(
            n.select_node("keyList/KeyValue/Key[text() = 'TripNr']").parent(),
            "Value")),
        .route_type_ =
            get_route_type(val(n, "TransportMode"),
                           val(n.child("TransportSubmode"), "RailSubmode")),
        .vehicle_type_ = vehicle_types.at(ref(n, "VehicleTypeRef")).get(),
        .journey_pattern_ =
            journey_patterns.at(ref(n, "ServiceJourneyPatternRef")).get(),
        .traffic_days_ = day_type_assignments.at(
            n.select_node("dayTypes/DayTypeRef/@ref").attribute().as_string()),
        .dbg_offset_ = n.offset_debug()};

    for (auto const [passing_time, stop_point] :
         utl::zip(n.select_nodes("passingTimes/TimetabledPassingTime"),
                  sj.journey_pattern_->stop_points_)) {
      auto const pn = passing_time.node();
      auto const stop_point_id = ref(pn, "StopPointInJourneyPatternRef");

      utl::verify(stop_point.id_ == stop_point_id,
                  "expected pointsInSequence.StopPointInJourneyPattern.id={}, "
                  "got TimetabledPassingTime.StopPointInJourneyPatternRef={}",
                  stop_point.id_, stop_point_id);

      sj.stop_times_.push_back(
          {.arr_ =
               parse_time(val(pn, "ArrivalTime"), val(pn, "ArrivalDayOffset")),
           .dep_ = parse_time(val(pn, "DepartureTime"),
                              val(pn, "DepartureDayOffset"))});
    }

    service_journeys.emplace_back(std::move(sj));
  }
  return service_journeys;
}

// ==============
// UTC CONVERSION
// --------------
struct key {
  basic_string<duration_t> utc_times_{};
  date::days first_dep_day_offset_;
  duration_t tz_offset_;
};

hash_map<key, bitfield> expand_local_to_utc(timetable const& tt,
                                            date::time_zone const* tz,
                                            service_journey const& sj) {
  auto utc_time_traffic_days = hash_map<key, bitfield>{};

  auto const first_day_offset =
      (sj.stop_times_.front().dep_ / 1_days) * date::days{1};
  auto const last_day_offset =
      (sj.stop_times_.back().arr_ / 1_days) * date::days{1};

  auto to_utc = [&, info = std::optional<date::local_info>{}](
                    date::sys_days const day,
                    minutes_after_midnight_t const x) mutable {
    if (!info || !interval{info->first.begin, info->first.end}.contains(
                     std::chrono::time_point_cast<date::sys_seconds::duration>(
                         day + x - info->first.offset))) {
      info = tz->get_info(
          date::local_time<i32_minutes>{day.time_since_epoch() + x});
    }
    return x - std::chrono::duration_cast<duration_t>(info->first.offset);
  };

  for (auto day = tt.internal_interval_days().from_;
       day != tt.internal_interval_days().to_; day += date::days{1}) {
    auto const service_days =
        interval{day + first_day_offset, day + last_day_offset + date::days{1}};
    if (!tt.date_range_.overlaps(service_days) ||
        !sj.traffic_days_->test(static_cast<unsigned>(
            (day - tt.internal_interval_days().from_).count()))) {
      continue;
    }

    auto const first_dep_utc = to_utc(day, sj.stop_times_.front().dep_);
    auto const first_dep_day_offset = date::days{static_cast<date::days::rep>(
        std::floor(static_cast<double>(first_dep_utc.count()) / 1440))};

    auto k = key{.first_dep_day_offset_ = first_dep_day_offset,
                 .tz_offset_ = sj.stop_times_.front().dep_ - first_dep_utc};
    for (auto const [dep, arr] :
         utl::pairwise(interval{0U, sj.stop_times_.size()})) {
      k.utc_times_.push_back(to_utc(day, sj.stop_times_[dep].dep_) +
                             first_dep_day_offset);
      k.utc_times_.push_back(to_utc(day, sj.stop_times_[arr].arr_) +
                             first_dep_day_offset);
    }

    auto const utc_traffic_day =
        (day - tt.internal_interval_days().from_ + first_dep_day_offset)
            .count();

    utc_time_traffic_days[k].set(static_cast<unsigned>(utc_traffic_day));
  }
  return utc_time_traffic_days;
}

// ====
// MAIN
// ----
bool is_xml_file(fs::path const& p) {
  return p.extension() == ".xml" || p.extension() == ".XML";
}

bool applicable(dir const& d) {
  return utl::any_of(d.list_files("."), is_xml_file);
}

void load_timetable(loader_config const& config,
                    source_idx_t const src,
                    dir const& d,
                    timetable& tt,
                    hash_map<bitfield, bitfield_idx_t>& bitfield_indices,
                    assistance_times* /*assistance*/,
                    shapes_storage* shapes_data) {
  auto const global_timer = nigiri::scoped_timer{"netex parser"};

  auto const locations_start = location_idx_t{tt.n_locations()};

  tt.n_sources_ = std::max(tt.n_sources_, to_idx(src + 1U));
  tt.fares_.emplace_back();
  tt.src_end_date_.push_back(date::sys_days::max());
  tt.route_ids_.emplace_back();

  auto const xml_files =
      utl::all(d.list_files("."))  //
      | utl::remove_if([&](fs::path const& f) { return !is_xml_file(f); })  //
      | utl::vec();

  struct intermediate {
    file f_;
    pugi::xml_document doc_;
    date::time_zone const* tz_;
    stop_map_t stops_;
    vehicle_type_map_t vehicle_types_;
    journey_pattern_map_t journey_patterns_;
    operating_period_map_t operating_periods_;
    destination_display_map_t destination_displays_;
    line_map_t lines_;
    authority_map_t authorities_;
    operator_map_t operators_;
    std::vector<service_journey> service_journeys_;
  };

  auto const r = script_runner{config.user_script_};

  struct utc_trip {
    date::days first_dep_offset_;
    duration_t tz_offset_;
    basic_string<duration_t> utc_times_;
    bitfield utc_traffic_days_;
    trip_idx_t trip_;
    basic_string<trip_direction_idx_t> trip_direction_;
    route_id_idx_t route_id_;
  };

  auto tt_mtx = std::mutex{};
  auto tz_map = gtfs::tz_map{};
  auto route_services =
      hash_map<gtfs::route_key_t, std::vector<std::vector<utc_trip>>,
               gtfs::route_key_hash, gtfs::route_key_equals>{};
  auto const add_expanded_trip = [&](clasz const c,
                                     gtfs::stop_seq_t const& stop_seq,
                                     utc_trip const& s) {
    auto const it = route_services.find(gtfs::route_key_ptr_t{c, &stop_seq});
    if (it != end(route_services)) {
      for (auto& route : it->second) {
        auto const idx = get_index(route, s);
        if (idx.has_value()) {
          route.insert(std::next(begin(route), static_cast<int>(*idx)), s);
          return;
        }
      }
      it->second.emplace_back(std::vector<utc_trip>{std::move(s)});
    } else {
      route_services.emplace(gtfs::route_key_t{c, stop_seq, {}, {}},
                             std::vector<std::vector<utc_trip>>{{s}});
    }
  };

  auto const pt = utl::get_active_progress_tracker();
  pt->status("Parse Files").out_bounds(0.F, 94.F).in_high(xml_files.size());

  utl::parallel_for(
      xml_files,
      [&](fs::path const& path) {
        auto im = intermediate{};

        try {
          auto f = d.get_file(path);
          auto doc = pugi::xml_document{};
          auto const result =
              f.is_mutable()
                  ? doc.load_buffer_inplace(
                        f.get_mutable(), f.size(),
                        pugi::parse_default | pugi::parse_trim_pcdata)
                  : doc.load_buffer(
                        f.data().data(), f.size(),
                        pugi::parse_default | pugi::parse_trim_pcdata);
          utl::verify(result, "Unable to parse XML buffer: {} at offset {}",
                      result.description(), result.offset);

          im.tz_ = date::locate_zone(
              doc.select_node("//FrameDefaults/DefaultLocale/TimeZone")
                  .node()
                  .child_value());
          im.authorities_ = get_authorities(doc);
          im.operators_ = get_operators(doc);
          im.stops_ = get_stops(doc);
          im.lines_ = get_lines(doc, im.authorities_, im.operators_);
          im.destination_displays_ = get_destination_displays(doc);
          im.journey_patterns_ = get_journey_patterns(
              doc, get_stop_assignments(doc, im.stops_),
              im.destination_displays_, im.lines_, im.stops_);
          im.operating_periods_ =
              get_operating_periods(doc, tt.internal_interval_days());
          im.vehicle_types_ = get_vehicle_types(doc);
          im.service_journeys_ = get_service_journeys(
              doc, get_day_type_assignments(doc, im.operating_periods_),
              im.vehicle_types_, im.journey_patterns_);
          im.f_ = std::move(f);
          im.doc_ = std::move(doc);
        } catch (std::exception const& e) {
          std::cout << "ERROR: " << e.what() << " IN " << path << "\n";
          return;
        }

        {
          auto lock = std::scoped_lock{tt_mtx};

          auto const source_file_idx =
              tt.register_source_file((d.path() / path).generic_string());
          auto const tz = gtfs::get_tz_idx(tt, tz_map, im.tz_->name());

          for (auto& [id, o] : im.operators_) {
            auto a = agency{src, id, o->name_, "", tz, tt, tz_map};
            if (process_agency(r, a)) {
              o->provider_ = register_agency(tt, a);
            }
          }

          auto const add_stop = [&](stop* stop) {
            auto const existing =
                tt.locations_.find(location_id{stop->id_, src});
            if (existing.has_value()) {
              stop->location_ = existing->l_;
              return;
            }

            auto s = loader::location{stop->id_,
                                      stop->name_,
                                      "",
                                      "",
                                      stop->pos_,
                                      src,
                                      location_type::kStation,
                                      stop->parent_ != nullptr
                                          ? stop->parent_->location_
                                          : location_idx_t::invalid(),
                                      tz,
                                      2_minutes,
                                      tt,
                                      tz_map};
            if (process_location(r, s)) {
              stop->location_ = register_location(tt, s);
            }
          };
          for (auto& [id, stop] : im.stops_) {
            if (stop->parent_ == nullptr) {
              add_stop(stop.get());
            }
          }
          for (auto& [id, stop] : im.stops_) {
            if (stop->parent_ != nullptr) {
              add_stop(stop.get());
            }
          }

          for (auto& [id, dd] : im.destination_displays_) {
            auto const idx = tt.trip_directions_.size();
            tt.trip_directions_.emplace_back(
                tt.register_trip_direction_string(dd->direction_));
            dd->trip_direction_ = trip_direction_idx_t{idx};
          }

          for (auto& sj : im.service_journeys_) {
            // Check if provider got filtered by script.
            if (sj.journey_pattern_->line_->operator_->provider_ ==
                provider_idx_t::invalid()) {
              continue;
            }

            // Create and register route from (line + route_type).
            auto& line = *sj.journey_pattern_->line_;
            auto const route_it = line.routes_.find(sj.route_type_);
            auto route_id = route_id_idx_t::invalid();
            if (route_it == end(line.routes_)) {
              auto const id = fmt::format("{}-{}", line.id_, sj.route_type_);
              auto rout = route{tt,
                                src,
                                id,
                                line.name_,
                                "",
                                route_type_t{sj.route_type_},
                                route_color{},
                                line.operator_->provider_};
              route_id = line.routes_
                             .emplace_hint(route_it, sj.route_type_,
                                           process_route(r, rout)
                                               ? register_route(tt, rout)
                                               : route_id_idx_t::invalid())
                             ->second;
            } else {
              route_id = route_it->second;
            }
            if (route_id == route_id_idx_t::invalid()) {
              continue;
            }

            // Create and register trip.
            auto const short_name = fmt::to_string(sj.trip_nr_);
            auto t = trip{
                src,
                sj.id_,
                tt.trip_direction(sj.journey_pattern_->stop_points_.front()
                                      .destination_display_->trip_direction_),
                short_name,
                line.name_,
                sj.vehicle_type_->name_,
                sj.vehicle_type_->short_name_,
                sj.journey_pattern_->direction_,
                route_id,
                tt};
            auto const trip_idx = process_trip(r, t) ? register_trip(tt, t)
                                                     : trip_idx_t::invalid();
            if (trip_idx == trip_idx_t::invalid()) {
              continue;
            }

            tt.trip_transport_ranges_.emplace_back();
            tt.trip_debug_.emplace_back().emplace_back(trip_debug{
                source_file_idx, static_cast<unsigned>(sj.dbg_offset_),
                static_cast<unsigned>(sj.dbg_offset_)});
            tt.trip_stop_seq_numbers_.add_back_sized(0U);
            if (shapes_data != nullptr) {
              shapes_data->add_trip_shape_offsets(
                  trip_idx, cista::pair{shape_idx_t::invalid(),
                                        shape_offset_idx_t::invalid()});
            }

            auto const stops = stop_seq(sj);
            auto const all_destinations_equal =
                utl::all_of(sj.journey_pattern_->stop_points_,
                            [&](journey_pattern::stop_point const& sp) {
                              return sp.destination_display_->trip_direction_ ==
                                     sj.journey_pattern_->stop_points_.front()
                                         .destination_display_->trip_direction_;
                            });
            auto trip_direction = basic_string<trip_direction_idx_t>{};
            if (all_destinations_equal) {
              trip_direction = {sj.journey_pattern_->stop_points_.front()
                                    .destination_display_->trip_direction_};
            } else {
              for (auto const& s : sj.journey_pattern_->stop_points_) {
                trip_direction.push_back(
                    s.destination_display_->trip_direction_);
              }
              trip_direction.pop_back();
            }
            for (auto const& [k, traffic_days] :
                 expand_local_to_utc(tt, im.tz_, sj)) {
              add_expanded_trip(
                  gtfs::to_clasz(
                      to_idx(tt.route_ids_[src].route_id_type_[route_id])),
                  stops,
                  {.first_dep_offset_ = k.first_dep_day_offset_,
                   .tz_offset_ = k.tz_offset_,
                   .utc_times_ = k.utc_times_,
                   .utc_traffic_days_ = traffic_days,
                   .trip_ = trip_idx,
                   .trip_direction_ = trip_direction,
                   .route_id_ = route_id});
            }
          }
        }
      },
      pt->update_fn());

  {
    auto const timer = scoped_timer{"loader.gtfs.write_trips"};

    pt->status("Write Trips")
        .out_bounds(94.F, 96.F)
        .in_high(route_services.size());

    auto const attributes = basic_string<attribute_combination_idx_t>{};
    auto lines = hash_map<std::string, trip_line_idx_t>{};
    auto section_providers = basic_string<provider_idx_t>{};
    auto route_colors = basic_string<route_color>{};
    auto external_trip_ids = basic_string<merged_trips_idx_t>{};
    auto location_routes = mutable_fws_multimap<location_idx_t, route_idx_t>{};
    for (auto const& [key, sub_routes] : route_services) {
      for (auto const& services : sub_routes) {
        auto const route_idx = tt.register_route(
            key.stop_seq_, {key.clasz_}, key.bikes_allowed_, key.cars_allowed_);

        for (auto const& s : key.stop_seq_) {
          auto s_routes = location_routes[nigiri::stop{s}.location_idx()];
          if (s_routes.empty() || s_routes.back() != route_idx) {
            s_routes.emplace_back(route_idx);
          }
        }

        for (auto const& s : services) {
          external_trip_ids.clear();
          section_providers.clear();
          route_colors.clear();

          tt.trip_transport_ranges_[s.trip_].emplace_back(transport_range_t{
              tt.next_transport_idx(),
              {static_cast<stop_idx_t>(0U),
               static_cast<stop_idx_t>(key.stop_seq_.size())}});

          auto const merged_trip = tt.register_merged_trip({s.trip_});
          external_trip_ids.push_back(merged_trip);
          route_colors.push_back({});
          section_providers.push_back(
              tt.route_ids_[src].route_id_provider_[s.route_id_]);

          assert(s.first_dep_offset_.count() >= -1);
          tt.add_transport(timetable::transport{
              .bitfield_idx_ = utl::get_or_create(
                  bitfield_indices, s.utc_traffic_days_,
                  [&]() { return tt.register_bitfield(s.utc_traffic_days_); }),
              .route_idx_ = route_idx,
              .first_dep_offset_ = {s.first_dep_offset_, s.tz_offset_},
              .external_trip_ids_ = external_trip_ids,
              .section_attributes_ = attributes,
              .section_providers_ = section_providers,
              .section_directions_ = s.trip_direction_,
              .route_colors_ = route_colors});
        }

        tt.finish_route();

        auto const stop_times_begin = tt.route_stop_times_.size();
        for (auto const [from, to] :
             utl::pairwise(interval{std::size_t{0U}, key.stop_seq_.size()})) {
          // Write departure times of all route services at stop i.
          for (auto const& s : services) {
            tt.route_stop_times_.emplace_back(s.utc_times_[from * 2]);
          }

          // Write arrival times of all route services at stop i+1.
          for (auto const& s : services) {
            tt.route_stop_times_.emplace_back(s.utc_times_[to * 2 - 1]);
          }
        }
        auto const stop_times_end = tt.route_stop_times_.size();
        tt.route_stop_time_ranges_.emplace_back(
            interval{stop_times_begin, stop_times_end});
      }

      pt->increment();
    }

    // Build location_routes map
    for (auto l = tt.location_routes_.size(); l != tt.n_locations(); ++l) {
      tt.location_routes_.emplace_back(location_routes[location_idx_t{l}]);
      assert(tt.location_routes_.size() == l + 1U);
    }
  }

  // Generate default footpaths.
  auto const locations_end = location_idx_t{tt.n_locations()};
  auto const new_locations = interval{locations_start, locations_end};
  auto const get_location = [&](std::size_t const i) {
    return new_locations.from_ + static_cast<unsigned>(i);
  };
  auto const get_new_location = [&](location_idx_t const l) {
    return to_idx(l - new_locations.from_);
  };

  auto const stop_rtree = geo::make_point_rtree(
      new_locations,
      [&](location_idx_t const l) { return tt.locations_.coordinates_[l]; });

  auto metas = std::vector<std::vector<footpath>>{};
  metas.resize(new_locations.size());

  pt->status("Stop Neighbors")
      .out_bounds(96.F, 98.F)
      .in_high(new_locations.size());
  utl::parallel_for_run(
      new_locations.size(),
      [&](std::size_t const new_l) {
        auto const l = get_location(new_l);
        auto const pos = tt.locations_.coordinates_[l];
        if (std::abs(pos.lat_) < 2.0 && std::abs(pos.lng_) < 2.0) {
          return;
        }
        auto const dist_lng_degrees = geo::approx_distance_lng_degrees(pos);
        for (auto const& eq :
             stop_rtree.in_radius(pos, config.link_stop_distance_)) {
          auto const neighbor = get_location(eq);
          auto const neighbor_pos = tt.locations_.coordinates_[neighbor];
          auto const dist = std::sqrt(geo::approx_squared_distance(
              pos, neighbor_pos, dist_lng_degrees));
          auto const duration = duration_t{std::max(
              2, static_cast<int>(std::ceil((dist / kWalkSpeed) / 60.0)))};
          metas[get_new_location(l)].emplace_back(neighbor, duration);
        }
      },
      pt->update_fn());

  for (auto const [i, fps] : utl::enumerate(metas)) {
    auto const from = get_location(i);
    for (auto const fp : fps) {
      tt.locations_.equivalences_[from].emplace_back(fp.target());
      tt.locations_.preprocessing_footpaths_out_[from].emplace_back(fp);
      tt.locations_.preprocessing_footpaths_in_[fp.target()].emplace_back(
          footpath{from, fp.duration()});
    }
  }

  if (shapes_data != nullptr) {
    auto const trips = vector_map<gtfs::gtfs_trip_idx_t, gtfs::trip>{};
    auto const shape_states = gtfs::shape_loader_state{};
    gtfs::calculate_shape_offsets_and_bboxes(tt, *shapes_data, shape_states,
                                             trips);
  }

  tt.location_areas_.resize(tt.n_locations());
}

}  // namespace nigiri::loader::netex
