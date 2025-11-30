#include "nigiri/loader/netex/load_timetable.h"

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

#include "nigiri/loader/loader_interface.h"
#include "nigiri/common/sort_by.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

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
        case cista::hash("regionalRail"): return 113;
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
duration_t parse_time(utl::cstr s) {
  if (s.len < 5) {
    return duration_t{0};
  }
  return duration_t{60 * utl::parse_verify<int>(s.substr(0, utl::size(2))) +
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
  for (auto const s :
       doc.select_nodes("//ResourceFrame/organisations/Authority")) {
    auto const n = s.node();
    authorities.emplace(n.attribute("id").as_string(),
                        uniq(authority{.name_ = val(n, "Name"),
                                       .short_name_ = val(n, "ShortName")}));
  }
  return authorities;
}

// =====
// LINES
// -----
struct line {
  std::string_view name_;
  authority const* authority_;
};
using line_map_t = hash_map<std::string_view, std::unique_ptr<line>>;
line_map_t get_lines(pugi::xml_document const& doc,
                     authority_map_t const& authorities) {
  auto lines = line_map_t{};
  for (auto const l : doc.select_nodes("//ServiceFrame/lines/Line")) {
    auto const n = l.node();
    auto const it = authorities.find(ref(n, "AuthorityRef"));
    lines.emplace(
        n.attribute("id").as_string(),
        uniq(line{.name_ = val(n, "Name"),
                  .authority_ =
                      it == end(authorities) ? nullptr : it->second.get()}));
  }
  return lines;
}

// ====================
// DESTINATION DISPLAYS
// --------------------
using destination_display_map_t = hash_map<std::string_view, std::string_view>;
destination_display_map_t get_destination_displays(
    pugi::xml_document const& doc) {
  auto destination_displays = hash_map<std::string_view, std::string_view>{};
  for (auto const display : doc.select_nodes(
           "//ServiceFrame/destinationDisplays/DestinationDisplay")) {
    auto const n = display.node();
    destination_displays.emplace(n.attribute("id").as_string(),
                                 val(n, "FrontText"));
  }
  return destination_displays;
}

// =============
// STOPS & QUAYS
// -------------
struct stop {
  std::string_view parent_;
  std::string_view id_;
  std::string_view name_;
  geo::latlng pos_;
};

using stop_map_t = hash_map<std::string_view, std::unique_ptr<stop>>;

stop_map_t get_stops(pugi::xml_document const& doc) {
  auto stops = stop_map_t{};
  for (auto const s : doc.select_nodes("//SiteFrame/stopPlaces/StopPlace | "
                                       "//SiteFrame/stopPlaces/Quay")) {
    auto const n = s.node();

    auto const get_global_id = [](pugi::xml_node const x) {
      return val(
          x.select_node("keyList/KeyValue/Key[text() = 'GlobalID']").parent(),
          "Value");
    };

    auto const get_pos = [](pugi::xml_node const x) {
      return geo::latlng{
          utl::parse<double>(
              x.select_node("Centroid/Location/Latitude").node().child_value()),
          utl::parse<double>(x.select_node("Centroid/Location/Longitude")
                                 .node()
                                 .child_value())};
    };

    auto const stop_id = n.attribute("id").as_string();
    auto const global_stop_id = get_global_id(n);
    stops.emplace(
        stop_id,
        uniq(stop{.parent_ = {},
                  .id_ = !global_stop_id.empty() ? global_stop_id : stop_id,
                  .name_ = val(n, "Name"),
                  .pos_ = get_pos(n)}));

    for (auto const q : n.select_nodes("quays/Quay")) {
      auto const qn = q.node();
      auto const quay_id = qn.attribute("id").as_string();
      auto const global_quay_id = get_global_id(qn);
      stops.emplace(
          quay_id,
          uniq(stop{.parent_ = stop_id,
                    .id_ = !global_quay_id.empty() ? global_quay_id : quay_id,
                    .name_ = val(qn, "Name"),
                    .pos_ = get_pos(qn)}));
    }
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
               bits.at(static_cast<unsigned>((day - from).count())));
      }
    }

    operating_periods.emplace(sn.attribute("id").as_string(),
                              uniq(std::move(bf)));
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
    auto const op_period = ref(n, "OperatingPeriodRef");
    auto const day_type = ref(n, "DayTypeRef");

    auto const it = operating_periods.find(op_period);
    if (it == end(operating_periods)) {
      log(log_lvl::error, "netex.DayTypeAssignment",
          "OperatingPeriodRef=\"{}\" not found", op_period);
      continue;
    }

    days.emplace(day_type, it->second.get());
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
    std::string_view destination_display_;
    bool in_allowed_;
    bool out_allowed_;
  };
  line const* line_;
  direction_id_t direction_;
  std::vector<stop_point> stop_points_;
};

using journey_pattern_map_t =
    hash_map<std::string_view, std::unique_ptr<journey_pattern>>;

journey_pattern_map_t get_journey_patterns(
    pugi::xml_document const& doc,
    stop_assignment_map_t const& stop_assignments,
    destination_display_map_t const& destination_displays,
    line_map_t const& lines) {

  auto journey_patterns = journey_pattern_map_t{};
  for (auto const j : doc.select_nodes(
           "//ServiceFrame/journeyPatterns/ServiceJourneyPattern")) {
    auto const n = j.node();

    auto stop_points = std::vector<journey_pattern::stop_point>{};
    for (auto const sp : n.child("pointsInSequence").children()) {
      auto const in = val(sp, "ForBoarding");
      auto const out = val(sp, "ForAlighting");
      stop_points.push_back(journey_pattern::stop_point{
          .id_ = sp.attribute("id").as_string(),
          .stop_ = stop_assignments.at(ref(sp, "ScheduledStopPointRef")),
          .destination_display_ =
              destination_displays.at(ref(sp, "DestinationDisplayRef")),
          .in_allowed_ = in.empty() || in == "true"sv,
          .out_allowed_ = out.empty() || out == "true"sv});
    }

    journey_patterns.emplace(
        n.attribute("id").as_string(),
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
  std::uint32_t trip_nr_;
  std::uint16_t route_type_;
  journey_pattern const* journey_pattern_{};
  bitfield const* traffic_days_;
  std::vector<stop_times> stop_times_{};
};

std::vector<service_journey> get_service_journeys(
    pugi::xml_document const& doc,
    day_type_assignment_map_t const& day_type_assignments,
    journey_pattern_map_t const& journey_patterns) {
  auto service_journeys = std::vector<service_journey>{};
  for (auto const s :
       doc.select_nodes("//TimetableFrame/vehicleJourneys/ServiceJourney")) {
    auto const n = s.node();

    auto sj = service_journey{
        .trip_nr_ = utl::parse<std::uint32_t>(val(
            n.select_node("keyList/KeyValue/Key[text() = 'TripNr']").parent(),
            "Value")),
        .route_type_ =
            get_route_type(val(n, "TransportMode"),
                           val(n.child("TransportSubmode"), "RailSubmodule")),
        .journey_pattern_ =
            journey_patterns.at(ref(n, "ServiceJourneyPatternRef")).get(),
        .traffic_days_ = day_type_assignments.at(
            n.select_node("dayTypes/DayTypeRef/@ref").attribute().as_string())};
    auto const passing_times =
        n.select_nodes("passingTimes/TimetabledPassingTime");
    std::cout << "DEBUG: passing_times size: " << passing_times.size()
              << ", stop_points size: "
              << sj.journey_pattern_->stop_points_.size() << std::endl;
    for (auto const [passing_time, stop_point] :
         utl::zip(passing_times, sj.journey_pattern_->stop_points_)) {
      auto const pn = passing_time.node();
      auto const stop_point_id = ref(pn, "StopPointInJourneyPatternRef");

      utl::verify(stop_point.id_ == stop_point_id,
                  "expected pointsInSequence.StopPointInJourneyPattern.id={}, "
                  "got TimetabledPassingTime.StopPointInJourneyPatternRef={}",
                  stop_point.id_, stop_point_id);

      sj.stop_times_.push_back({.arr_ = parse_time(val(pn, "ArrivalTime")),
                                .dep_ = parse_time(val(pn, "DepartureTime"))});
    }
  }
  return service_journeys;
}

bool is_xml_file(fs::path const& p) {
  return p.extension() == ".xml" || p.extension() == ".XML";
}

cista::hash_t hash(dir const& d) {
  if (d.type() == dir_type::kZip) {
    return d.hash();
  }

  auto h = std::uint64_t{0U};
  auto const hash_file = [&](fs::path const& p) {
    if (!d.exists(p)) {
      h = wyhash64(h, _wyp[0]);
    } else {
      auto const f = d.get_file(p);
      auto const data = f.data();
      h = wyhash(data.data(), data.size(), h, _wyp);
    }
  };

  for (auto const& f : d.list_files("/")) {
    if (is_xml_file(f)) {
      hash_file(f);
    }
  }

  return h;
}

bool applicable(dir const& d) {
  return utl::any_of(d.list_files("/"), is_xml_file);
}

void load_timetable(loader_config const&,
                    source_idx_t const,
                    dir const& d,
                    timetable& tt,
                    hash_map<bitfield, bitfield_idx_t>& /*bitfield_indices*/,
                    assistance_times* /*assistance*/,
                    shapes_storage* /*shapes_data*/) {
  auto const global_timer = nigiri::scoped_timer{"netex parser"};

  auto const xml_files =
      utl::all(d.list_files(""))  //
      | utl::remove_if([&](fs::path const& f) { return !is_xml_file(f); })  //
      | utl::vec();

  auto const pt = utl::get_active_progress_tracker();
  pt->status("Parse Files").out_bounds(0.F, 100.F).in_high(xml_files.size());

  struct intermediate {
    file f_;
    pugi::xml_document doc_;
    stop_map_t stops_;
    journey_pattern_map_t journey_patterns_;
    operating_period_map_t operating_periods_;
    line_map_t lines_;
    authority_map_t authorities_;
    std::vector<service_journey> service_journeys_;
  };

  utl::parallel_ordered_collect_threadlocal<none>(
      xml_files.size(),
      [&](none&, std::size_t const i) {
        auto f = d.get_file(xml_files.at(i));
        auto doc = pugi::xml_document{};
        auto const result =
            f.is_mutable() ? doc.load_buffer_inplace(
                                 f.get_mutable(), f.size(),
                                 pugi::parse_default | pugi::parse_trim_pcdata)
                           : doc.load_buffer(
                                 f.data().data(), f.size(),
                                 pugi::parse_default | pugi::parse_trim_pcdata);
        utl::verify(result, "Unable to parse XML buffer: {} at offset {}",
                    result.description(), result.offset);

        auto im = intermediate{};
        im.authorities_ = get_authorities(doc);
        im.stops_ = get_stops(doc);
        im.lines_ = get_lines(doc, im.authorities_);
        im.journey_patterns_ =
            get_journey_patterns(doc, get_stop_assignments(doc, im.stops_),
                                 get_destination_displays(doc), im.lines_);
        im.operating_periods_ =
            get_operating_periods(doc, tt.internal_interval_days());
        im.service_journeys_ = get_service_journeys(
            doc, get_day_type_assignments(doc, im.operating_periods_),
            im.journey_patterns_);
        im.f_ = std::move(f);
        im.doc_ = std::move(doc);
        return im;
      },
      [](std::size_t const i, intermediate&& im) {
        CISTA_UNUSED_PARAM(i)
        CISTA_UNUSED_PARAM(im)
      },
      pt->update_fn());
}

}  // namespace nigiri::loader::netex
