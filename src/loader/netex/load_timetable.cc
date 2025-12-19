#include "nigiri/loader/netex/load_timetable.h"

#include "nigiri/loader/get_index.h"

#include <filesystem>
#include <ranges>
#include <string>

#include "fmt/std.h"

#include "cista/hash.h"
#include "cista/mmap.h"

#include "wyhash.h"

#include "pugixml.hpp"

#include "utl/concat.h"
#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"
#include "utl/parser/arg_parser.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"
#include "utl/visit.h"

#include "nigiri/loader/gtfs/route_key.h"
#include "nigiri/loader/gtfs/seated.h"
#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/shape_prepare.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/loader/netex/utc_trip.h"
#include "nigiri/common/gunzip.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"

using namespace std::string_view_literals;
namespace fs = std::filesystem;

namespace nigiri::loader::netex {

// =====
// UTILS
// -----
std::string_view operator||(std::string_view a, std::string_view b) {
  return a.empty() ? b : a;
}

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

bool is_true_or_empty(std::string_view s) { return s.empty() || s == "true"sv; }

struct none {};

template <typename Map>
struct lookup {
  using key_t = typename Map::key_type;
  using value_t = typename Map::mapped_type;

  template <typename K>
  value_t const& at(K const& k) const {
    auto const it = timetable_.find(k);
    if (it != end(timetable_)) {
      return it->second;
    }
    try {
      return base_.at(k);
    } catch (std::exception const&) {
      throw std::runtime_error{fmt::format(
          "{:?} key {:?} not found\nBASE:\n\t{:?}\nTIMETABLE:\n\t{:?}\n",
          cista::type_str<value_t>(), k,
          fmt::join(base_ | std::views::keys, "\n\t"),
          fmt::join(timetable_ | std::views::keys, "\n\t"))};
    }
  }

  template <typename K>
  std::optional<value_t const*> find(K const& k) const {
    if (auto const it = timetable_.find(k); it != end(timetable_)) {
      return std::optional{&it->second};
    }

    if (auto const it = base_.find(k); it != end(base_)) {
      return std::optional{&it->second};
    }

    return std::nullopt;
  }

  hash_map<key_t, value_t> const& base_;
  hash_map<key_t, value_t>& timetable_;
};

std::string_view ref(pugi::xml_node n, char const* child) {
  auto const str = n.child(child).attribute("ref").as_string();
  return str == nullptr ? std::string_view{} : std::string_view{str};
}

std::string_view val(pugi::xml_node n, char const* child) {
  auto const str = n.child_value(child);
  return str == nullptr ? std::string_view{} : std::string_view{str};
}

std::string_view id(pugi::xml_node n) {
  auto const str = n.attribute("id").as_string();
  return str == nullptr ? std::string_view{} : std::string_view{str};
}

geo::latlng get_pos(pugi::xml_node const x) {
  auto const loc = x.child("Location");
  auto const gml_pos = val(loc, "gml:pos");
  if (!gml_pos.empty()) {
    auto s = utl::cstr{gml_pos};
    auto ret = geo::latlng{};
    utl::parse_fp(s, ret.lat_);
    if (s) {
      ++s;
    }
    utl::parse_fp(s, ret.lng_);
    return ret;
  }
  return geo::latlng{utl::parse<double>(val(loc, "Latitude")),
                     utl::parse<double>(val(loc, "Longitude"))};
}

std::uint16_t get_route_type(pugi::xml_node const n) {
  auto const transport_mode = val(n, "TransportMode");
  auto const submode = n.child("TransportSubmode");
  switch (cista::hash(transport_mode)) {
    case cista::hash("coach"):
      switch (cista::hash(val(submode, "CoachSubmode"))) {
        case cista::hash("internationalCoach"): return 201;
        case cista::hash("nationalCoach"): return 202;
        case cista::hash("shuttleCoach"): return 203;
        case cista::hash("regionalCoach"): return 204;
        case cista::hash("specialCoach"): return 205;
        case cista::hash("sightseeingCoach"): return 206;
        case cista::hash("touristCoach"): return 207;
        case cista::hash("commuterCoach"): return 208;
        default: return 200;
      }

    case cista::hash("bus"):
      switch (cista::hash(val(submode, "BusSubmode"))) {
        case cista::hash("regionalBus"): return 701;
        case cista::hash("expressBus"): return 702;
        case cista::hash("localBus"): return 704;
        case cista::hash("mobilityBusForRegisteredDisabled"): return 709;
        case cista::hash("sightseeingBus"): return 710;
        case cista::hash("shuttleBus"): return 711;
        case cista::hash("schoolBus"): return 712;
        case cista::hash("schoolAndPublicServiceBus"): return 713;
        case cista::hash("railReplacementBus"): return 714;
        case cista::hash("demandAndResponseBus"): return 715;
        default: return 700;
      }

    // Bus with two overhead wires using spring-loaded trolley poles.
    case cista::hash("trolleyBus"): return 11;

    case cista::hash("tram"):
      switch (cista::hash(val(submode, "TramSubmode"))) {
        case cista::hash("cityTram"): return 901;
        case cista::hash("localTram"): return 902;
        case cista::hash("regionalTram"): return 903;
        case cista::hash("sightseeingTram"): return 904;
        case cista::hash("shuttleTram"): return 905;
        default: return 900;
      }

    case cista::hash("rail"):
      switch (cista::hash(val(submode, "RailSubmode"))) {
        case cista::hash("highSpeedRail"): return 101;
        case cista::hash("rackAndPinionRailway"): return 1400;  // ?
        case cista::hash("regionalRail"): return 106;
        case cista::hash("interregionalRail"): return 103;
        case cista::hash("crossCountryRail"):
        case cista::hash("longDistance"):
        case cista::hash("international"): return 102;
        case cista::hash("sleeperRailService"):
        case cista::hash("nightRail"): return 105;
        case cista::hash("carTransportRailService"): return 104;
        case cista::hash("touristRailway"): return 107;
        case cista::hash("airportLinkRail"):
        case cista::hash("railShuttle"): return 108;
        case cista::hash("suburbanRailway"): return 109;
        case cista::hash("replacementRailService"): return 110;  // bus?
        case cista::hash("specialTrain"): return 111;
        default: return 100;
      }

    // DEPRECATED. Use mode rail with RailSubmode longDistance.
    case cista::hash("intercityRail"): return 102;

    // DEPRECATED. Use mode metro with MetroSubmode urbanRail (or in some few
    // cases rail with RailSubmode local).
    case cista::hash("urbanRail"):
    case cista::hash("metro"):
      // Within an urban area. For underground and railway.
      switch (cista::hash(val(submode, "MetroSubmode"))) {
        case cista::hash("metro"): return 401;
        case cista::hash("tube"): return 402;
        case cista::hash("urbanRailway"): return 400;
        default: return 401;
      }

    // All air related mode. No special distinction is made.
    case cista::hash("air"): return 1100;

    // Most water related modes. The only specialisation is the mode ferry. In
    // Transmodel also ship was used for this mode.
    case cista::hash("water"): return 1000;

    // Can be only two cabines or multiple.
    case cista::hash("cableway"):
    case cista::hash("telecabin"):
      switch (cista::hash(val(submode, "TelecabinSubmode"))) {
        case cista::hash("telecabin"): return 1301;
        case cista::hash("cableCar"): return 1302;
        case cista::hash("lift"): return 1307;
        case cista::hash("chairLift"): return 1304;
        case cista::hash("dragLift"): return 1305;
        default: return 1300;
      }

    // Cable railway on steep slope using two counterbalanced carriages.
    case cista::hash("funicular"): return 1400;

    case cista::hash("taxi"):
      switch (cista::hash(val(submode, "TaxiSubmode"))) {
        case cista::hash("communalTaxi"): return 1501;
        case cista::hash("waterTaxi"): return 1502;
        case cista::hash("railTaxi"): return 1503;
        case cista::hash("bikeTaxi"): return 1504;
        case cista::hash("allTaxiServices"): return 1507;
        default: return 1500;
      }

    // Ferry can be passenger ferries and/or car ferries. The specialisation
    // from water is the detailed schedule and usually the very rigid pattern
    // with only a few stops on the route.
    case cista::hash("ferry"): return 1000;

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

std::uint16_t get_more_precise_route_type(std::uint16_t const a,
                                          std::uint16_t const b) {
  if (a == 1700) {
    return b;
  } else if (b == 1700) {
    return a;
  } else if (a % 100 == 0 || a < 100) {
    return b;
  } else if (b % 100 == 0 || b < 100) {
    return a;
  }
  return a;
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
  std::string_view id_;
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
    operators.emplace(id(n), uniq(operätor{.id_ = id(n),
                                           .code_ = val(n, "PublicCode"),
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

// ========
// PRODUCTS
// --------
using product_map_t = hash_map<std::string_view, std::string_view>;

product_map_t get_products(pugi::xml_document const& doc) {
  auto products = product_map_t{};
  products.emplace("", "");
  for (auto const v : doc.select_nodes("//ResourceFrame/typesOfValue/ValueSet/"
                                       "values/TypeOfProductCategory")) {
    auto const n = v.node();
    products.emplace(id(n), val(n, "ShortName"));
  }
  return products;
}

// =====
// LINES
// -----
struct line {
  struct route_key {
    std::uint16_t route_type_;
    operätor const* operator_;
  };
  hash_map<route_key, route_id_idx_t> routes_{};
  std::string_view id_;
  std::string_view name_;
  std::string_view product_;
  authority const* authority_;
  operätor const* operator_;
  std::uint16_t route_type_;
  route_color color_;
};
using line_map_t = hash_map<std::string_view, std::unique_ptr<line>>;
line_map_t get_lines(pugi::xml_document const& doc,
                     lookup<authority_map_t> const& authorities,
                     lookup<operator_map_t> const& operators,
                     lookup<product_map_t> const& products) {
  auto lines = line_map_t{};
  lines.emplace("", uniq(line{
                        .id_ = "",
                        .name_ = "",
                        .product_ = "",
                        .authority_ = authorities.at(std::string_view{}).get(),
                        .operator_ = operators.at(std::string_view{}).get(),
                        .route_type_ = {},
                        .color_ = route_color{},
                    }));
  for (auto const l : doc.select_nodes("//ServiceFrame/lines/Line")) {
    auto const n = l.node();
    auto const ppt = n.child("Presentation");
    lines.emplace(
        id(n),
        uniq(line{
            .id_ = id(n),
            .name_ = val(n, "Name"),
            .product_ = products.at(ref(n, "TypeOfProductCategoryRef")),
            .authority_ = authorities.at(ref(n, "AuthorityRef")).get(),
            .operator_ =
                operators.at(ref(n.child("additionalOperators"), "OperatorRef"))
                    .get(),
            .route_type_ = get_route_type(n),
            .color_ = {.color_ = to_color(val(ppt, "Colour")),
                       .text_color_ = to_color(val(ppt, "TextColour"))},
        }));
  }
  return lines;
}

// ====================
// DESTINATION DISPLAYS
// --------------------
struct destination_display {
  translation_idx_t trip_direction_{translation_idx_t::invalid()};
  std::string_view direction_;
};
using destination_display_map_t =
    hash_map<std::string_view, std::unique_ptr<destination_display>>;
destination_display_map_t get_destination_displays(
    pugi::xml_document const& doc) {
  auto destination_displays = destination_display_map_t{};
  destination_displays.emplace("", uniq(destination_display{.direction_ = ""}));
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
  std::string_view public_code_;
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
          x.select_node(
               R"(keyList/KeyValue/Key[text() = 'GlobalID' or text() = 'SLOID'])")
              .parent(),
          "Value");
    };

    auto const stop_id = id(n);
    auto const global_stop_id = get_global_id(n);
    auto const parent =
        stops
            .emplace(stop_id, uniq(stop{.id_ = global_stop_id || stop_id,
                                        .name_ = val(n, "Name"),
                                        .public_code_ = {},
                                        .pos_ = get_pos(n.child("Centroid"))}))
            .first->second.get();

    auto const parent_ref = ref(n, "ParentSiteRef");
    if (!parent_ref.empty()) {
      parents.emplace(parent, parent_ref);
    }

    auto parent_added = false;
    for (auto const q : n.select_nodes("quays/Quay")) {
      auto const qn = q.node();
      auto const quay_id = id(qn);
      auto const global_quay_id = get_global_id(qn);
      auto const pos = get_pos(qn.child("Centroid"));
      auto const name = val(qn, "Name");
      stops.emplace(
          quay_id,
          uniq(stop{.parent_ = parent,
                    .id_ = !global_quay_id.empty() ? global_quay_id : quay_id,
                    .name_ = name.empty() ? parent->name_ : name,
                    .public_code_ = val(qn, "PublicCode"),
                    .pos_ = pos == geo::latlng{} ? parent->pos_ : pos}));

      // hack for CH
      constexpr auto kSloidPrefix = "ch:1:sloid:"sv;
      if (global_quay_id.starts_with(kSloidPrefix) && !parent_added) {
        auto const x = global_quay_id.substr(kSloidPrefix.length());
        if (auto const end = x.find(':'); end != std::string_view::npos) {
          auto const parent_id =
              global_quay_id.substr(0, kSloidPrefix.size() + end);
          stops.emplace(parent_id, uniq(stop{.id_ = parent_id,
                                             .name_ = parent->name_,
                                             .public_code_ = {},
                                             .pos_ = parent->pos_}));
          parent_added = true;
        }
      }
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
                                           lookup<stop_map_t> all_stops) {
  auto stop_assigments = hash_map<std::string_view, stop const*>{};
  for (auto const& a : doc.select_nodes("//ServiceFrame/stopAssignments/"
                                        "PassengerStopAssignment")) {
    auto const n = a.node();
    auto const sstop = ref(n, "ScheduledStopPointRef");
    auto const quay = ref(n, "QuayRef");
    auto const stop_place = ref(n, "StopPlaceRef");
    if (!sstop.empty() && (!quay.empty() || !stop_place.empty())) {
      auto const& [base, timetable] = all_stops;

      auto const get = [](stop_map_t const& map,
                          std::string_view key) -> std::optional<stop const*> {
        auto const it = map.find(key);
        return it == end(map) ? std::nullopt : std::optional{it->second.get()};
      };

      auto const s = std::optional<stop const*>{}
                         .or_else([&]() { return get(timetable, quay); })
                         .or_else([&]() { return get(timetable, stop_place); })
                         .or_else([&]() { return get(base, quay); })
                         .or_else([&]() { return get(base, stop_place); })
                         .value_or(nullptr);

      if (s != nullptr) {
        stop_assigments[sstop] = s;
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

  auto always = bitfield{};
  always.one_out();
  operating_periods.emplace("", uniq(std::move(always)));

  for (auto const s : doc.select_nodes(
           "//ServiceCalendarFrame//operatingPeriods/UicOperatingPeriod |"
           "//ServiceCalendarFrame/validityConditions/AvailabilityCondition")) {
    auto const sn = s.node();
    auto const from = parse_date(val(sn, "FromDate"));
    auto const to = parse_date(val(sn, "ToDate"));
    auto const bits = val(sn, "ValidDayBits");

    utl::verify((to - from).count() + 1 <= static_cast<int>(bits.size()),
                "from={}, to={} => n_days={} != n_bits={}", from, to,
                (to - from).count() + 1, bits.size());

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
    lookup<operating_period_map_t> operating_periods) {
  auto days = day_type_assignment_map_t{};
  for (auto const x :
       doc.select_nodes("//ServiceCalendarFrame//dayTypeAssignments/"
                        "DayTypeAssignment")) {
    auto const n = x.node();
    days.emplace(ref(n, "DayTypeRef"),
                 operating_periods.at(ref(n, "OperatingPeriodRef")).get());
  }
  return days;
}

// =========
// TRAIN NRS
// ---------
using train_nr_map_t = hash_map<std::string_view, unsigned>;
train_nr_map_t get_train_numbers(pugi::xml_document const& doc) {
  auto train_nrs = train_nr_map_t{};
  for (auto const x :
       doc.select_nodes("//TimetableFrame/trainNumbers/TrainNumber")) {
    auto const n = x.node();
    train_nrs.emplace(id(n),
                      utl::parse<unsigned>(n.child_value("ForAdvertisement")));
  }
  return train_nrs;
}

// =======
// Notices
// -------
struct notice {
  attribute_idx_t idx_{attribute_idx_t::invalid()};
  std::string_view code_;
  std::vector<translation> translations_;
};

using notice_map_t = hash_map<std::string_view, std::unique_ptr<notice>>;

notice_map_t get_notices(pugi::xml_document const& doc) {
  auto notices = notice_map_t{};
  notices.emplace(
      "", uniq(notice{.code_ = "", .translations_ = {translation{"", ""}}}));
  // DE-DELFI
  // ServiceFrame/journeyPatterns/ServiceJourneyPattern/StopPointInJourneyPattern/NoticeAssignment/Notice
  // CH-SKI
  // ServiceFrame/notices/Notice
  for (auto const x : doc.select_nodes("//Notice")) {
    auto const n = x.node();
    if (!is_true_or_empty(val(n, "CanBeAdvertised"))) {
      continue;
    }

    auto const parse_text = [](pugi::xml_node const notice) {
      auto const text = notice.child("Text");
      return translation{text.attribute("lang").as_string(),
                         text.child_value()};
    };

    auto translations = std::vector<translation>{parse_text(n)};
    for (auto const alt :
         n.child("alternativeTexts").children("AlternativeText")) {
      translations.push_back(parse_text(alt));
    }

    notices.emplace(id(n),
                    uniq(notice{.code_ = val(n, "ShortCode"),
                                .translations_ = std::move(translations)}));
  }
  return notices;
}

std::vector<notice const*> get_notice_assignments(lookup<notice_map_t> notices,
                                                  pugi::xml_node n) {
  return utl::all(n.child("noticeAssignments").children("NoticeAssignment"))  //
         | utl::transform([&](pugi::xml_node const na) {
             return notices.find(ref(na, "NoticeRef"));
           })  //
         | utl::remove_if([](auto&& opt) { return !opt.has_value(); })  //
         | utl::transform([](auto&& opt) -> notice const* {
             return opt.value()->get();
           })  //
         | utl::vec();
}

// ===============
// JOURNEY PATTERN
// ---------------
struct journey_pattern {
  struct stop_point {
    std::string_view id_{};
    stop const* stop_;
    destination_display const* destination_display_;
    bool in_allowed_;
    bool out_allowed_;
    std::vector<notice const*> notices_;
  };
  line* line_;
  direction_id_t direction_;
  std::vector<stop_point> stop_points_;
};

using journey_pattern_map_t =
    hash_map<std::string_view, std::unique_ptr<journey_pattern>>;

journey_pattern_map_t get_journey_patterns(
    pugi::xml_document const& doc,
    lookup<stop_assignment_map_t> stop_assignments,
    lookup<destination_display_map_t> destination_displays,
    lookup<line_map_t> lines,
    lookup<stop_map_t> stops,
    lookup<notice_map_t> notices) {
  auto journey_patterns = journey_pattern_map_t{};

  auto const get_stop =
      [&](std::string_view stop_point_ref) -> netex::stop const* {
    auto const timetable_it = stop_assignments.timetable_.find(stop_point_ref);
    if (timetable_it != end(stop_assignments.timetable_)) {
      return timetable_it->second;
    }

    auto const base_it = stop_assignments.base_.find(stop_point_ref);
    if (base_it != end(stop_assignments.base_)) {
      return base_it->second;
    }

    // Invalid - fall back to information from ScheduledStopPoint
    return utl::get_or_create(
               stops.timetable_, stop_point_ref,
               [&]() {
                 auto const stop_point = doc.select_node(
                     fmt::format("//ServiceFrame/scheduledStopPoints/"
                                 "ScheduledStopPoint[@id='{}']",
                                 stop_point_ref)
                         .c_str());
                 return uniq(stop{
                     .id_ = stop_point_ref,
                     .name_ = val(stop_point.node(), "Name"),
                     .public_code_ = val(stop_point.node(), "PublicCode"),
                     .pos_ = get_pos(stop_point.node()),
                 });
               })
        .get();
  };

  for (auto const j : doc.select_nodes(
           "//ServiceFrame/journeyPatterns/ServiceJourneyPattern")) {
    auto const n = j.node();

    auto stop_points = std::vector<journey_pattern::stop_point>{};
    for (auto const sp : n.child("pointsInSequence").children()) {
      stop_points.push_back({
          .id_ = id(sp),
          .stop_ = get_stop(ref(sp, "ScheduledStopPointRef")),
          .destination_display_ =
              destination_displays.at(ref(sp, "DestinationDisplayRef")).get(),
          .in_allowed_ = is_true_or_empty(val(sp, "ForBoarding")),
          .out_allowed_ = is_true_or_empty(val(sp, "ForAlighting")),
          .notices_ = get_notice_assignments(notices, sp),
      });
    }

    if (stop_points.size() < 2U) {
      continue;
    }

    journey_patterns.emplace(
        id(n),
        uniq(journey_pattern{
            .line_ = lines.at(ref(n.child("RouteView"), "LineRef")).get(),
            .direction_ =
                direction_id_t{ref(n, "DirectionRef").ends_with("1::") ? 0 : 1},
            .stop_points_ = std::move(stop_points),
        }));
  }

  return journey_patterns;
}

// ==============
// Meeting Points
// --------------
struct journey_meeting {
  friend bool operator==(journey_meeting const& a, journey_meeting const& b) {
    return a.to_journey_id_ == b.to_journey_id_;
  }
  cista::hash_t hash() const { return cista::hash(to_journey_id_); }

  std::string to_journey_id_;
  bitfield const* bitfield_;
  std::string_view stop_;
};

using journey_meeting_map_t =
    hash_map<std::string /* ServiceJourney.id */, hash_set<journey_meeting>>;

journey_meeting_map_t get_journey_meetings(
    pugi::xml_document const& doc,
    lookup<operating_period_map_t> operating_periods) {
  auto journey_meetings = journey_meeting_map_t{};
  for (auto const x : doc.select_nodes(
           "//TimetableFrame/journeyMeetings/JourneyMeeting | "
           "//TimetableFrame/journeyInterchanges/ServiceJourneyInterchange")) {
    auto const n = x.node();
    if (!is_true_or_empty(val(n, "StaySeated"))) {
      continue;
    }
    auto const m = journey_meeting{
        .to_journey_id_ = std::string{ref(n, "ToJourneyRef")},
        .bitfield_ = operating_periods
                         .at(ref(n.child("validityConditions"),
                                 "AvailabilityConditionRef"))
                         .get(),
        .stop_ = ref(n, "FromPointRef") || ref(n, "AtStopPointRef"),
    };
    journey_meetings[ref(n, "FromJourneyRef")].insert(m);
  }
  return journey_meetings;
}

// ===============
// SERVICE_JOURNEY
// ---------------
struct service_journey {
  struct stop_times {
    duration_t arr_, dep_;
  };

  journey_pattern const* get_journey_pattern() const {
    return utl::visit(
        journey_pattern_,  //
        [](journey_pattern const& jp) { return &jp; },
        [](journey_pattern const* jp) { return jp; });
  }

  std::string_view id_;
  std::string_view branding_ref_;  // ref because SNCF ref links to nothing
  std::uint32_t trip_nr_;
  std::uint16_t route_type_;
  vehicle_type const* vehicle_type_{};
  std::variant<journey_pattern const*, journey_pattern> journey_pattern_{};
  bitfield const* traffic_days_;
  operätor const* operator_;
  std::vector<stop_times> stop_times_{};
  ptrdiff_t dbg_offset_;
};

gtfs::stop_seq_t stop_seq(service_journey const& sj) {
  auto seq = gtfs::stop_seq_t{};
  for (auto const& x : sj.get_journey_pattern()->stop_points_) {
    seq.push_back(nigiri::stop{x.stop_->location_, x.in_allowed_,
                               x.out_allowed_, x.in_allowed_, x.out_allowed_}
                      .value());
  }
  return seq;
}

std::vector<service_journey> get_service_journeys(
    pugi::xml_document const& doc,
    lookup<stop_assignment_map_t> stop_assignments,
    lookup<operating_period_map_t> operating_periods,
    lookup<day_type_assignment_map_t> day_type_assignments,
    lookup<destination_display_map_t> destination_displays,
    lookup<vehicle_type_map_t> vehicle_types,
    lookup<line_map_t> lines,
    lookup<operator_map_t> operators,
    lookup<notice_map_t> notices,
    lookup<journey_pattern_map_t> journey_patterns,
    train_nr_map_t const& train_nrs) {
  auto service_journeys = std::vector<service_journey>{};
  for (auto const s :
       doc.select_nodes("//TimetableFrame/vehicleJourneys/ServiceJourney")) {
    auto const n = s.node();
    auto const train_nr_ref = ref(n.child("trainNumbers"), "TrainNumberRef");
    auto sj = service_journey{
        .id_ = id(n),
        .branding_ref_ = ref(n, "BrandingRef"),
        .trip_nr_ =
            train_nr_ref.empty()
                ? utl::parse<std::uint32_t>(val(
                      n.select_node("keyList/KeyValue/Key[text() = 'TripNr']")
                          .parent(),
                      "Value"))
                : train_nrs.at(train_nr_ref),
        .route_type_ = get_route_type(n),
        .vehicle_type_ = vehicle_types.at(ref(n, "VehicleTypeRef")).get(),
        .journey_pattern_ =
            n.child("ServiceJourneyPatternRef")  // DE-DELFI
                ? journey_patterns.at(ref(n, "ServiceJourneyPatternRef")).get()
                : n.child("JourneyPatternRef")  // FR-SNCF
                      ? journey_patterns.at(ref(n, "JourneyPatternRef")).get()
                      : nullptr,
        .traffic_days_ = n.child("validityConditions")
                             ? operating_periods
                                   .at(ref(n.child("validityConditions"),
                                           "AvailabilityConditionRef"))
                                   .get()
                             : day_type_assignments.at(
                                   n.select_node("dayTypes/DayTypeRef/@ref")
                                       .attribute()
                                       .as_string()),
        .operator_ = operators.at(ref(n, "OperatorRef")).get(),
        .dbg_offset_ = n.offset_debug()};

    auto const calls = n.child("calls");
    if (calls) {
      auto jp = journey_pattern{
          .line_ = lines.at(ref(n, "LineRef")).get(),
          .direction_ =
              direction_id_t{val(n, "DirectionType") == "outbound"sv ? 1 : 2},
          .stop_points_ = {},
      };
      auto const notice_assignments = get_notice_assignments(notices, n);
      for (auto const call : calls.children("Call")) {
        // CH-SKI
        auto const arr = call.child("Arrival");
        auto const dep = call.child("Departure");
        jp.stop_points_.push_back({
            .stop_ = stop_assignments.at(ref(call, "ScheduledStopPointRef")),
            .destination_display_ =
                destination_displays.at(ref(call, "DestinationDisplayRef"))
                    .get(),
            .in_allowed_ = is_true_or_empty(val(dep, "ForBoarding")),
            .out_allowed_ = is_true_or_empty(val(arr, "ForAlighting")),
            .notices_ = utl::merge(notice_assignments,
                                   get_notice_assignments(notices, call)),
        });
        sj.stop_times_.push_back({
            .arr_ = parse_time(val(arr, "Time"), val(arr, "DayOffset")),
            .dep_ = parse_time(val(dep, "Time"), val(dep, "DayOffset")),
        });
      }
      sj.journey_pattern_ = std::move(jp);
    } else {
      // DE-DELFI / FR-SNCF
      utl::verify(
          std::holds_alternative<journey_pattern const*>(sj.journey_pattern_) &&
              std::get<journey_pattern const*>(sj.journey_pattern_) != nullptr,
          "journey pattern required");

      auto const& jp = *std::get<journey_pattern const*>(sj.journey_pattern_);

      for (auto const [passing_time, stop_point] :
           utl::zip(n.select_nodes("passingTimes/TimetabledPassingTime"),
                    jp.stop_points_)) {
        auto const pn = passing_time.node();

        auto stop_point_id = ref(pn, "StopPointInJourneyPatternRef");  // DELFI
        if (stop_point_id.empty()) {
          stop_point_id = ref(pn, "PointInJourneyPatternRef");  // SNCF
        }

        utl::verify(
            stop_point.id_ == stop_point_id,
            "expected pointsInSequence.StopPointInJourneyPattern.id={}, "
            "got TimetabledPassingTime.StopPointInJourneyPatternRef={}",
            stop_point.id_, stop_point_id);

        sj.stop_times_.push_back(
            {.arr_ = parse_time(val(pn, "ArrivalTime"),
                                val(pn, "ArrivalDayOffset")),
             .dep_ = parse_time(val(pn, "DepartureTime"),
                                val(pn, "DepartureDayOffset"))});
      }
    }

    if (sj.get_journey_pattern()->stop_points_.size() < 2U) {
      continue;
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
      k.utc_times_.push_back(to_utc(day, sj.stop_times_[dep].dep_) -
                             first_dep_day_offset);
      k.utc_times_.push_back(to_utc(day, sj.stop_times_[arr].arr_) -
                             first_dep_day_offset);
    }

    auto pred = minutes_after_midnight_t{0U};
    for (auto& x : k.utc_times_) {
      x = std::max(pred, x);
      pred = x;
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
bool has_gz_extension(fs::path const& x) {
  return x.extension() == ".gz" || x.extension() == ".GZ";
}

bool is_xml_file(fs::path const& p) {
  auto const has_xml_extension = [](fs::path const& x) {
    return x.extension() == ".xml" || x.extension() == ".XML";
  };
  return has_xml_extension(p) ||
         (has_gz_extension(p) && has_xml_extension(p.stem()));
}

bool applicable(dir const& d) {
  return utl::any_of(d.list_files("."), is_xml_file);
}

struct intermediate {
  void steal(intermediate& o) {
    auto const merge = [](auto& target, auto& source) {
      for (auto& [k, v] : source) {
        target.emplace(k, std::move(v));
      }
      source = {};
    };

    merge(notices_, o.notices_);
    merge(stops_, o.stops_);
    merge(vehicle_types_, o.vehicle_types_);
    merge(journey_patterns_, o.journey_patterns_);
    merge(operating_periods_, o.operating_periods_);
    merge(day_type_assignments_, o.day_type_assignments_);
    merge(stop_assignments_, o.stop_assignments_);
    merge(destination_displays_, o.destination_displays_);
    merge(products_, o.products_);
    merge(lines_, o.lines_);
    merge(authorities_, o.authorities_);
    merge(operators_, o.operators_);
  }

  file f_;
  std::string unzip_buf_;
  pugi::xml_document doc_;
  date::time_zone const* tz_;
  notice_map_t notices_;
  stop_map_t stops_;
  vehicle_type_map_t vehicle_types_;
  journey_pattern_map_t journey_patterns_;
  operating_period_map_t operating_periods_;
  day_type_assignment_map_t day_type_assignments_;
  stop_assignment_map_t stop_assignments_;
  destination_display_map_t destination_displays_;
  journey_meeting_map_t journey_meetings_;
  product_map_t products_;
  line_map_t lines_;
  authority_map_t authorities_;
  operator_map_t operators_;
  std::vector<service_journey> service_journeys_;
};

std::optional<intermediate> get_intermediate(intermediate const& base,
                                             timetable const& tt,
                                             dir const& d,
                                             fs::path const& path,
                                             std::string const& default_tz) {
  auto im = intermediate{};
  try {
    auto f = d.get_file(path);
    auto doc = pugi::xml_document{};

    auto const opt = pugi::parse_default | pugi::parse_trim_pcdata;
    auto parse_result = pugi::xml_parse_result{};
    if (has_gz_extension(path)) {
      im.unzip_buf_ = gunzip({f.data().data(), f.size()});
      parse_result = doc.load_buffer_inplace(im.unzip_buf_.data(),
                                             im.unzip_buf_.size(), opt);
    } else if (f.is_mutable()) {
      parse_result = doc.load_buffer_inplace(f.get_mutable(), f.size(), opt);
    } else {
      parse_result = doc.load_buffer(f.data().data(), f.size(), opt);
    }

    utl::verify(parse_result, "Unable to parse XML buffer {}: {} at offset {}",
                path, parse_result.description(), parse_result.offset);

    try {
      im.tz_ = date::locate_zone(
          utl::parse<int>(
              doc.select_node(
                     "//FrameDefaults/DefaultLocale/SummerTimeZoneOffset")
                  .node()
                  .child_value()) == 2
              ? "CET"
              : doc.select_node("//FrameDefaults/DefaultLocale/TimeZone")
                    .node()
                    .child_value());
    } catch (...) {
      im.tz_ = date::locate_zone(default_tz);
    }
    im.notices_ = get_notices(doc);
    im.authorities_ = get_authorities(doc);
    im.operators_ = get_operators(doc);
    im.stops_ = get_stops(doc);
    im.products_ = get_products(doc);
    im.lines_ = get_lines(doc, {base.authorities_, im.authorities_},
                          {base.operators_, im.operators_},
                          {base.products_, im.products_});
    im.destination_displays_ = get_destination_displays(doc);
    im.stop_assignments_ = get_stop_assignments(doc, {base.stops_, im.stops_});
    im.journey_patterns_ = get_journey_patterns(
        doc, {base.stop_assignments_, im.stop_assignments_},
        {base.destination_displays_, im.destination_displays_},
        {base.lines_, im.lines_}, {base.stops_, im.stops_},
        {base.notices_, im.notices_});
    im.operating_periods_ =
        get_operating_periods(doc, tt.internal_interval_days());
    im.vehicle_types_ = get_vehicle_types(doc);
    im.day_type_assignments_ = get_day_type_assignments(
        doc, {base.operating_periods_, im.operating_periods_});
    im.journey_meetings_ = get_journey_meetings(
        doc, {base.operating_periods_, im.operating_periods_});
    im.service_journeys_ = get_service_journeys(
        doc, {base.stop_assignments_, im.stop_assignments_},
        {base.operating_periods_, im.operating_periods_},
        {base.day_type_assignments_, im.day_type_assignments_},
        {base.destination_displays_, im.destination_displays_},
        {base.vehicle_types_, im.vehicle_types_}, {base.lines_, im.lines_},
        {base.operators_, im.operators_}, {base.notices_, im.notices_},
        {base.journey_patterns_, im.journey_patterns_}, get_train_numbers(doc));
    im.f_ = std::move(f);
    im.doc_ = std::move(doc);
  } catch (std::exception const& e) {
    std::clog << "ERROR: " << e.what() << " IN " << path << "\n";
    return std::nullopt;
  }
  return im;
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

  auto base_files = std::vector<fs::path>{};
  auto timetable_files = std::vector<fs::path>{};
  for (auto const& f : d.list_files(".")) {
    if (!is_xml_file(f)) {
      continue;
    }
    if (config.base_paths_.contains(f.filename()) ||
        (f.generic_string().contains("SCHWEIZ") &&
         f.generic_string().contains("_1_1_"))) {
      base_files.push_back(f);
    } else {
      timetable_files.push_back(f);
    }
  }

  auto const ch_prio = [](std::string const& s) {
    if (s.contains("SITE")) return 0;
    if (s.contains("SERVICECALENDAR")) return 1;
    if (s.contains("COMMON")) return 2;
    if (s.contains("RESOURCE")) return 3;
    if (s.contains("SERVICE")) return 4;
    return 99;
  };
  utl::sort(base_files, [&](fs::path const& a, fs::path const& b) {
    return ch_prio(a.generic_string()) < ch_prio(b.generic_string());
  });

  auto const r = script_runner{config.user_script_};

  auto tt_mtx = std::mutex{};
  auto tz_map = gtfs::tz_map{};

  auto get_attribute_combination_idx =
      [&tt,
       combis = hash_map<basic_string<attribute_idx_t>,
                         attribute_combination_idx_t>{},
       cache = basic_string<attribute_idx_t>{}](
          std::vector<notice const*> const& attributes) mutable {
        cache.clear();
        for (auto const& a : attributes) {
          if (a->idx_ != attribute_idx_t::invalid()) {
            cache.push_back(a->idx_);
          }
        }
        utl::erase_duplicates(cache);
        return utl::get_or_create(combis, cache, [&]() {
          auto const combination_idx =
              attribute_combination_idx_t{tt.attribute_combinations_.size()};
          tt.attribute_combinations_.emplace_back(cache);
          return combination_idx;
        });
      };

  auto global_journey_meetings = journey_meeting_map_t{};
  auto sj_utc_trips = vecvec<service_journey_idx_t, utc_trip>{};
  auto sj_ids = string_store<service_journey_idx_t>{};
  auto sj_trips = vector_map<service_journey_idx_t, trip_idx_t>{};
  auto const add_to_tt = [&](fs::path const& path, intermediate& im) {
    auto lock = std::scoped_lock{tt_mtx};

    for (auto const& [k, v] : im.journey_meetings_) {
      global_journey_meetings[k].insert(begin(v), end(v));
    }

    auto const source_file_idx =
        tt.register_source_file((d.path() / path).generic_string());
    auto const tz = gtfs::get_tz_idx(tt, tz_map, im.tz_->name());

    for (auto& [id, o] : im.operators_) {
      auto a = agency{
          tt, src,   id, tt.register_translation(o->name_), kEmptyTranslation,
          tz, tz_map};
      if (process_agency(r, a)) {
        o->provider_ = register_agency(tt, a);
      }
    }

    auto const add_stop = [&](stop* stop) {
      auto const existing = tt.find(location_id{stop->id_, src});
      if (existing.has_value()) {
        stop->location_ = *existing;
        return;
      }

      auto s = location{tt,
                        src,
                        stop->id_,
                        tt.register_translation(stop->name_),
                        tt.register_translation(stop->public_code_),
                        kEmptyTranslation,
                        stop->pos_,
                        location_type::kStation,
                        stop->parent_ != nullptr ? stop->parent_->location_
                                                 : location_idx_t::invalid(),
                        tz,
                        2_minutes,
                        tz_map};
      if (process_location(r, s)) {
        stop->location_ = register_location(tt, s);
        if (stop->parent_ != nullptr) {
          tt.locations_.children_[stop->parent_->location_].emplace_back(
              stop->location_);
        }
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
      dd->trip_direction_ = tt.register_translation(dd->direction_);
    }

    for (auto& [id, attr] : im.notices_) {
      auto x = attribute{&tt, attr->code_,
                         tt.register_translation(attr->translations_)};
      attr->idx_ = process_attribute(r, x) ? register_attribute(tt, x)
                                           : attribute_idx_t::invalid();
    }

    for (auto& sj : im.service_journeys_) {
      auto const& jp = *sj.get_journey_pattern();

      // Check if provider got filtered by script.
      auto& line = *jp.line_;
      auto const op = sj.operator_->id_.empty() ? line.operator_ : sj.operator_;
      if (op->provider_ == provider_idx_t::invalid()) {
        continue;
      }

      // Create and register route from (line + route_type).
      auto const route_it = line.routes_.find({sj.route_type_, op});
      auto route_id = route_id_idx_t::invalid();
      if (route_it == end(line.routes_)) {
        auto const id =
            fmt::format("{}-{}-{}", line.id_, op->id_, sj.route_type_);
        auto rout = route{tt,
                          src,
                          id,
                          tt.register_translation(line.name_),
                          tt.register_translation(line.product_),
                          route_type_t{get_more_precise_route_type(
                              sj.route_type_, line.route_type_)},
                          line.color_,
                          op->provider_};
        route_id = line.routes_
                       .emplace_hint(
                           route_it, line::route_key{sj.route_type_, op},
                           process_route(r, rout) ? register_route(tt, rout)
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
          tt,
          src,
          sj.id_,
          sj.get_journey_pattern()
              ->stop_points_.front()
              .destination_display_->trip_direction_,
          tt.register_translation(short_name),
          tt.route_ids_[src].route_id_short_names_[route_id],
          sj.vehicle_type_->name_,
          !sj.branding_ref_.empty() ? sj.branding_ref_
                                    : sj.vehicle_type_->short_name_,
          jp.direction_,
          route_id,
          trip_debug{source_file_idx, static_cast<unsigned>(sj.dbg_offset_),
                     static_cast<unsigned>(sj.dbg_offset_)},
      };
      auto const trip_idx =
          process_trip(r, t) ? register_trip(tt, t) : trip_idx_t::invalid();
      if (trip_idx == trip_idx_t::invalid()) {
        continue;
      }

      tt.trip_stop_seq_numbers_.add_back_sized(0U);
      if (shapes_data != nullptr) {
        shapes_data->add_trip_shape_offsets(
            trip_idx,
            cista::pair{shape_idx_t::invalid(), shape_offset_idx_t::invalid()});
      }

      auto const all_destinations_equal = utl::all_of(
          jp.stop_points_, [&](journey_pattern::stop_point const& sp) {
            return sp.destination_display_->trip_direction_ ==
                   sj.get_journey_pattern()
                       ->stop_points_.front()
                       .destination_display_->trip_direction_;
          });
      auto trip_direction = basic_string<translation_idx_t>{};
      if (all_destinations_equal) {
        trip_direction = {sj.get_journey_pattern()
                              ->stop_points_.front()
                              .destination_display_->trip_direction_};
      } else {
        for (auto const& s : jp.stop_points_) {
          trip_direction.push_back(s.destination_display_->trip_direction_);
        }
        trip_direction.pop_back();
      }

      auto attr = basic_string<attribute_combination_idx_t>{};
      for (auto const& s : jp.stop_points_) {
        attr.push_back(get_attribute_combination_idx(s.notices_));
      }
      if (utl::all_of(attr, [&](auto&& x) { return x == attr.front(); })) {
        attr = {attr.front()};
      } else {
        attr.pop_back();
      }

      [[maybe_unused]] auto const sj_idx = sj_ids.store(sj.id_);
      assert(sj_idx == sj_utc_trips.size());
      assert(sj_idx == sj_trips.size());
      sj_trips.push_back(trip_idx);
      auto bucket = sj_utc_trips.add_back_sized(0U);
      for (auto const& [k, traffic_days] :
           expand_local_to_utc(tt, im.tz_, sj)) {
        bucket.push_back(utc_trip{
            .first_dep_offset_ = k.first_dep_day_offset_,
            .tz_offset_ = k.tz_offset_,
            .utc_times_ = k.utc_times_,
            .utc_traffic_days_ = traffic_days,
            .stop_seq_ = stop_seq(sj),
            .trips_ = {sj_idx},
            .trip_direction_ = std::move(trip_direction),
            .attributes_ = std::move(attr),
            .route_id_ = route_id,
        });
      }
    }
  };

  auto base = intermediate{};
  auto const base_im = utl::to_vec(base_files, [&](fs::path const& path) {
    auto im = get_intermediate(base, tt, d, path, config.default_tz_);
    utl::verify(im.has_value(), "failed to parse base {}", path);
    add_to_tt(path, *im);
    base.steal(*im);
    return im;
  });

  auto const pt = utl::get_active_progress_tracker();
  pt->status("Parse Files")
      .out_bounds(0.F, 94.F)
      .in_high(timetable_files.size());

  utl::parallel_for(
      timetable_files,
      [&](fs::path const& path) {
        auto im = get_intermediate(base, tt, d, path, config.default_tz_);
        if (!im.has_value()) {
          return;
        }
        add_to_tt(path, *im);
      },
      pt->update_fn());

  fmt::println(std::clog, "GLOBAL JOURNEY MEETINGS");
  for (auto const& [from, to] : global_journey_meetings) {
    fmt::println(std::clog, "from={}, to={}", from,
                 to | std::views::transform([](journey_meeting const& x) {
                   return x.to_journey_id_;
                 }));
  }
  fmt::println(std::clog, "----");

  auto route_services =
      hash_map<gtfs::route_key_t, std::vector<std::vector<utc_trip>>,
               gtfs::route_key_hash, gtfs::route_key_equals>{};
  auto const add_expanded_trip = [&](utc_trip const& s) {
    auto const c =
        gtfs::to_clasz(to_idx(tt.route_ids_[src].route_id_type_[s.route_id_]));
    auto const it = route_services.find(gtfs::route_key_ptr_t{c, &s.stop_seq_});
    if (it != end(route_services)) {
      for (auto& route : it->second) {
        auto const idx = get_index(route, s);
        if (idx.has_value()) {
          route.insert(std::next(begin(route), static_cast<int>(*idx)), s);
          return;
        }
      }
      it->second.emplace_back(std::vector<utc_trip>{s});
    } else {
      route_services.emplace(gtfs::route_key_t{c, s.stop_seq_, {}, {}},
                             std::vector<std::vector<utc_trip>>{{s}});
    }
  };

  {
    auto const timer = scoped_timer{"loader.gtfs.seated_trips"};

    auto sj_rule_trip =
        vector_map<service_journey_idx_t, gtfs::rule_trip_idx_t>{};
    auto rule_trip_sj =
        vector_map<gtfs::rule_trip_idx_t, service_journey_idx_t>{};
    sj_rule_trip.resize(sj_utc_trips.size(), gtfs::rule_trip_idx_t::invalid());
    auto seated_out =
        paged_vecvec<service_journey_idx_t, service_journey_idx_t>{};
    auto seated_in =
        paged_vecvec<service_journey_idx_t, service_journey_idx_t>{};
    seated_out.resize(sj_utc_trips.size());
    seated_in.resize(sj_utc_trips.size());
    for (auto i = service_journey_idx_t{0U}; i != sj_utc_trips.size(); ++i) {
      auto const id = sj_ids.get(i);
      auto const meeting_it = global_journey_meetings.find(id);
      if (meeting_it != end(global_journey_meetings)) {
        for (auto const& out : meeting_it->second) {
          auto const to = sj_ids.find(out.to_journey_id_).value();
          seated_out[i].emplace_back(to);
          seated_in[to].emplace_back(i);
        }
      } else {
        for (auto const& t : sj_utc_trips[i]) {
          add_expanded_trip(t);
        }
      }
    }

    for (auto i = service_journey_idx_t{0U}; i != sj_utc_trips.size(); ++i) {
      if (seated_out[i].empty() && seated_in[i].empty()) {
        continue;
      }
      sj_rule_trip[i] = gtfs::rule_trip_idx_t{rule_trip_sj.size()};
      rule_trip_sj.emplace_back(i);
    }

    auto ret = gtfs::expanded_seated<netex::utc_trip>{};
    for (auto const [rule_trip, sj] : utl::enumerate(rule_trip_sj)) {
      using std::views::transform;
      auto const to_rule_trip_idx = [&](service_journey_idx_t const x) {
        return sj_rule_trip.at(x);
      };
      auto bucket = ret.expanded_.add_back_sized(0U);
      for (auto const& utc_trip : sj_utc_trips[sj]) {
        bucket.push_back(utc_trip);
        ret.remaining_rule_trip_.push_back(gtfs::rule_trip_idx_t{rule_trip});
      }
      ret.seated_out_.emplace_back(seated_out[sj] |
                                   transform(to_rule_trip_idx));
      ret.seated_in_.emplace_back(seated_in[sj] | transform(to_rule_trip_idx));
    }

    gtfs::build_seated_trips<utc_trip, service_journey_idx_t>(
        tt, ret,
        [&](service_journey_idx_t const sj_idx) {
          auto const trip_idx = sj_trips[sj_idx];
          return fmt::format(
              "({}, {})", tt.trip_id_strings_[tt.trip_ids_[trip_idx][0]].view(),
              tt.get_default_translation(tt.trip_display_names_[trip_idx]));
        },
        [&](netex::utc_trip&& x) { add_expanded_trip(std::move(x)); });
  }

  {
    auto const timer = scoped_timer{"loader.gtfs.write_trips"};

    pt->status("Write Trips")
        .out_bounds(94.F, 96.F)
        .in_high(route_services.size());

    auto lines = hash_map<std::string, trip_line_idx_t>{};
    auto section_providers = basic_string<provider_idx_t>{};
    auto route_colors = basic_string<route_color>{};
    auto external_trip_ids = basic_string<merged_trips_idx_t>{};
    auto section_directions = basic_string<translation_idx_t>{};
    auto location_routes = mutable_fws_multimap<location_idx_t, route_idx_t>{};
    auto section_attributes = basic_string<attribute_combination_idx_t>{};
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
          section_attributes.clear();
          section_providers.clear();
          section_directions.clear();
          route_colors.clear();

          auto prev_end = std::uint16_t{0U};
          for (auto const [i, sj_idx] : utl::enumerate(s.trips_)) {
            auto& trp = sj_utc_trips[sj_idx].front();
            auto const trip_idx = sj_trips[sj_idx];

            auto const end =
                static_cast<std::uint16_t>(prev_end + trp.stop_seq_.size());

            tt.trip_transport_ranges_[trip_idx].emplace_back(
                transport_range_t{tt.next_transport_idx(), {prev_end, end}});
            prev_end = end - 1;

            auto const merged_trip = tt.register_merged_trip({trip_idx});
            if (s.trips_.size() == 1U) {
              external_trip_ids.push_back(merged_trip);
              section_directions.insert(std::end(section_directions),
                                        std::begin(trp.trip_direction_),
                                        std::end(trp.trip_direction_));
              section_providers.push_back(
                  tt.route_ids_[src].route_id_provider_[trp.route_id_]);
              section_attributes = trp.attributes_;
            } else {
              for (auto section = 0U; section != trp.stop_seq_.size() - 1;
                   ++section) {
                external_trip_ids.push_back(merged_trip);
                section_directions.push_back(
                    trp.trip_direction_.size() == 1U
                        ? trp.trip_direction_[0]
                        : trp.trip_direction_.at(section));
                section_attributes.push_back(trp.attributes_.size() == 1U
                                                 ? trp.attributes_[0]
                                                 : trp.attributes_.at(section));
                section_providers.push_back(
                    tt.route_ids_[src].route_id_provider_[trp.route_id_]);
              }
            }
          }

          assert(!section_directions.empty());
          assert(s.first_dep_offset_.count() >= -1);
          tt.add_transport(timetable::transport{
              .bitfield_idx_ = utl::get_or_create(
                  bitfield_indices, s.utc_traffic_days_,
                  [&]() { return tt.register_bitfield(s.utc_traffic_days_); }),
              .route_idx_ = route_idx,
              .first_dep_offset_ = {s.first_dep_offset_, s.tz_offset_},
              .external_trip_ids_ = external_trip_ids,
              .section_attributes_ = section_attributes,
              .section_providers_ = section_providers,
              .section_directions_ = section_directions});
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
  tt.location_location_groups_.resize(tt.n_locations());
}

}  // namespace nigiri::loader::netex
