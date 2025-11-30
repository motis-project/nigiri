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

#include "nigiri/loader/netex/intermediate.h"

namespace fs = std::filesystem;

namespace nigiri::loader::netex {

struct none {};

struct stop {
  std::string_view parent_;
  std::string_view id_;
  std::string_view name_;
  geo::latlng pos_;
};

struct authority {
  std::string_view name_;
  std::string_view short_name_;
};

struct line {
  std::string_view name_;
  authority const* authority_;
};

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

template <typename T>
concept Pointer = std::is_pointer_v<T>;

template <Pointer T>
auto format_as(T const t) {
  return cista::to_tuple(*t);
}

template <typename T>
concept Tuplable = cista::to_tuple_works_v<T>;

template <Tuplable T>
auto format_as(T const& t) {
  return cista::to_tuple(t);
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

std::string_view child_attr(pugi::xml_node n,
                            char const* child,
                            char const* attr) {
  auto const str = n.child(child).attribute(attr).as_string();
  return str == nullptr ? std::string_view{} : std::string_view{str};
}

void load_timetable(loader_config const&,
                    source_idx_t const,
                    dir const& d,
                    timetable& tt,
                    hash_map<bitfield, bitfield_idx_t>& /*bitfield_indices*/,
                    assistance_times* /*assistance*/,
                    shapes_storage* /*shapes_data*/) {
  auto const global_timer = nigiri::scoped_timer{"netex parser"};

  auto const pt = utl::get_active_progress_tracker();

  auto const xml_files =
      utl::all(d.list_files(""))  //
      | utl::remove_if([&](fs::path const& f) { return !is_xml_file(f); })  //
      | utl::vec();

  pt->status("Parse Files").out_bounds(0.F, 90.F).in_high(xml_files.size());

  utl::parallel_ordered_collect_threadlocal<none>(
      xml_files.size(),
      [&](none&, std::size_t const i) {
        CISTA_UNUSED_PARAM(i)

        auto const f = d.get_file(xml_files.at(i));
        auto const file_content = f.data();
        auto doc = pugi::xml_document{};
        auto const result =
            doc.load_buffer(file_content.data(), file_content.size(),
                            pugi::parse_default | pugi::parse_trim_pcdata);
        utl::verify(result, "Unable to parse XML buffer: {} at offset {}",
                    result.description(), result.offset);

        auto authorities = hash_map<std::string_view, authority>{};
        for (auto const s :
             doc.select_nodes("//ResourceFrame/organisations/Authority")) {
          auto const n = s.node();
          authorities.emplace(
              n.attribute("id").as_string(),
              authority{.name_ = n.child("Name").child_value(),
                        .short_name_ = n.child("ShortName").child_value()});
        }
        fmt::println("Authorities:\n\t{}", fmt::join(authorities, "\n\t"));

        auto stops = hash_map<std::string_view, stop>{};
        for (auto const s :
             doc.select_nodes("//SiteFrame/stopPlaces/StopPlace | "
                              "//SiteFrame/stopPlaces/Quay")) {
          auto const n = s.node();

          auto const get_global_id = [](pugi::xml_node const x) {
            return x.select_node("keyList/KeyValue/Key[text() = 'GlobalID']")
                .parent()
                .child("Value")
                .child_value();
          };

          auto const get_pos = [](pugi::xml_node const x) {
            return geo::latlng{
                utl::parse<double>(x.select_node("Centroid/Location/Latitude")
                                       .node()
                                       .child_value()),
                utl::parse<double>(x.select_node("Centroid/Location/Longitude")
                                       .node()
                                       .child_value())};
          };

          auto const stop_id = n.attribute("id").as_string();
          auto const global_stop_id = get_global_id(n);
          stops.emplace(stop_id,
                        stop{.parent_ = {},
                             .id_ = global_stop_id ? global_stop_id : stop_id,
                             .name_ = n.child("Name").child_value(),
                             .pos_ = get_pos(n)});

          for (auto const q : n.select_nodes("quays/Quay")) {
            auto const qn = q.node();
            auto const quay_id = qn.attribute("id").as_string();
            auto const global_quay_id = get_global_id(qn);
            stops.emplace(quay_id,
                          stop{.parent_ = stop_id,
                               .id_ = global_quay_id ? global_quay_id : quay_id,
                               .name_ = qn.child("Name").child_value(),
                               .pos_ = get_pos(qn)});
          }
        }

        auto psa = hash_map<std::string_view, stop const*>{};
        for (auto const& a : doc.select_nodes("//ServiceFrame/stopAssignments/"
                                              "PassengerStopAssignment")) {
          auto const n = a.node();
          auto const sstop = child_attr(n, "ScheduledStopPointRef", "ref");
          auto const quay = child_attr(n, "QuayRef", "ref");
          auto const stop_place = child_attr(n, "StopPlaceRef", "ref");
          if (!sstop.empty() && (!quay.empty() || !stop_place.empty())) {
            auto s = end(stops);

            if (!quay.empty()) {
              s = stops.find(quay);
            }
            if (s == end(stops) && !stop_place.empty()) {
              s = stops.find(stop_place);
            }

            if (s != end(stops)) {
              psa[sstop] = &s->second;
            }
          }
        }

        fmt::println("Stops:\n\t{}", fmt::join(stops, "\n\t"));
        fmt::println("PassengerStopAssignments:\n\t{}",
                     fmt::join(psa | std::views::transform([](auto&& x) {
                                 return std::pair{x.first, *x.second};
                               }),
                               "\n\t"));

        auto operating_periods = hash_map<std::string_view, bitfield>{};
        for (auto const s : doc.select_nodes(
                 "//ServiceCalendarFrame/ServiceCalendar/operatingPeriods/"
                 "UicOperatingPeriod")) {
          auto const sn = s.node();
          auto const from = parse_date(sn.child("FromDate").child_value());
          auto const to = parse_date(sn.child("ToDate").child_value());
          auto const bits =
              std::string_view{sn.child("ValidDayBits").child_value()};

          utl::verify((to - from).count() + 1U == bits.size(),
                      "from={}, to={} != n_bits={}", from, to, bits.size());

          auto bf = bitfield{};
          auto day = std::max(from, tt.internal_interval_days().from_);
          for (; day <= to && day < tt.internal_interval_days().to_;
               day += std::chrono::days{1}) {
            auto const tt_day_idx =
                (day - tt.internal_interval_days().from_).count();
            if (tt_day_idx >= 0 && tt_day_idx < static_cast<int>(bf.size())) {
              bf.set(tt_day_idx, bits.at((day - from).count()));
            }
          }

          operating_periods.emplace(sn.attribute("id").as_string(), bf);
        }
        fmt::println("UicOperatingPeriods:\n\t{}",
                     fmt::join(operating_periods, "\n\t"));

        auto days = hash_map<std::string_view, bitfield const*>{};
        for (auto const x : doc.select_nodes(
                 "//ServiceCalendarFrame/ServiceCalendar/dayTypeAssignments/"
                 "DayTypeAssignment")) {
          auto const n = x.node();
          auto const op_period = child_attr(n, "OperatingPeriodRef", "ref");
          auto const day_type = child_attr(n, "DayTypeRef", "ref");

          auto const it = operating_periods.find(op_period);
          if (it == end(operating_periods)) {
            log(log_lvl::error, "netex.DayTypeAssignment",
                "OperatingPeriodRef=\"{}\" not found", op_period);
            continue;
          }

          days.emplace(day_type, &it->second);
        }
        fmt::println("DayTypes:\n\t{}",
                     fmt::join(days | std::views::transform([](auto&& x) {
                                 return std::pair{x.first, *x.second};
                               }),
                               "\n\t"));

        auto lines = hash_map<std::string_view, line>{};
        for (auto const l : doc.select_nodes("//ServiceFrame/lines/Line")) {
          auto const n = l.node();
          auto const it =
              authorities.find(child_attr(n, "AuthorityRef", "ref"));
          lines.emplace(
              n.attribute("id").as_string(),
              line{.name_ = n.child("Name").child_value(),
                   .authority_ =
                       it == end(authorities) ? nullptr : &it->second});
        }

        auto destination_displays =
            hash_map<std::string_view, std::string_view>{};
        for (auto const display : doc.select_nodes(
                 "//ServiceFrame/destinationDisplays/DestinationDisplay")) {
          auto const n = display.node();
          destination_displays.emplace(n.attribute("id").as_string(),
                                       n.child("FrontText").child_value());
        }

        auto journey_patterns = hash_map<std::string_view, journey_pattern>{};
        for (auto const j : doc.select_nodes(
                 "//ServiceFrame/journeyPatterns/ServiceJourneyPattern")) {
          auto const n = j.node();

          auto stop_points = std::vector<journey_pattern::stop_point>{};
          for (auto const sp : n.child("pointsInSequence").children()) {
            stop_points.push_back(journey_pattern::stop_point{
                .id_ = sp.attribute("id").as_string(),
                .stop_ = psa.at(child_attr(sp, "ScheduledStopPointRef", "ref")),
                .destination_display_ = destination_displays.at(
                    child_attr(sp, "DestinationDisplayRef", "ref")),
                .in_allowed_ =
                    std::string_view{sp.child("ForBoarding").child_value()} ==
                    "true",
                .out_allowed_ =
                    std::string_view{sp.child("ForAlighting").child_value()} ==
                    "true"});
          }

          journey_patterns.emplace(
              n.attribute("id").as_string(),
              journey_pattern{
                  .line_ = &lines.at(
                      child_attr(n.child("RouteView"), "LineRef", "ref")),
                  .direction_ =
                      direction_id_t{
                          child_attr(n, "DirectionRef", "ref").ends_with("1::")
                              ? 0
                              : 1},
                  .stop_points_ = stop_points});
        }
        fmt::println(
            "JourneyPatterns:\n\t{}",
            fmt::join(
                journey_patterns |
                    std::views::transform(
                        [](std::pair<std::string_view, journey_pattern> const&
                               x) {
                          return std::tuple{
                              x.first, x.second.line_->name_,
                              x.second.line_->authority_->name_,
                              x.second.stop_points_ |
                                  std::views::transform(
                                      [](journey_pattern::stop_point const s) {
                                        return std::tuple{s.stop_->name_,
                                                          s.stop_->id_};
                                      })};
                        }),
                "\n\t"));

        for (auto const s : doc.select_nodes(
                 "//TimetableFrame/vehicleJourneys/ServiceJourney")) {
          auto const n = s.node();
          auto const trip_nr =
              n.select_node("keyList/KeyValue/Key[text() = 'TripNr']")
                  .parent()
                  .child("Value")
                  .child_value();
          auto const day_type =
              n.select_node("dayTypes/DayTypeRef/@ref").attribute().as_string();
          std::cout << "trip_nr=" << trip_nr
                    << ", day_type=" << *days.at(day_type) << "\n";

          // auto const transport_mode = n.child("TransportMode").child_value();
          // auto const transport_sub_mode =
          //   n.child("TransportSubmode").child("RailSubmodule").child_value();
        }

        auto im = intermediate{};
        return im;
      },
      [](std::size_t const i, intermediate const& im) {
        CISTA_UNUSED_PARAM(i)
        CISTA_UNUSED_PARAM(im)
      },
      pt->update_fn());
}

}  // namespace nigiri::loader::netex
