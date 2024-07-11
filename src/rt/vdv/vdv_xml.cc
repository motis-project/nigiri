#include "nigiri/rt/vdv/vdv_xml.h"

#include <sstream>

#include "utl/verify.h"

#include "nigiri/rt/vdv/vdv_run.h"

namespace nigiri::rt {

pugi::xpath_node get_req(pugi::xml_node const& node, std::string const& str) {
  auto const xpath = node.select_node(str.c_str());
  utl::verify(xpath, "required xml node not found: {}", str);
  return xpath;
}

std::optional<std::string> get_opt_str(pugi::xml_node const& node,
                                       std::string const& str) {
  auto const xpath = node.select_node(str.c_str());
  return xpath ? std::optional{xpath.node().child_value()} : std::nullopt;
}

bool parse_bool(std::string const& str) {
  auto result = false;
  std::istringstream{str} >> std::boolalpha >> result;
  return result;
}

std::optional<bool> get_opt_bool(
    pugi::xml_node const& node,
    std::string const& str,
    std::optional<bool> default_to = std::nullopt) {
  auto const xpath = node.select_node(str.c_str());
  return xpath ? std::optional{parse_bool(xpath.node().child_value())}
               : default_to;
}

unixtime_t parse_time(std::string const& str) {
  unixtime_t parsed;
  auto ss = std::stringstream{str};
  ss >> date::parse("%FT%T", parsed);
  return parsed;
}

std::optional<unixtime_t> get_opt_time(pugi::xml_node const& node,
                                       std::string const& str) {
  auto const xpath = node.select_node(str.c_str());
  return xpath ? std::optional{parse_time(xpath.node().child_value())}
               : std::nullopt;
}

date::sys_days parse_date(std::string const& str) {
  auto result = date::sys_days{};
  auto ss = std::stringstream{str};
  ss >> date::parse("%F", result);
  return result;
}

vdv_run parse_run(pugi::xml_node const& run_node) {
  auto const get_stops = [&]() {
    auto const get_stop = [](pugi::xml_node const& stop_node) -> vdv_stop {
      auto const pass_through = get_opt_bool(stop_node, "Durchfahrt");
      auto const no_enter = get_opt_bool(stop_node, "Einsteigeverbot");
      auto const no_exit = get_opt_bool(stop_node, "Aussteigeverbot");
      return {.stop_id_ = get_req(stop_node, "HaltID").node().child_value(),
              .platform_arr_ = get_opt_str(stop_node, "AnkunftssteigText"),
              .platform_dep_ = get_opt_str(stop_node, "AbfahrtssteigText"),
              .t_arr_ = get_opt_time(stop_node, "Ankunftszeit"),
              .t_dep_ = get_opt_time(stop_node, "Abfahrtszeit"),
              .t_arr_rt_ = get_opt_time(stop_node, "IstAnkunftPrognose"),
              .t_dep_rt_ = get_opt_time(stop_node, "IstAbfahrtPrognose"),
              .in_allowed_ =
                  pass_through.has_value() && pass_through.value() ? false
                  : no_enter.has_value() ? std::optional{!no_enter.value()}
                                         : std::nullopt,
              .out_allowed_ =
                  pass_through.has_value() && pass_through.value() ? false
                  : no_exit.has_value() ? std::optional{!no_exit.value()}
                                        : std::nullopt,
              .additional_stop_ = get_opt_bool(stop_node, "Zusatzhalt", false)};
    };

    auto stops = std::vector<vdv_stop>{};
    auto const stops_xpath = run_node.select_nodes("IstHalt");
    for (auto const& stop_xpath : stops_xpath) {
      stops.emplace_back(get_stop(stop_xpath.node()));
    }
    return stops;
  };

  return {
      .t_ = parse_time(get_req(run_node, "@Zst").attribute().value()),
      .route_id_ = get_req(run_node, "LinienID").node().child_value(),
      .route_text_ = get_req(run_node, "LinienText").node().child_value(),
      .direction_id_ = get_req(run_node, "RichtungsID").node().child_value(),
      .direction_text_ =
          get_req(run_node, "RichtungsText").node().child_value(),
      .vehicle_ = get_req(run_node, "ProduktID").node().child_value(),
      .trip_ref_ = get_req(run_node, "FahrtRef/FahrtID/FahrtBezeichner")
                       .node()
                       .child_value(),
      .operator_ = get_req(run_node, "BetreiberID").node().child_value(),
      .date_ = parse_date(get_req(run_node, "FahrtRef/FahrtID/Betriebstag")
                              .node()
                              .child_value()),
      .complete_ =
          parse_bool(get_req(run_node, "Komplettfahrt").node().child_value()),
      .canceled_ = get_opt_bool(run_node, "FaelltAus", false).value(),
      .additional_run_ = get_opt_bool(run_node, "Zusatzfahrt", false).value(),
      .stops_ = get_stops()};
}

}  // namespace nigiri::rt