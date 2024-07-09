#include "nigiri/rt/vdv/vdv_xml.h"

#include <sstream>

#include "utl/verify.h"

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

std::optional<bool> get_opt_bool(pugi::xml_node const& node,
                                 std::string const& str,
                                 std::optional<bool> default_to) {
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

}  // namespace nigiri::rt