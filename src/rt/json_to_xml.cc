#include "nigiri/rt/json_to_xml.h"

#include <cctype>
#include <string>

#include "boost/json.hpp"
#include "boost/system/error_code.hpp"

#include "pugixml.hpp"

#include "utl/verify.h"

namespace nigiri::rt {

namespace {

std::string to_pascal_case(std::string_view name) {
  auto out = std::string{name};
  if (!out.empty()) {
    out.front() = static_cast<char>(
        std::toupper(static_cast<unsigned char>(out.front())));
  }
  return out;
}

std::string scalar_to_string(boost::json::value const& v) {
  switch (v.kind()) {
    case boost::json::kind::string: return std::string{v.as_string()};
    case boost::json::kind::int64:
    case boost::json::kind::uint64:
    case boost::json::kind::double_:
    case boost::json::kind::bool_:  // serialize handles bool/int/double
      return boost::json::serialize(v);
    default:  // null/array/object are not represented as scalars
      return {};
  }
}

void fill_node(pugi::xml_node parent, boost::json::value const& value);

void append_value(pugi::xml_node parent,
                  std::string_view name,
                  boost::json::value const& value) {
  auto const node_name = to_pascal_case(name);
  if (value.is_array()) {
    for (auto const& element : value.as_array()) {
      auto child = parent.append_child(node_name.c_str());
      fill_node(child, element);
    }
    return;
  }

  auto child = parent.append_child(node_name.c_str());
  fill_node(child, value);
}

void fill_object(pugi::xml_node parent, boost::json::object const& obj) {
  auto const value_it = obj.if_contains("value");
  auto const treat_value_as_text =
      value_it != nullptr &&
      (value_it->is_string() || value_it->is_bool() || value_it->is_int64() ||
       value_it->is_uint64() || value_it->is_double());
  if (treat_value_as_text) {
    auto const text = scalar_to_string(*value_it);
    if (!text.empty()) {
      parent.text().set(text.c_str());
    }
  }

  for (auto const& kv : obj) {
    auto const& val = kv.value();
    if (value_it != nullptr && value_it == &val && treat_value_as_text) {
      continue;
    }

    auto const key_name = std::string{kv.key()};
    if (val.is_array() || val.is_object()) {
      append_value(parent, key_name, val);
      continue;
    }

    if (val.is_null()) {
      continue;
    }

    if (treat_value_as_text) {
      auto attr = parent.append_attribute(key_name.c_str());
      attr.set_value(scalar_to_string(val).c_str());
      continue;
    }

    append_value(parent, key_name, val);
  }
}

void fill_node(pugi::xml_node parent, boost::json::value const& value) {
  switch (value.kind()) {
    case boost::json::kind::object:
      fill_object(parent, value.as_object());
      break;
    case boost::json::kind::array:
      for (auto const& element : value.as_array()) {
        fill_node(parent, element);
      }
      break;
    case boost::json::kind::null: break;
    default: parent.text().set(scalar_to_string(value).c_str()); break;
  }
}

}  // namespace

pugi::xml_document to_xml(std::string_view s) {
  auto ec = boost::system::error_code{};
  auto const root = boost::json::parse(s, ec);
  utl::verify(!ec, "json_to_xml: Unable to parse json: {}", ec.message());

  utl::verify(root.is_object(), "json_to_xml: root must be an object");

  auto doc = pugi::xml_document{};
  auto siri = doc.append_child("Siri");
  siri.append_attribute("xmlns:datex")
      .set_value("http://datex2.eu/schema/2_0RC1/2_0");
  siri.append_attribute("xmlns").set_value("http://www.siri.org.uk/siri");
  siri.append_attribute("xmlns:acsb").set_value("http://www.ifopt.org.uk/acsb");
  siri.append_attribute("xmlns:ifopt")
      .set_value("http://www.ifopt.org.uk/ifopt");
  siri.append_attribute("xmlns:xsi")
      .set_value("http://www.w3.org/2001/XMLSchema-instance");
  siri.append_attribute("xmlns:xs")
      .set_value("http://www.w3.org/2001/XMLSchema");
  siri.append_attribute("version").set_value("2.0");

  auto service_delivery = siri.append_child("ServiceDelivery");
  for (auto const& [key, value] : root.as_object()) {
    append_value(service_delivery, key, value);
  }

  return doc;
}

}  // namespace nigiri::rt
