#include "nigiri/loader/netex/line.h"

namespace nigiri::loader::netex {

void parse_lines(const pugi::xml_document& doc,
                 hash_map<std::string_view, line>& line_map) {
  for (const auto& next_line : doc.select_nodes("//Line")) {
    auto id = next_line.node().attribute("id").value();
    auto name = next_line.node().child("Name").text().get();
    auto transport_mode = next_line.node().child("TransportMode").text().get();

    line new_line{id, name, transport_mode};
    line_map.insert(std::make_pair(id, new_line));
  }
}
}  // namespace nigiri::loader::netex
