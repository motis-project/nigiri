#include "nigiri/rt/vdv/vdv_update.h"

#include <string>
#include <string_view>

#include "pugixml.hpp"
#include "utl/verify.h"

#include "nigiri/rt/vdv/vdv_run.h"

namespace nigiri::rt {

void process_run(pugi::xml_node const& run_node) {
  auto const vdv_run = parse_run(run_node);
}

void vdv_update(timetable const& tt,
                rt_timetable& rtt,
                source_idx_t const src,
                std::string const& vdv_msg) {
  auto doc = pugi::xml_document{};
  auto result = doc.load_string(vdv_msg.c_str());
  utl::verify(result, "XML [{}] parsed with errors: {}\n", vdv_msg,
              result.description());
  utl::verify(
      std::string_view{doc.first_child().name()} == "DatenAbrufenAntwort",
      "Invalid message type {} for vdv update", doc.first_child().name());

  auto const runs_xpath =
      doc.select_nodes("DatenAbrufenAntwort/AUSNachricht/IstFahrt");
  for (auto const& run_xpath : runs_xpath) {
    process_run(run_xpath.node());
  }
}

}  // namespace nigiri::rt