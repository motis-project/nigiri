#include "nigiri/rt/vdv/vdv_update.h"

#include <string>
#include <string_view>
#include <unordered_set>

#include "pugixml.hpp"
#include "utl/verify.h"

#include "nigiri/logging.h"
#include "nigiri/rt/vdv/vdv_run.h"
#include "nigiri/timetable.h"

namespace nigiri::rt {

std::optional<location_idx_t> match_location(timetable const& tt,
                                             std::string_view vdv_stop_id) {
  auto loc_match = std::optional<location_idx_t>{};
  for (auto l = 0U; l != tt.n_locations(); ++l) {
    auto const loc_idx = location_idx_t{l};
    if (std::string_view{begin(tt.locations_.ids_[loc_idx]),
                         end(tt.locations_.ids_[loc_idx])} == vdv_stop_id) {
      loc_match = loc_idx;
      break;
    }
  }
  return loc_match;
}

std::unordered_set<transport_idx_t> match_stops(timetable const& tt,
                                                rt_timetable& rtt,
                                                source_idx_t const src,
                                                vdv_run const& r) {
  auto matched_transports = std::unordered_set<transport_idx_t>{};
  for (auto& vdv_stop : r.stops_) {
    auto const loc_idx = match_location(tt, vdv_stop.stop_id_);
    if (!loc_idx.has_value()) {
      log(log_lvl::error, "vdv_update.match_stops",
          "could not match stop {} to a location in the timetable",
          vdv_stop.stop_id_);
      continue;
    }
  }
  return matched_transports;
}

void process_run(timetable const& tt,
                 rt_timetable& rtt,
                 source_idx_t const src,
                 pugi::xml_node const& run_node) {
  auto const vdv_run = parse_run(run_node);
  auto transport_matches = match_stops(tt, rtt, src, vdv_run);
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
    process_run(tt, rtt, src, run_xpath.node());
  }
}

}  // namespace nigiri::rt