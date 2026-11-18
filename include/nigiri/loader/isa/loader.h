#pragma once

#include "nigiri/loader/loader_interface.h"

namespace nigiri::loader {
struct station_graph;
}

namespace nigiri::loader::isa {

// Station graphs (Umsteigebauwerke, umbw*.asc) of an ISA export (IVU.pool
// ASCII interface). Loads no timetable: areas attach to the stops of other
// sources by their global stop id (DHID, halteste.asc).
struct isa_station_graph_loader : public loader_interface {
  bool applicable(dir const&) const override;
  void load(loader_config const&,
            source_idx_t,
            dir const&,
            timetable&,
            import_context&,
            assistance_times*,
            shapes_storage*) const override;
  std::string_view name() const override;
};

void load_station_graph(dir const&, station_graph&);

}  // namespace nigiri::loader::isa
