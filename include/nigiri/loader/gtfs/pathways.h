#pragma once

#include <string_view>

#include "nigiri/loader/gtfs/stop.h"

namespace nigiri::loader {
struct station_graph;
}

namespace nigiri::loader::gtfs {

// Adds the pathways (nodes from stops.txt, levels from levels.txt) to the
// station graph. Footpath endpoints: stops with location_type 0, boarding
// areas connect to their platform at no cost.
void read_pathways(source_idx_t,
                   station_graph&,
                   stops_map_t const&,
                   std::string_view stops_file_content,
                   std::string_view levels_file_content,
                   std::string_view pathways_file_content,
                   script_runner const& = script_runner{});

}  // namespace nigiri::loader::gtfs
