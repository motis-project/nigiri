#pragma once

#include <utility>

#include "nigiri/loader/import_context.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

// Fills missing metadata (URLs, colors, Wikidata ids, translations, flags) of
// entities from their matched counterparts.
struct metadata_enricher {
  template <typename T>
  using pairs = hash_map<std::pair<T, T>, unsigned>;  // (target, source)
  using route_id = std::pair<source_idx_t, route_id_idx_t>;

  void add_transport(transport_idx_t target, transport_idx_t source);
  void add_stop(location_idx_t target, location_idx_t source);
  void apply();

  timetable& tt_;
  pairs<route_id> route_ids_{};
  pairs<provider_idx_t> providers_{};
  pairs<location_idx_t> stops_{};
  hash_map<transport_idx_t, std::uint8_t> flags_{};
};

void match_metadata_routes(
    timetable&,
    import_context const&,
    vecvec<location_idx_t, location_idx_t> const& root_neighbours,
    metadata_enricher&);

}  // namespace nigiri::loader
