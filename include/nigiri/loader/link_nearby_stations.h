namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

// Collects the equivalences between stations of different sources that sit
// within walking distance. Their walking transfers are only emitted when
// emit_footpaths is set: with street routing, the whole footpath layer is
// computed in one pass afterwards and anything derived here would be thrown
// away (or, worse, survive as a shortcut nothing recomputed).
void link_nearby_stations(timetable&, bool emit_footpaths);

}  // namespace nigiri::loader