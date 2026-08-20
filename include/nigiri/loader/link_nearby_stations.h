namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

// Collects the equivalences between stations of different sources that sit
// within walking distance, and emits their walking transfers.
void link_nearby_stations(timetable&);

}  // namespace nigiri::loader