namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

struct import_context;

void link_nearby_stations(timetable&, import_context&);

}  // namespace nigiri::loader