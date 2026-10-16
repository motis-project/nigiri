#pragma once

namespace nigiri {
struct timetable;
struct shapes_storage;
}  // namespace nigiri

namespace nigiri::loader {

bool add_virtual_locations(timetable&, shapes_storage*);

}  // namespace nigiri::loader
