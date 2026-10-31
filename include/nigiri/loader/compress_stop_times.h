#pragma once

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

struct import_context;

void compress_stop_times(timetable&, import_context&);

}  // namespace nigiri::loader
