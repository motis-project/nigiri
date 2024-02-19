#include "nigiri/rust.h"

std::unique_ptr<Timetable> new_timetable(rust::Vec<rust::String> const& paths) {
    for (auto const& p : paths) {
        std::cout << "path: " << p << "\n";
    }
    return std::make_unique<Timetable>();
}