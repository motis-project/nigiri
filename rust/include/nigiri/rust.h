#include "nigiri/timetable.h"
#include "./cxx.h"

using Timetable = nigiri::timetable;

std::unique_ptr<Timetable> new_timetable(rust::Vec<rust::String> const& paths);