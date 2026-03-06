#pragma once

#include <filesystem>

#include "nigiri/timetable.h"

namespace nigiri {

struct timetable;

void export_gtfs(timetable const& tt, std::filesystem::path const& output_dir);

}  // namespace nigiri
