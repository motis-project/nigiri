#pragma once

#include <filesystem>

#include "nigiri/timetable.h"

namespace nigiri {

struct timetable;

void export_gtfs(timetable const& tt, std::filesystem::path const& output_dir);

void write_feed_info(timetable const& tt,
                     std::filesystem::path const& output_dir);

void write_agencies(timetable const& tt,
                    std::filesystem::path const& output_dir);

void write_stops(timetable const& tt, std::filesystem::path const& output_dir);

void write_stop_times(timetable const& tt,
                      std::filesystem::path const& output_dir);

void write_trips(timetable const& tt, std::filesystem::path const& output_dir);

void write_routes(timetable const& tt, std::filesystem::path const& output_dir);

void write_calendar(timetable const& tt,
                    std::filesystem::path const& output_dir);

void write_calendar_dates(timetable const& tt,
                          std::filesystem::path const& output_dir);

void write_transfers(timetable const& tt,
                     std::filesystem::path const& output_dir);

}  // namespace nigiri
