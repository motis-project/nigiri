#pragma once

#include <filesystem>

#include "nigiri/timetable.h"

namespace nigiri {

struct timetable;

// Skip the first 9 stops, bcs they are sentinels
constexpr int stopOffset{9};

void export_gtfs(timetable const& tt, std::filesystem::path const& output_dir);

void write_feed_info(std::filesystem::path const& output_dir);

void write_agencies(timetable const& tt,
                    std::filesystem::path const& output_dir);

void write_stops(timetable const& tt, std::filesystem::path const& output_dir);

void write_stop_times(timetable const& tt,
                      std::filesystem::path const& output_dir);

void write_trips(timetable const& tt,
                 std::filesystem::path const& output_dir,
                 std::vector<size_t> const& route_offsets);

void write_routes(timetable const& tt,
                  std::filesystem::path const& output_dir,
                  std::vector<size_t> const& route_offsets);

void write_calendar(timetable const& tt,
                    std::filesystem::path const& output_dir);

void write_calendar_dates(timetable const& tt,
                          std::filesystem::path const& output_dir);

void write_transfers(timetable const& tt,
                     std::filesystem::path const& output_dir);

}  // namespace nigiri
