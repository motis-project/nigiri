#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "nigiri/timetable.h"

namespace nigiri {

struct timetable;

// Skip the first 9 stops, bcs they are sentinels
constexpr int stopOffset{9};

struct gtfs_export_target {
  virtual ~gtfs_export_target() = default;

  virtual std::ostream& create_file(std::string const& filename) = 0;

  virtual void finalize() {}
};

struct gtfs_dir_export_target : public gtfs_export_target {
  explicit gtfs_dir_export_target(std::filesystem::path dir);

  std::ostream& create_file(std::string const& filename) override;

  std::filesystem::path dir_;
  std::vector<std::unique_ptr<std::ofstream>> streams_;
};

struct gtfs_zip_export_target : public gtfs_export_target {
  explicit gtfs_zip_export_target(std::filesystem::path zip_path);
  ~gtfs_zip_export_target() override;

  gtfs_zip_export_target(gtfs_zip_export_target const&) = delete;
  gtfs_zip_export_target(gtfs_zip_export_target&&) = delete;
  gtfs_zip_export_target& operator=(gtfs_zip_export_target const&) = delete;
  gtfs_zip_export_target& operator=(gtfs_zip_export_target&&) = delete;

  std::ostream& create_file(std::string const& filename) override;
  void finalize() override;

  std::filesystem::path zip_path_;
  std::filesystem::path tmp_dir_;
  std::vector<std::string> filenames_;
  std::vector<std::unique_ptr<std::ofstream>> streams_;

  struct impl;  // hides mz_zip_archive (miniz) from this header
  std::unique_ptr<impl> impl_;
};

std::unique_ptr<gtfs_export_target> make_gtfs_export_target(
    std::filesystem::path const& output_path);

void export_gtfs(timetable const& tt, std::filesystem::path const& output_path);

void write_feed_info(gtfs_export_target& out);

void write_agencies(timetable const& tt, gtfs_export_target& out);

void write_stops(timetable const& tt, gtfs_export_target& out);

void write_stop_times(timetable const& tt, gtfs_export_target& out);

void write_trips(timetable const& tt,
                 gtfs_export_target& out,
                 std::vector<size_t> const& route_offsets);

void write_routes(timetable const& tt,
                  gtfs_export_target& out,
                  std::vector<size_t> const& route_offsets);

void write_calendar(timetable const& tt, gtfs_export_target& out);

void write_transfers(timetable const& tt, gtfs_export_target& out);

}  // namespace nigiri
