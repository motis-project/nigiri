#include <filesystem>
#include <iostream>

#include "utl/progress_tracker.h"

#include "nigiri/loader/load.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

std::unique_ptr<cista::wrapped<nigiri::timetable>> load_timetable(
    std::filesystem::path const& input_path) {
  // gather paths of input files in target folder
  std::vector<std::filesystem::path> input_files;
  if (std::filesystem::is_regular_file(input_path) &&
      input_path.has_extension() && input_path.extension() == ".zip") {
    // input path directly to GTFS zip file
    input_files.emplace_back(input_path);
  } else if (std::filesystem::is_directory(input_path)) {
    // input path to directoy
    for (auto const& dir_entry :
         std::filesystem::directory_iterator(input_path)) {
      std::filesystem::path file_path(dir_entry);
      if (std::filesystem::is_regular_file(dir_entry) &&
          file_path.has_extension() && file_path.extension() == ".zip") {
        input_files.emplace_back(file_path);
      }
    }
  } else {
    std::cout << "path provided is invalid\n";
  }

  auto const config = nigiri::loader::loader_config{100U, "Europe/Berlin"};
  auto const tt_interval = nigiri::interval<date::sys_days>{
      {date::January / 1 / 2024}, {date::December / 31 / 2024}};

  // hash tt settings and input files
  auto h = cista::hash_combine(
      cista::BASE_HASH, tt_interval.from_.time_since_epoch().count(),
      tt_interval.to_.time_since_epoch().count(), config.link_stop_distance_,
      cista::hash(config.default_tz_));
  for (auto const& input_file : input_files) {
    h = cista::hash_combine(h, cista::hash(input_file.string()));
  }

  auto const data_dir = std::filesystem::is_directory(input_files[0])
                            ? input_files[0]
                            : input_files[0].parent_path();
  auto const dump_file_path = data_dir / fmt::to_string(h);

  std::unique_ptr<cista::wrapped<nigiri::timetable>> tt;
  auto loaded = false;

  // try to load timetable from cache
  if (exists(dump_file_path)) {
    log(nigiri::log_lvl::info, "benchmark.load", "loading cached timetable {}",
        fmt::to_string(h));
    try {
      tt = std::make_unique<cista::wrapped<nigiri::timetable>>(
          nigiri::timetable::read(cista::memory_holder{
              cista::file{dump_file_path.c_str(), "r"}.content()}));
      (**tt).locations_.resolve_timezones();
      loaded = true;
    } catch (std::exception const& e) {
      log(nigiri::log_lvl::error, "benchmark.load",
          "can not read cached timetable image: {}", e.what());
    }
  }

  // load timetable from GTFS files
  if (!loaded) {
    log(nigiri::log_lvl::info, "benchmark.load",
        "no cached timetable found, loading from files");
    tt = std::make_unique<cista::wrapped<nigiri::timetable>>(
        cista::raw::make_unique<nigiri::timetable>(
            load(input_files, config, tt_interval)));
    // write cache file
    (**tt).write(dump_file_path);
  }

  return tt;
}

int main(int argc, char* argv[]) {
  auto const progress_tracker = utl::activate_progress_tracker("benchmark");
  auto const silencer = utl::global_progress_bars{true};

  if (argc != 2) {
    std::cout << "usage: nigiri-benchmark "
              << "[GTFS_ZIP_FILE] | [DIRECTORY]\nloads a zip file "
                 "containing a timetable in GTFS format, or attempts to load "
                 "all zip files within a given directory\n";

    return 1;
  }

  auto tt = load_timetable({argv[1]});

  return 0;
}