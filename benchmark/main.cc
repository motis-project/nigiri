#include <filesystem>
#include <iostream>

#include "utl/progress_tracker.h"

#include "nigiri/loader/load.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;

int main(int argc, char* argv[]) {
  auto const progress_tracker = utl::activate_progress_tracker("server");
  auto const silencer = utl::global_progress_bars{true};

  if (argc != 2) {
    std::cout << "usage: nigiri-benchmark "
              << "[GTFS_ZIP_FILE] | [DIRECTORY]\nloads a zip file "
                 "containing a timetable in GTFS format, or attempts to load "
                 "all zip files within a given directory\n";

    return 1;
  }

  // gather paths of input files in target folder
  std::vector<std::filesystem::path> input_files;
  std::filesystem::path input_path{argv[1]};
  if (std::filesystem::is_regular_file(input_path)) {
    input_files.emplace_back(input_path);
  } else if (std::filesystem::is_directory(input_path)) {
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

  auto const tt = load(input_files, {100U, "Europe/Berlin"},
                       {date::sys_days{January / 1 / 2024},
                        date::sys_days{December / 31 / 2024}});

  return 0;
}