#include <filesystem>
#include <iostream>

#include "utl/progress_tracker.h"

#include "nigiri/loader/gtfs/loader.h"
#include "nigiri/loader/hrd/loader.h"
#include "nigiri/loader/init_finish.h"
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

  timetable tt;
  tt.date_range_ = {date::sys_days{January / 1 / 2024},
                    date::sys_days{December / 31 / 2024}};
  register_special_stations(tt);

  gtfs::gtfs_loader l;
  auto src_idx_counter = 0U;
  for (auto const zip_file : input_files) {
    auto const dir = make_dir(zip_file);
    if (l.applicable(*dir)) {
      log(log_lvl::info, "main", "loading GTFS timetable in {}",
          zip_file.filename().string());
      l.load({}, source_idx_t{src_idx_counter++}, *dir, tt);
    }
  }

  finalize(tt);

  return 0;
}