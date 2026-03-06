#include <filesystem>
#include <iostream>

#include "boost/program_options.hpp"

#include "nigiri/export_gtfs.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

using namespace nigiri;

int main(int argc, char* argv[]) {
  namespace bpo = boost::program_options;

  auto tt_path = std::filesystem::path{};
  auto gtfs_dir = std::filesystem::path{};

  bpo::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce this help message")  //
      ("tt_path,p", bpo::value(&tt_path)->required(),
       "path to a binary file containing a serialized nigiri timetable")  //
      ("gtfs_export_dir,o", bpo::value(&gtfs_dir)->required(),
       "path to the directory to export the timetable as GTFS");
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);

  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  bpo::notify(vm);

  std::cout << "loading timetable...\n";
  auto tt = *nigiri::timetable::read(tt_path);
  tt.resolve();

  std::cout << "exporting timetable as GTFS...\n";
  export_gtfs(tt, gtfs_dir);
  std::cout << "done.\n";

  return 0;
}
