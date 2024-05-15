#include <regex>
#include <vector>

#include "date/date.h"

#include "boost/program_options.hpp"

#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "nigiri/loader/load.h"
#include "nigiri/loader/loader_interface.h"

date::sys_days parse_date(std::string const& str) {
  if (str == "TODAY") {
    return std::chrono::time_point_cast<date::days>(
        std::chrono::system_clock::now());
  }

  date::sys_days parsed;
  std::stringstream ss;
  ss << str;
  ss >> date::parse("%F", parsed);
  return parsed;
}

int main(int argc, char** argv) {
  namespace bpo = boost::program_options;
  auto const progress_tracker = utl::activate_progress_tracker("importer");
  auto const silencer = utl::global_progress_bars{true};

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", "produce this help message")
    ("path,p", bpo::value<std::string>(),
                "path to the timetable, can be a directory or a zip file")
    ("zips,z", bpo::bool_switch()->default_value(false),
                "if a directory is provided as input path, "
                "loads all GTFS zip files within the directory")
    ("start_date,s", bpo::value<std::string>()->default_value("TODAY"),
                "start date of the timetable, format: YYYY-MM-DD")
    ("num_days,n", bpo::value<std::uint32_t>()->default_value(365U),
                "the length of the timetable in days")
    ("end_date,e", bpo::value<std::string>(),
                "end date of the timetable, format: YYYY-MM-DD, overrides num_days")
    ("out,o", bpo::value<std::string>()->default_value("tt.bin"),
                "the name of the output file")
    ("link_stop_distance,l",bpo::value<std::uint32_t>()->default_value(100U),
                "the maximum distance at which stops in proximity will be linked")
    ("time_zone,t", bpo::value<std::string>()->default_value("Europe/Berlin"),
                "the default time zone to use")
  ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);  // sets default values

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  auto const input_path = std::filesystem::path{vm["path"].as<std::string>()};
  auto input_files = std::vector<std::filesystem::path>{};
  if (vm.count("path")) {
    if (vm["zips"].as<bool>()) {
      // input path to directory of GTFS zips
      for (auto const& dir_entry :
           std::filesystem::directory_iterator(input_path)) {
        std::filesystem::path file_path(dir_entry);
        if (std::filesystem::is_regular_file(dir_entry) &&
            file_path.has_extension() && file_path.extension() == ".zip") {
          input_files.emplace_back(file_path);
        }
      }
    } else {
      // input path to single GTFS zip file or directory
      input_files.emplace_back(input_path);
    }
  } else {
    std::cout << "Error: timetable path missing\n";
  }

  date::sys_days const start_date =
      parse_date(vm["start_date"].as<std::string>());
  date::sys_days end_date =
      start_date + date::days{vm["num_days"].as<std::uint32_t>()};
  if (vm.count("end_date")) {
    end_date = parse_date(vm["end_date"].as<std::string>());
  }

  auto const output_file = std::filesystem::path{vm["out"].as<std::string>()};
  auto output_path = output_file;
  if (output_file.has_filename() && !output_file.has_parent_path()) {
    if (is_directory(input_path)) {
      output_path = input_path;
    } else {
      output_path = input_path.parent_path();
    }
    output_path /= output_file;
  } else if (is_directory(output_file)) {
    output_path /= "tt.bin";
  }

  auto const config = nigiri::loader::loader_config{
      vm["link_stop_distance"].as<std::uint32_t>(),
      vm["time_zone"].as<std::string>()};
  auto const tt =
      nigiri::loader::load(input_files, config, {start_date, end_date});
  tt.write(output_path);

  return 0;
}