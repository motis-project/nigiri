#include <regex>
#include <vector>

#include "date/date.h"

#include "boost/program_options.hpp"

#include "utl/helpers/algorithm.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "nigiri/loader/load.h"
#include "nigiri/loader/loader_interface.h"

std::optional<date::sys_days> parse_date(std::string const& str) {
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
    ("path,p", bpo::value<std::string>(),"path to the timetable, can be a directory or a zip file")
    ("zips,z", bpo::bool_switch()->default_value(false), "if a directory is provided as input path, loads all GTFS zip files within the directory")
    ("start_date,s", bpo::value<std::string>(), "start date of the timetable, format: YYYY-MM-DD")
    ("end_date,e", bpo::value<std::string>(), "end date of the timetable, format: YYYY-MM-DD")
    ("output_file,o",bpo::value<std::string>()->default_value("tt.bin"),"the name of the output file")
    ("link_stop_distance,l",bpo::value<std::uint32_t>()->default_value(100U),"the maximum distance at which stops in proximity will be linked")
    ("time_zone,t", bpo::value<std::string>()->default_value("Europe/Berlin"), "the default time zone to use")
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

  std::optional<date::sys_days> start_date;
  if (vm.count("start_date")) {
    start_date = parse_date(vm["start_date"].as<std::string>());
    if (!start_date.has_value()) {
      std::cout << "Error: malformed start date input\n";
      return 1;
    }
  } else {
    std::cout << "Error: start date of timetable missing\n";
    return 1;
  }

  std::optional<date::sys_days> end_date;
  if (vm.count("end_date")) {
    end_date = parse_date(vm["end_date"].as<std::string>());
    if (!end_date.has_value()) {
      std::cout << "Error: malformed end date input\n";
      return 1;
    }
  } else {
    std::cout << "Error: end date of timetable missing\n";
    return 1;
  }

  auto const output_file =
      std::filesystem::path{vm["output_file"].as<std::string>()};
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
  auto const tt = nigiri::loader::load(input_files, config,
                                       {start_date.value(), end_date.value()});
  tt.write(output_path);

  return 0;
}