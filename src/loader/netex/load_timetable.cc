//
// Created by mirko on 8/21/23.
//
#include "nigiri/loader/dir.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "nigiri/loader/netex/load_timetable.h"
#include "nigiri/loader/netex/route_operator.h"
#include "nigiri/loader/netex/service_calendar.h"
#include "pugixml.hpp"

#include <filesystem>

namespace nigiri::loader::netex::utils {
bool isDirectory(const char* filename) {
  size_t length = std::strlen(filename);
  return filename[length - 1] == '/';
}

// Entry point - this method is called for every single xml file, and we
// gradually build a timetable object
void handle_xml_parse_result(
    timetable& t,
    pugi::xml_document& doc,
    gtfs::tz_map& timezones,
    hash_map<std::string_view, provider_idx_t>& operatorMap,
    hash_map<std::string_view, bitfield>& calendar,
    hash_map<std::string_view, bitfield>& operating_periods) {

  // 1. ResourceFrame -> Defines responsibilities, authorities etc
  read_resource_frame(t, doc, operatorMap);
  std::cout << timezones.size();  // todo delete
  // 2. ServiceCalendarFrame -> Defines the days when a connection is valid
  read_service_calendar(t, doc, calendar, operating_periods);
  //  processServiceCalendarFrame(frames.child("ServiceCalendarFrame"))

  // 3. ServiceFrame -> Contains all the logic about stops, routes, journeys
  // processServiceFrame(t, frames.child("ServiceFrame"));

  // 4. SiteFrame -> Seems to define the exact stops
  // auto const siteFrame = frames.child("SiteFrame");

  // 5. TimetableFrame, contains the exact timetable info about journeys
  // auto const timetableFrame = frames.child("TimetableFrame");
}
}  // namespace nigiri::loader::netex::utils

/*
 * Entry point that is called by load_timetable
 */
namespace fs = std::filesystem;
namespace utils = nigiri::loader::netex::utils;

namespace nigiri::loader::netex {

void load_timetable(source_idx_t src, dir const& d, timetable& t) {

  // Step 0: Define all the variables needed throughout the parsing
  gtfs::tz_map timezones{};

  hash_map<std::string_view, provider_idx_t> operatorMap{};
  hash_map<std::string_view, bitfield> calendar{};
  hash_map<std::string_view, bitfield> operating_periods{};

  // Step 1: Load the zip file
  auto vecOfPaths = d.list_files("");
  // Step 2: Process all files iteratively
  for (auto const& path : vecOfPaths) {
    auto file = d.get_file(path);

    // Break if the filename is just the directory
    if (utils::isDirectory(file.filename())) {
      continue;
    }

    // Step 3: Parse the xml document into the pugi data structure
    pugi::xml_document doc;
    // Here, we need to convert the xml data to a pugi::char_t* string
    std::basic_string<pugi::char_t> convertedData(file.data().begin(),
                                                  file.data().end());
    auto const result = doc.load_string(convertedData.c_str());

    log(log_lvl::info, "loader.netex.load_timetable",
        R"(Status of parsed xml: "{}"for file: "{}")", result.status,
        file.filename());

    // Step 4: Handle the parse result and build up the data structures
    utils::handle_xml_parse_result(t, doc, timezones, operatorMap, calendar,
                                   operating_periods);

    // Step 5: Process the data structures in the same way as
    // gtfs/load_timetable.cc from line 106 on
    std::cout << src;  // TODO DELETE
  }
}

}  // namespace nigiri::loader::netex