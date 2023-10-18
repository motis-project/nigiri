//
// Created by mirko on 8/21/23.
//
#include "nigiri/loader/dir.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "nigiri/loader/netex/load_timetable.h"
#include "pugixml.hpp"
#include "resource_frame_handler.h"
// #include "service_calendar_frame_handler.h"
#include "service_frame_handler.h"

#include <filesystem>

namespace nigiri::loader::netex::utils {
bool isDirectory(const char* filename) {
  size_t length = std::strlen(filename);
  return filename[length - 1] == '/';
}

// Entry point - this method is called for every single xml file, and we
// gradually build a timetable object
void handle_xml_parse_result(timetable& t,
                             pugi::xml_document& doc,
                             gtfs::tz_map& timezones,
                             gtfs::agency_map_t& agencyMap) {
  auto const frames = doc.child("PublicationDelivery")
                          .child("dataObjects")
                          .child("CompositeFrame")
                          .child("frames");

  // 1. ResourceFrame -> Defines responsibilities, authorities etc
  processResourceFrame(t, frames.child("ResourceFrame"), timezones, agencyMap);

  // 2. ServiceCalendarFrame -> Defines the days when a connection is valid
  //  processServiceCalendarFrame(frames.child("ServiceCalendarFrame"));

  // 3. ServiceFrame -> Contains all the logic about stops, routes, journeys
  processServiceFrame(t, frames.child("ServiceFrame"));

  // 4. SiteFrame -> Seems to define the exact stops
  // auto const siteFrame = frames.child("SiteFrame");

  // 5. TimetableFrame, contains the exact timetable info about journeys
  // auto const timetableFrame = frames.child("TimetableFrame");

  auto const scheduledStopPoints =
      frames.child("ServiceFrame").child("scheduledStopPoints");
  for (const auto& ssp : scheduledStopPoints) {
    std::cout << "Name: " << ssp.child("Name").text().get()
              << ", id: " << ssp.attribute("id").value() << "\n";
  }

  // TODO DELETE THIS DUMMY HANDLING:
  date::year_month_day day{};
  t.day_idx(day);
}
}  // namespace nigiri::loader::netex::utils

/*
 *
 *
 *
 *  MAIN METHOD
 *
 *
 *
 *
 */

namespace fs = std::filesystem;
namespace utils = nigiri::loader::netex::utils;

namespace nigiri::loader::netex {

void load_timetable(source_idx_t src, dir const& d, timetable& t) {

  // Step 0: Define all the variables needed throughout the parsing
  gtfs::tz_map timezones{};
  gtfs::agency_map_t agencyMap{};

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

    std::cout << "Status of parsed xml: " << result.status
              << " for file: " << file.filename() << "\n";

    // Step 4: Handle the parse result and build up the data structures
    utils::handle_xml_parse_result(t, doc, timezones, agencyMap);

    /*
     *
     *
     *
     *
     */

    // Step 5: Process the data structures in the same way as
    // gtfs/load_timetable.cc from line 106 on
  }

  // changing nothing relevant just for the compiling to perform without
  // warnings for unused parameters
  // TODO delete
  std::cout << src;
  date::year_month_day day{};
  t.day_idx(day);
}

}  // namespace nigiri::loader::netex