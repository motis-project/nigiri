//
// Created by mirko on 10/11/23.
//
#include "service_frame_handler.h"
using namespace std;

namespace nigiri::loader::netex {
void processServiceFrame(timetable& t, const pugi::xml_node& frame) {
  // do something with timetable and frame
  cout << "Beginning of Service Frame, dummy outputs: " << t.n_locations()
       << frame.name();
}
}  // namespace nigiri::loader::netex