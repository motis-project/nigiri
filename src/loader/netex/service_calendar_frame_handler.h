//
// Created by mirko on 10/11/23.
//

#pragma once
#include "nigiri/timetable.h"
#include <pugixml.hpp>

namespace nigiri::loader::netex {
void processServiceCalendarFrame(const pugi::xml_node& serviceCalFrame);
}