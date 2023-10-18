//
// Created by mirko on 10/11/23.
//

#pragma once
#include "nigiri/timetable.h"
#include <pugixml.hpp>

namespace nigiri::loader::netex {
void processServiceFrame(timetable& t, const pugi::xml_node& frame);
}