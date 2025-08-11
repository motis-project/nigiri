#pragma once

#include "nigiri/loader/loader_interface.h"

#include "nigiri/loader/netex/data.h"

#include "pugixml.hpp"

namespace nigiri::loader::netex {

void parse_netex_file(netex_data&,
                      loader_config const&,
                      pugi::xml_document const&);

}  // namespace nigiri::loader::netex
