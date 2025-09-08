#pragma once

#include "nigiri/loader/loader_interface.h"

#include "nigiri/loader/netex/data.h"

#include "pugixml.hpp"

namespace nigiri::loader::netex {

void handle_stop_place(netex_data&, netex_ctx const&, pugi::xml_node const&);
void handle_quay(netex_data&, netex_ctx const&, pugi::xml_node const&);

void finalize_stop_places(netex_data&);

}  // namespace nigiri::loader::netex
