#pragma once

#include <optional>

#include "nigiri/loader/loader_interface.h"
#include "nigiri/loader/netex/data.h"

#include "pugixml.hpp"

namespace nigiri::loader::netex {

netex_locale parse_locale(netex_data&, netex_ctx const&, pugi::xml_node const&);
std::optional<netex_locale> get_default_locale(netex_data&,
                                               loader_config const&);

}  // namespace nigiri::loader::netex
