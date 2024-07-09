#pragma once

#include <optional>
#include <string>

#include "pugixml.hpp"

#include "nigiri/types.h"

namespace nigiri::rt {

pugi::xpath_node get_req(pugi::xml_node const&, std::string const&);

std::optional<std::string> get_opt_str(pugi::xml_node const&,
                                       std::string const&);

bool parse_bool(std::string const&);

std::optional<bool> get_opt_bool(pugi::xml_node const&,
                                 std::string const&,
                                 std::optional<bool> default_to = std::nullopt);

unixtime_t parse_time(std::string const&);

std::optional<unixtime_t> get_opt_time(pugi::xml_node const&,
                                       std::string const&);

date::sys_days parse_date(std::string const&);

}  // namespace nigiri::rt