#pragma once

#include <string_view>

#include "pugixml.hpp"

namespace nigiri::rt {

pugi::xml_document to_xml(std::string_view s);

}  // namespace nigiri::rt
