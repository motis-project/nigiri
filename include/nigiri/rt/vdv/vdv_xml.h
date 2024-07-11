#pragma once

#include "pugixml.hpp"

namespace nigiri::rt {

struct vdv_run;

vdv_run parse_run(pugi::xml_node const&);

}  // namespace nigiri::rt