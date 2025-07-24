#pragma once

#include <string>

namespace nigiri::rt {

std::string json_to_protobuf(std::string_view);
std::string protobuf_to_json(std::string_view);

}  // namespace nigiri::rt