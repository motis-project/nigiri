#pragma once

#include <string>

namespace nigiri::rt {

std::string json_to_protobuf(std::string const& json);
std::string protobuf_to_json(std::string const& protobuf);

}  // namespace nigiri::rt