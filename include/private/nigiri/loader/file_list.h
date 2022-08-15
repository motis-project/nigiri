#pragma once

#include <optional>
#include <string_view>
#include <vector>

namespace nigiri::loader {

using file_list = std::vector<std::optional<std::string_view>>;

}  // namespace nigiri::loader