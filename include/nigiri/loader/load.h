#pragma once

#include <filesystem>
#include <vector>

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

timetable load(std::vector<std::filesystem::path> const&);

}  // namespace nigiri::loader