#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "nigiri/types.h"

namespace nigiri {

struct timetable;

constexpr auto const kWikidataBaseUrl =
    std::string_view{"https://www.wikidata.org/wiki/"};

wikidata_id parse_wikidata_id(timetable&, std::string_view);

std::optional<std::string> wikidata_url(timetable const&, wikidata_id);

}  // namespace nigiri
