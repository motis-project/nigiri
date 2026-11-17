#include "nigiri/wikidata.h"

#include <charconv>

#include "fmt/format.h"

#include "nigiri/timetable.h"

namespace nigiri {

wikidata_id parse_wikidata_id(timetable& tt, std::string_view const s) {
  auto const q = s.rfind('Q');
  if (q == std::string_view::npos) {
    return {};
  }
  auto item = std::uint32_t{0U};
  auto const end = s.data() + s.size();
  auto const [ptr, ec] = std::from_chars(s.data() + q + 1U, end, item);
  if (ec != std::errc{} || ptr != end || item == 0U) {
    return {};
  }

  auto const base = s.substr(0U, q);
  if (base.empty() || base == kWikidataBaseUrl) {
    return {.item_ = item, .wikibase_ = wikibase_idx_t{0U}};
  }
  if (base.find("://") == std::string_view::npos) {
    return {};
  }
  if (tt.wikibase_urls_.empty()) {
    tt.wikibase_urls_.emplace_back(kWikidataBaseUrl);
  }
  for (auto i = wikibase_idx_t{0U}; i != tt.wikibase_urls_.size(); ++i) {
    if (tt.wikibase_urls_[i].view() == base) {
      return {.item_ = item, .wikibase_ = i};
    }
  }
  tt.wikibase_urls_.emplace_back(base);
  return {.item_ = item,
          .wikibase_ = wikibase_idx_t{tt.wikibase_urls_.size() - 1U}};
}

std::optional<std::string> wikidata_url(timetable const& tt,
                                        wikidata_id const id) {
  if (!id.valid()) {
    return std::nullopt;
  }
  return fmt::format("{}Q{}",
                     id.wikibase_ == wikibase_idx_t{0U}
                         ? kWikidataBaseUrl
                         : tt.wikibase_urls_[id.wikibase_].view(),
                     id.item_);
}

}  // namespace nigiri
