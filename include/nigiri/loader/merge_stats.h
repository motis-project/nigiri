#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

template <typename Idx>
struct overlap_counters {
  using hit_t = std::tuple<Idx, Idx, transport_idx_t>;
  using entry_t = std::tuple<Idx, Idx, std::uint32_t>;

  void add(Idx const a, Idx const b, transport_idx_t const t) {
    hits_.emplace_back(a, b, t);
  }

  void finish(cista::base_t<Idx> const n_entities) {
    utl::erase_duplicates(hits_);
    utl::equal_ranges_linear(
        hits_,
        [](hit_t const& a, hit_t const& b) {
          return std::get<0>(a) == std::get<0>(b) &&
                 std::get<1>(a) == std::get<1>(b);
        },
        [&](auto&& from, auto&& to) {
          entries_.emplace_back(
              std::get<0>(*from), std::get<1>(*from),
              static_cast<std::uint32_t>(std::distance(from, to)));
        });

    // a transport with duplicates in several others must count once here,
    // so the pair counts above cannot simply be summed
    auto cross = std::vector<std::pair<Idx, transport_idx_t>>{};
    cross.reserve(hits_.size());
    for (auto const& [a, b, t] : hits_) {
      if (b != a) {
        cross.emplace_back(a, t);
      }
    }
    utl::erase_duplicates(cross);
    n_duplicated_transports_.resize(n_entities);
    for (auto const& [a, t] : cross) {
      ++n_duplicated_transports_[a];
    }

    hits_ = {};
  }

  std::vector<hit_t> hits_;
  // (a, b, n): n of a's transports also exist in b. not symmetric - the same
  // two feeds with a and b swapped is a different number.
  std::vector<entry_t> entries_;  // sorted by (a, b)

  // per entity: distinct transports that also exist in some other one
  vector_map<Idx, std::uint32_t> n_duplicated_transports_;
};

struct merge_stats {
  // sizes the counters and fills in the per-agency / per-feed totals
  explicit merge_stats(timetable const&);

  overlap_counters<provider_idx_t> provider_overlap_;
  overlap_counters<source_idx_t> src_overlap_;

  vector_map<provider_idx_t, std::uint32_t> provider_n_transports_;
  vector_map<source_idx_t, std::uint32_t> src_n_transports_;

  vector_map<provider_idx_t, std::uint32_t> provider_n_absorbed_transports_;
  vector_map<source_idx_t, std::uint32_t> src_n_absorbed_transports_;

  std::uint32_t n_route_pairs_{0U};
  std::uint32_t n_merges_{0U};
  std::uint32_t n_absorbed_transports_{0U};
};

// what was found, as JSON
void write_merge_stats(timetable const&,
                       merge_stats const&,
                       std::filesystem::path const&,
                       std::vector<std::string> const& src_tags);

// the same thing as a self-contained HTML report
void write_merge_html(timetable const&,
                      merge_stats const&,
                      std::filesystem::path const&,
                      std::vector<std::string> const& src_tags);

}  // namespace nigiri::loader
