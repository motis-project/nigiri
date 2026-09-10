#include "nigiri/loader/merge_stats.h"

#include <fstream>

#include "boost/json.hpp"

#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace json = boost::json;

namespace nigiri::loader {

merge_stats::merge_stats(timetable const& tt) {
  provider_n_transports_.resize(tt.n_agencies());
  src_n_transports_.resize(tt.n_sources());
  provider_n_absorbed_transports_.resize(tt.n_agencies());
  src_n_absorbed_transports_.resize(tt.n_sources());

  for (auto t = transport_idx_t{0U}; t != tt.next_transport_idx(); ++t) {
    auto const p = tt.transport_section_providers_[t].front();
    ++provider_n_transports_[p];
    ++src_n_transports_[tt.providers_[p].src_];
  }
}

void write_merge_stats(timetable const& tt,
                       merge_stats const& s,
                       std::filesystem::path const& out,
                       vector_map<source_idx_t, std::string> const& src_tags) {
  utl::verify(src_tags.size() == tt.n_sources(),
              "merge stats: {} tags for {} sources", src_tags.size(),
              tt.n_sources());
  auto const tag = [&](source_idx_t const src) { return src_tags[src]; };
  auto const unique = [](std::uint32_t const total, std::uint32_t const dup) {
    return total > dup ? total - dup : 0U;
  };

  auto srcs = json::array{};
  for (auto src = source_idx_t{0U}; src != s.src_n_transports_.size(); ++src) {
    auto const total = s.src_n_transports_[src];
    if (total == 0U) {
      continue;
    }
    srcs.emplace_back(json::object{
        {"src", tag(src)},
        {"transports", total},
        {"absorbed", s.src_n_absorbed_transports_[src]},
        {"dup_other", s.src_overlap_.n_duplicated_transports_[src]},
        {"unique",
         unique(total, s.src_overlap_.n_duplicated_transports_[src])}});
  }

  auto src_pairs = json::array{};
  for (auto const& [a, b, n] : s.src_overlap_.entries_) {
    src_pairs.emplace_back(
        json::object{{"a", tag(a)}, {"b", tag(b)}, {"n", n}});
  }

  auto providers = json::array{};
  for (auto p = provider_idx_t{0U}; p != s.provider_n_transports_.size(); ++p) {
    auto const total = s.provider_n_transports_[p];
    if (total == 0U) {
      continue;
    }
    auto const dup = s.provider_overlap_.n_duplicated_transports_[p];
    providers.emplace_back(json::object{
        {"provider", to_idx(p)},
        {"src", tag(tt.providers_[p].src_)},
        {"name", tt.get_default_translation(tt.providers_[p].name_)},
        {"transports", total},
        {"absorbed", s.provider_n_absorbed_transports_[p]},
        {"dup_other", dup},
        {"unique", unique(total, dup)}});
  }

  auto provider_pairs = json::array{};
  for (auto const& [a, b, n] : s.provider_overlap_.entries_) {
    provider_pairs.emplace_back(
        json::object{{"a", to_idx(a)}, {"b", to_idx(b)}, {"n", n}});
  }

  auto const doc = json::object{{"merges", s.n_merges_},
                                {"absorbed", s.n_absorbed_transports_},
                                {"route_pairs", s.n_route_pairs_},
                                {"sources", std::move(srcs)},
                                {"source_pairs", std::move(src_pairs)},
                                {"providers", std::move(providers)},
                                {"provider_pairs", std::move(provider_pairs)}};

  auto f = std::ofstream{out};
  f.exceptions(std::ios::failbit | std::ios::badbit);
  f << json::serialize(doc);

  log(log_lvl::info, "loader.merge",
      "merge stats: merges={} absorbed={} route_pairs={} -> {}", s.n_merges_,
      s.n_absorbed_transports_, s.n_route_pairs_, out.string());
}

void write_merge_html(timetable const& tt,
                      merge_stats const& s,
                      std::filesystem::path const& out,
                      vector_map<source_idx_t, std::string> const& src_tags) {
  utl::verify(src_tags.size() == tt.n_sources(),
              "merge stats: {} tags for {} sources", src_tags.size(),
              tt.n_sources());

  auto const tag = [&](source_idx_t const src) { return src_tags[src]; };
  auto const esc = [](std::string_view const in) {
    auto o = std::string{};
    for (auto const c : in) {
      switch (c) {
        case '&': o += "&amp;"; break;
        case '<': o += "&lt;"; break;
        case '>': o += "&gt;"; break;
        case '"': o += "&quot;"; break;
        default: o += c;
      }
    }
    return o;
  };
  auto const pct = [](std::uint32_t const n, std::uint32_t const total) {
    if (total == 0U) {
      return std::string{"-"};
    }
    auto const p = 100.0 * n / total;
    return n != 0U && p < 0.05 ? fmt::format("{} (< 0.1%)", n)
                               : fmt::format("{} ({:.1f}%)", n, p);
  };

  // Create reverse mapping for feed + agency pairs.
  auto src_n = hash_map<std::pair<source_idx_t, source_idx_t>, std::uint32_t>{};
  for (auto const& [a, b, n] : s.src_overlap_.entries_) {
    src_n[{a, b}] = n;
  }
  auto provider_n =
      hash_map<std::pair<provider_idx_t, provider_idx_t>, std::uint32_t>{};
  for (auto const& [a, b, n] : s.provider_overlap_.entries_) {
    provider_n[{a, b}] = n;
  }
  auto agencies = hash_map<
      std::pair<source_idx_t, source_idx_t>,
      std::vector<std::tuple<provider_idx_t, provider_idx_t, std::uint32_t>>>{};
  for (auto const& [a, b, n] : s.provider_overlap_.entries_) {
    agencies[{tt.providers_[a].src_, tt.providers_[b].src_}].emplace_back(a, b,
                                                                          n);
  }

  auto rows =
      std::vector<std::tuple<source_idx_t, source_idx_t, std::uint32_t>>{};
  for (auto const& [a, b, n] : s.src_overlap_.entries_) {
    if (a != b) {
      rows.emplace_back(a, b, n);
    }
  }

  // Sort by unique share.
  auto const uniq_share = [&](source_idx_t const src) {
    auto const t = s.src_n_transports_[src];
    return t == 0U
               ? 1.0
               : 1.0 * (t - s.src_overlap_.n_duplicated_transports_[src]) / t;
  };

  utl::sort(rows, [&](auto const& x, auto const& y) {
    auto const [xa, xb, xn] = x;
    auto const [ya, yb, yn] = y;
    return std::tuple{uniq_share(xa), tag(xa), yn} <
           std::tuple{uniq_share(ya), tag(ya), xn};
  });

  // Open file.
  auto f = std::ofstream{out};
  f.exceptions(std::ios::failbit | std::ios::badbit);

  // Write HTML header + table header.
  f << R"(<!doctype html><meta charset="utf-8"><title>merge stats</title>
<style>
body{font-family:system-ui,sans-serif;font-size:14px;margin:16px}
table{width:100%;table-layout:fixed;border-collapse:collapse}
th,td{border:1px solid #ccc;padding:2px 5px;text-align:left}
tr.f{cursor:pointer}
tr.a{background:#f7f7f7}
tr.a td:first-child{padding-left:20px}
tr.a[hidden]{display:none}
</style>
<table>
  <colgroup>
    <col>
    <col style="width:6em">
    <col style="width:6em">
    <col>
    <col style="width:7em">
    <col style="width:7em">
  </colgroup>
  <tr>
    <th>feed A</th>
    <th>A in B</th>
    <th>B in A</th>
    <th>feed B</th>
    <th>unique A</th>
    <th>unique B</th>
  </tr>
)";

  auto i = 0U;
  for (auto const& [a, b, n] : rows) {
    auto const back = src_n.find({b, a});
    auto const uniq = [&](source_idx_t const x) {
      auto const t = s.src_n_transports_[x];
      return pct(t - s.src_overlap_.n_duplicated_transports_[x], t);
    };
    f << fmt::format(
        R"(  <tr class="f" onclick='t({0})'>
    <td>{1}</td>
    <td>{2}</td>
    <td>{3}</td>
    <td>{4}</td>
    <td>{5}</td>
    <td>{6}</td>
  </tr>
)",
        i, esc(tag(a)), pct(n, s.src_n_transports_[a]),
        pct(back == end(src_n) ? 0U : back->second, s.src_n_transports_[b]),
        esc(tag(b)), uniq(a), uniq(b));

    auto const it = agencies.find({a, b});
    if (it != end(agencies)) {
      for (auto const& [po, pp, pn] : it->second) {
        f << fmt::format(
            R"(  <tr class="a" data-r="{0}" hidden>
    <td>{1}</td>
    <td>{2}</td>
    <td>{3}</td>
    <td>{4}</td>
    <td>{5}</td>
    <td>{6}</td>
  </tr>
)",
            i, esc(tt.get_default_translation(tt.providers_[po].name_)),
            pct(pn, s.provider_n_transports_[po]),
            pct(provider_n.contains({pp, po}) ? provider_n.at({pp, po}) : 0U,
                s.provider_n_transports_[pp]),
            esc(tt.get_default_translation(tt.providers_[pp].name_)),
            pct(s.provider_n_transports_[po] -
                    s.provider_overlap_.n_duplicated_transports_[po],
                s.provider_n_transports_[po]),
            pct(s.provider_n_transports_[pp] -
                    s.provider_overlap_.n_duplicated_transports_[pp],
                s.provider_n_transports_[pp]));
      }
    }
    ++i;
  }

  f << R"(</table>
<script>
function t(i) {
  const rows = document.querySelectorAll('tr.a[data-r="' + i + '"]');
  for (const r of rows) {
    r.hidden = !r.hidden;
  }
}
</script>
)";
}

}  // namespace nigiri::loader
