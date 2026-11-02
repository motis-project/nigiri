#include "nigiri/loader/virt/signatures.h"

#include <algorithm>
#include <span>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/transfer_rule_sides.h"

namespace nigiri::loader {

using sig_t = std::vector<transfer_rule_side_idx>;

struct sig_hash {
  using is_transparent = void;
  cista::hash_t operator()(std::span<transfer_rule_side_idx const> s) const {
    auto h = cista::BASE_HASH;
    for (auto const x : s) {
      h = cista::hash_combine(h, x.v_);
    }
    return h;
  }
  cista::hash_t operator()(signature_idx_t const i) const {
    return (*this)(sigs_->rules_[i]);
  }
  signatures const* sigs_;
};

struct sig_equals {
  using is_transparent = void;
  bool operator()(std::span<transfer_rule_side_idx const> a,
                  signature_idx_t const b) const {
    return std::ranges::equal(a, sigs_->rules_[b]);
  }
  bool operator()(signature_idx_t const a, signature_idx_t const b) const {
    return a == b;
  }
  signatures const* sigs_;
};

signatures get_signatures(timetable const& tt) {
  auto sigs = signatures{};
  sigs.rules_.emplace_back(sig_t{});  // kEmptySig

  auto signature_of = hash_set<signature_idx_t, sig_hash, sig_equals>{
      0U, sig_hash{&sigs}, sig_equals{&sigs}};

  auto sig = sig_t{};
  auto rules = std::vector<section_rules>{};
  auto route_sigs = std::vector<std::vector<signature_idx_t>>{};

  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = tt.route_location_seq_[r];
    for (auto const t : transports) {
      auto const t_in_r_idx = to_idx(t - transports.from_);

      // Get trip rules.
      auto const sections = tt.transport_to_trip_section_[t];
      rules.resize(sections.size());
      for (auto k = 0U; k != sections.size(); ++k) {
        get_section_rules(tt, sections[k], rules[k]);
      }
      if (route_sigs.empty() &&
          utl::all_of(rules, [](auto&& x) { return x.empty(); })) {
        continue;
      }

      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const stp = stop{seq[i]};
        auto const base = stp.location_idx();

        // Gather all rules that apply to this stop.
        sig.clear();
        if (i != 0U && stp.out_allowed()) {
          auto const& arr_rules = rules[sections.size() == 1U ? 0U : i - 1U];
          add_stop_rules(tt, arr_rules, event_type::kArr, base, sig);
        }
        if (i + 1U != seq.size() && stp.in_allowed()) {
          auto const& dep_rules = rules[sections.size() == 1U ? 0U : i];
          add_stop_rules(tt, dep_rules, event_type::kDep, base, sig);
        }

        // Get or create: rules at stop -> signature_idx
        auto s = kEmptySig;
        if (!sig.empty()) {
          utl::erase_duplicates(sig);  // sort + unique
          if (auto const it = signature_of.find(sig); it != end(signature_of)) {
            // use existing signature
            s = *it;
          } else {
            // create new signature
            s = signature_idx_t{sigs.rules_.size()};
            sigs.rules_.emplace_back(sig);
            signature_of.emplace(s);
          }
        }

        if (route_sigs.empty()) {
          if (s == kEmptySig) {
            continue;
          }
          route_sigs.resize(seq.size());
        }

        auto& stop_sigs = route_sigs[i];
        if (stop_sigs.size() <= 1U) {
          auto const same = stop_sigs.empty() ? kEmptySig : stop_sigs.front();
          if (s == same) {
            continue;
          }
          if (stop_sigs.empty() && t_in_r_idx == 0U) {
            stop_sigs = {s};
            continue;
          }
          stop_sigs.assign(transports.size(), same);
        }
        stop_sigs[t_in_r_idx] = s;
      }
    }

    sigs.route_stop_sigs_.emplace_back(route_sigs);
    route_sigs.clear();
  }
  return sigs;
}

}  // namespace nigiri::loader
