#include "nigiri/transfer_rule_sides.h"

#include <algorithm>
#include <vector>

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/to_vec.h"

#include "nigiri/footpath.h"
#include "nigiri/timetable.h"

namespace nigiri {

void add_trip_rules(timetable const& tt,
                    std::span<trip_idx_t const> trips,
                    std::vector<transfer_rule_side_idx>& out) {
  auto const& tr = tt.transfer_rules_;
  for (auto const trip : trips) {
    auto const trip_rules = values_of(tr.trip_rules_, trip);
    out.insert(end(out), trip_rules.begin(), trip_rules.end());
    auto const src = tt.trip_id_src_[tt.trip_ids_[trip].front()];
    for (auto const s : values_of(tr.route_rules_, tt.trip_route_id_[trip])) {
      if (tr.rules_[s.rule()].src_ == src) {
        out.push_back(s);
      }
    }
  }
}

void get_signature(timetable const& tt,
                   std::span<transfer_rule_side_idx const> arriving,
                   std::span<transfer_rule_side_idx const> departing,
                   bool const is_handover,
                   location_idx_t const base,
                   std::vector<transfer_rule_side_idx>& sig) {
  auto const& rules = tt.transfer_rules_.rules_;
  sig.clear();
  auto const add = [&](transfer_rule_side_idx const s, bool const is_arriving) {
    if (is_handover && s.is_from() != is_arriving) {
      // stay seated / block_id transfers -> only keep:
      // - the from side matching the arriving trip
      // - the to side matching the departing trip
      return;
    }

    auto const& r = rules[s.rule()];
    if (covers(tt, s.is_from() ? r.from_stop_ : r.to_stop_, base)) {
      sig.push_back(s);
    }
  };
  for (auto const s : arriving) {
    add(s, true);
  }
  for (auto const s : departing) {
    add(s, false);
  }
  utl::erase_duplicates(sig);
}

duration_t get_transfer_time(timetable const& tt,
                             std::span<transfer_rule_side_idx const> sig,
                             location_idx_t const base) {
  auto const& rules = tt.transfer_rules_.rules_;
  auto const applies = [&](transfer_rule_idx_t const rule, bool const is_from) {
    auto const& r = rules[rule];
    return (is_from ? r.from_qualified() : r.to_qualified())
               ? std::binary_search(begin(sig), end(sig),
                                    transfer_rule_side_idx{rule, is_from})
               : covers(tt, is_from ? r.from_stop_ : r.to_stop_, base);
  };
  auto best = transfer_rule_idx_t::invalid();
  for (auto const s : sig) {
    auto const rule = s.rule();
    if (applies(rule, true) && applies(rule, false)) {
      best = best == transfer_rule_idx_t::invalid()
                 ? rule
                 : get_more_specific(tt, best, rule);
    }
  }
  return best != transfer_rule_idx_t::invalid()
             ? rules[best].duration_
             : to_fp_duration(tt.locations_.transfer_time_[base]);
}

std::vector<transfer_rule_side> get_transfer_rule_sides(
    timetable const& tt, std::span<transfer_rule_side_idx const> rules) {
  // Transform to rule sides.
  auto v = utl::to_vec(rules, [&](transfer_rule_side_idx const s) {
    auto const& r = tt.transfer_rules_.rules_[s.rule()];
    auto const is_from = s.is_from();
    return transfer_rule_side{
        .is_from_ = is_from,
        .rule_stop_ = is_from ? r.from_stop_ : r.to_stop_,
        .other_stop_ = is_from ? r.to_stop_ : r.from_stop_,
        .src_ = r.src_,
        .other_route_ = is_from ? r.to_route_ : r.from_route_,
        .other_trip_ = is_from ? r.to_trip_ : r.from_trip_,
        .duration_ = r.duration_,
        .specificity_ = r.specificity_};
  });

  // Lower each side's specificity to the lowest value that keeps its order
  // against competing rule sides (same partners, different duration): the same
  // rules win, but more stops get the same key and share a virtual location.
  auto const& parents = tt.locations_.parents_;
  auto const same_partners = [&](transfer_rule_side const& a,
                                 transfer_rule_side const& b) {
    return a.is_from_ == b.is_from_ &&
           (a.other_stop_ == b.other_stop_ ||
            parents[a.other_stop_] == b.other_stop_ ||
            parents[b.other_stop_] == a.other_stop_);
  };
  utl::sort(v, [](transfer_rule_side const& a, transfer_rule_side const& b) {
    return a.specificity_ < b.specificity_;
  });
  utl::equal_ranges_linear(
      v,
      [](transfer_rule_side const& a, transfer_rule_side const& b) {
        return a.specificity_ == b.specificity_;
      },
      [&](auto const from, auto const to) {
        for (auto a = from; a != to; ++a) {
          auto lowered = transfer_rule_specificity_t{0U};

          // Iterate rules with lower specificity.
          // Higher specificity rules win anyway -> no need to check them.
          for (auto b = begin(v); b != from; ++b) {
            if (b->duration_ != a->duration_ && same_partners(*a, *b)) {
              // b has a lower specificity, another duration, and can apply to
              // the same transfers -> stay above it, otherwise a stop where
              // this side loses against b would get the same key and share the
              // virtual location (incorrect)
              lowered =
                  std::max(lowered, static_cast<transfer_rule_specificity_t>(
                                        b->specificity_ + 1U));
            }
          }

          a->specificity_ = lowered;
        }
      });

  utl::erase_duplicates(v);

  return v;
}

virt_key get_virt_key(timetable const& tt,
                      std::span<transfer_rule_side_idx const> sig,
                      location_idx_t const base) {
  return {.base_ = base,
          .transfer_time_ = to_transfer_time(get_transfer_time(tt, sig, base)),
          .sides_ = get_transfer_rule_sides(tt, sig)};
}

}  // namespace nigiri
