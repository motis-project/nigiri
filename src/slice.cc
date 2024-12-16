#include "nigiri/slice.h"

#include <ranges>

#include "utl/get_or_create.h"

#include "nigiri/timetable.h"

namespace nigiri {

slice::slice(nigiri::timetable const& tt,
             interval<nigiri::day_idx_t> extract_interval)
    : base_day_{extract_interval.from_} {
  using stop_seq_view_t = std::basic_string_view<stop::value_type>;
  using route_times_cache_t =
      paged_vecvec_helper<std::uint32_t, delta_t, std::uint16_t, 8U,
                          1U << 15U>::type;

  auto const to_delta = [&](unixtime_t const t) {
    return unix_to_delta(tt.internal_interval_days().from_, t);
  };

  auto get_transport_times =
      [&, times = std::vector<delta_t>{}](
          route_idx_t const r,
          transport const t) mutable -> std::vector<delta_t> const& {
    auto const stop_indices =
        interval{stop_idx_t{0U},
                 static_cast<stop_idx_t>(tt.route_location_seq_[r].size())};
    auto i = 0U;
    times.resize(static_cast<unsigned>(stop_indices.size()) * 2U - 2U);
    for (auto const [a, b] : utl::pairwise(stop_indices)) {
      times[i++] = to_delta(tt.event_time(t, a, event_type::kDep));
      times[i++] = to_delta(tt.event_time(t, b, event_type::kArr));
    }
    return times;
  };

  auto const get_index = [&](route_times_cache_t const& route_times,
                             route_idx_t const r_idx,
                             std::vector<delta_t> const& transport_times)
      -> std::optional<std::uint32_t> {
    auto const index = static_cast<unsigned>(std::distance(
        begin(route_times),
        std::lower_bound(begin(route_times), end(route_times), transport_times,
                         [](route_times_cache_t::const_bucket const a,
                            std::vector<delta_t> const& b) {
                           return std::ranges::lexicographical_compare(a, b);
                         })));
    for (auto i = 0U; i != tt.route_location_seq_[r_idx].size(); ++i) {
      auto const is_earlier_eq =
          index > 0 && transport_times[i] < route_times[index - 1U][i];
      auto const is_later_eq = index < route_times.size() &&
                               transport_times[i] > route_times[index][i];
      if (is_earlier_eq || is_later_eq) {
        return std::nullopt;
      }
    }
    return index;
  };

  location_l_.resize(tt.n_locations(), l_idx_t::invalid());

  auto stop_seq_rs = hash_map<stop_seq_view_t, std::vector<r_idx_t>>{};
  auto r_l_seq = vecvec<r_idx_t, l_idx_t>{};
  auto r_t_times = vector_map<r_idx_t, route_times_cache_t>{};
  auto l_r = paged_vecvec<l_idx_t, r_idx_t>{};

  auto const get_or_create_l = [&](location_idx_t const l) {
    if (location_l_[l] == l_idx_t::invalid()) {
      auto const next = l_idx_t{l_location_.size()};
      location_l_[l] = next;
      l_location_.push_back(l);
    }
    return location_l_[l];
  };

  auto const get_or_create_stop = [&](stop::value_type const v) {
    return get_or_create_l(stop{v}.location_idx());
  };

  auto const add_transport = [&](std::vector<r_idx_t>& r_candidates,
                                 route_idx_t const r, transport const tr) {
    auto const& times = get_transport_times(r, tr);
    for (auto const& r_candidate : r_candidates) {
      if (auto const index = get_index(r_t_times[r_candidate], r, times);
          index.has_value()) {
        r_t_times[r_candidate].insert(*index, times);
        return;
      }
    }

    auto const next = r_idx_t{r_l_seq.size()};
    r_l_seq.emplace_back(tt.route_location_seq_[r].view() |
                         std::views::transform(get_or_create_stop));
    r_t_times.emplace_back().emplace_back(times);
    r_candidates.emplace_back(next);

    for (auto const s : r_l_seq_[next]) {
      l_r[get_or_create_stop(s)].push_back(next);
    }
  };

  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto& r_candidates = stop_seq_rs[tt.route_location_seq_[r]];
    auto const last_stop_idx =
        static_cast<stop_idx_t>(tt.route_location_seq_[r].size() - 1U);
    for (auto const t : tt.route_transport_ranges_[r]) {
      auto const day_extend =
          tt.event_mam(r, t, last_stop_idx, event_type::kArr).days_;
      auto const first_day_interval =
          interval{std::max(day_idx_t{0U}, extract_interval.from_ - day_extend),
                   extract_interval.to_};
      for (auto const day : first_day_interval) {
        if (!tt.bitfields_[tt.transport_traffic_days_[t]].test(to_idx(day))) {
          continue;
        }
        add_transport(r_candidates, r, transport{t, day});
      }
    }
  }

  auto const filter_and_translate_footpaths =
      [&](vecvec<location_idx_t, footpath> const& fps) {
        auto ret = vecvec<l_idx_t, fp>{};
        for (auto l = l_idx_t{0U}; l != l_location_.size(); ++l) {
          using namespace std::views;
          ret.emplace_back(
              fps[l_location_[l]]  //
              | drop_while([&](footpath const x) {
                  return location_l_[x.target()] != l_idx_t::invalid();
                })  //
              | transform([&](footpath const x) {
                  return fp{.target_ = to_idx(location_l_[x.target()]),
                            .duration_ = static_cast<l_idx_t::value_t>(
                                x.duration().count())};
                }));
        }
        return ret;
      };
  for (auto [full, filtered] :
       utl::zip(tt.locations_.footpaths_out_, footpaths_out_)) {
    filtered = filter_and_translate_footpaths(full);
  }
  for (auto [full, filtered] :
       utl::zip(tt.locations_.footpaths_in_, footpaths_in_)) {
    filtered = filter_and_translate_footpaths(full);
  }
}

}  // namespace nigiri