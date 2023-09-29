#pragma once

#include <cstring>

#include "nigiri/timetable.h"
#include "nigiri/types.h"
#include "utl/erase_if.h"

namespace nigiri::routing {
struct station_filter {

  struct weight_info {
    location_idx_t l_;
    int weight_;
  };

  static void percentage_filter(std::vector<start>& starts, double percent, bool fwd) {
    auto const min = [&](start const& a, start const& b) {
      return a.time_at_stop_ < b.time_at_stop_;
    };
    auto const cmp = [&](start const& a, start const& b) {
      return fwd ? b < a : a < b;
    };
    std::sort(starts.begin(), starts.end(), min);
    size_t percent_to_dismiss = static_cast<size_t>(starts.size() * percent);
    size_t new_size = starts.size() - percent_to_dismiss;
    if(starts.at(starts.size()-1).time_at_stop_ == starts.at(0).time_at_stop_ || percent_to_dismiss == 0) {
      return;
    }
    starts.resize(new_size);
    //starts.shrink_to_fit();
    std::sort(starts.begin(), starts.end(), cmp);
  }

  static void weighted_filter(std::vector<start>& starts, timetable const& tt, bool linefilter, bool fwd) {
    double threshold = 20.0;
    if(starts.size() > 80 && starts.size() < 400) {
      threshold = 18.0;
    }
    if(starts.size() > 1000) {
      threshold = 15.0;
    }
    if(starts.size() > 2000) {
      threshold = 10.0;
    }
    vector<weight_info> v_weights;
    int most = 1;
    auto const weighted = [&](start const& a) {
      location_idx_t l = a.stop_;
      for(auto const w : v_weights) {
        if(l == w.l_) {
          double percent = w.weight_ * 100.0 / most;
          return percent > threshold ? false : true;
        }
      }
      return true;
    };
    // Example:
    // dep_count = 3, local_count=2 *2=4, slow_count=1 *3=3, o=5 +4
    //      -> weight = 14
    // dep_count = 2, slow_count=1 *3=3, fast_count=1 *4=4,  o=10 +2
    //      -> weight = 11
    // Offset Weights:
    // 0-3min = 6; 3-5min = 5; 5-7min = 4; 7-10min = 3
    // 10-15min = 2; 15-20min = 1; >20min = 0
    for (auto const& s : starts) {
      auto const l = s.stop_;
      auto const o = fwd ? s.time_at_stop_ - s.time_at_start_ :
                           s.time_at_start_ - s.time_at_stop_;
      auto dep_count = 0;
      bool not_found = true;
      for(auto dc : tt.depature_count_) {
        if(dc.first == l) {
          not_found = false;
          dep_count = dc.second;
          break;
        }
      }
      if(not_found) {
        continue;
      }
      int local_count = tt.get_groupclass_count(l, group::klocal) * 2;
      int slow_count = tt.get_groupclass_count(l, group::kslow) * 3;
      int fast_count = tt.get_groupclass_count(l, group::kfast) * 4;
      int weight = local_count + slow_count + fast_count + dep_count;
      //if(o.count() >= 15 && o.count() < 20) weight += 1;
      //if(o.count() >= 10 && o.count() < 15) weight += 2;
      //if(o.count() >= 7 && o.count() < 10) weight += 3;
      //if(o.count() >= 5 && o.count() < 7) weight += 4;
      //if(o.count() >= 3 && o.count() < 5) weight += 5;
      //if(o.count() >= 0 && o.count() < 3) weight += 6;
      int extra_weight = 0;
      if(linefilter) {
        extra_weight = line_filter(starts, tt, s, fwd);
      }
      weight += extra_weight;
      weight_info wi = {l, weight};
      v_weights.emplace_back(wi);
      most = weight > most ? weight : most;
    }
    if(most == 1) {
      return;
    }
    utl::erase_if(starts, weighted);
  }

  static vector<location_idx_t> find_lines(route_idx_t find, timetable const& tt) {
    vector<location_idx_t> found_at;
    for(auto lidx = location_idx_t{0}; lidx < tt.location_routes_.size(); lidx++) {
      for(route_idx_t rix : tt.location_routes_.at(lidx)) {
        if(find == rix) {
          found_at.emplace_back(lidx);
        }
      }
    }
    return found_at;
  }

  static start find_start_from_locidx(std::vector<start>& starts, location_idx_t locidx) {
    for(start s : starts) {
      if(s.stop_ == locidx) return s;
    }
    return start();
  }

  static int line_filter(std::vector<start>& starts, timetable const& tt, start this_start, bool fwd) {
    duration_t o = fwd ? this_start.time_at_stop_ - this_start.time_at_start_ :
                         this_start.time_at_start_ - this_start.time_at_stop_;
    location_idx_t l = this_start.stop_;
    int weight_count = 0;
    duration_t dur_off;
    unixtime_t null{};
    auto const this_start_lines = tt.location_routes_.at(l);
    vector<location_idx_t> v_li;
    for(route_idx_t line : this_start_lines) {
      v_li = find_lines(line, tt);
      for (auto const a : v_li) {
        start s = find_start_from_locidx(starts, a);
        if (s.time_at_stop_ == null && s.time_at_start_ == null &&
            s.stop_ == 0) {
          continue;
        }
        dur_off = fwd ? s.time_at_stop_ - s.time_at_start_ :
                        s.time_at_start_ - s.time_at_stop_;
        if (o < dur_off || this_start.time_at_stop_ < s.time_at_stop_) {
          weight_count++;
        }
      }
    }
    return weight_count;
  }

  static void filter_stations(std::vector<start>& starts, timetable const& tt, bool fwd) {
    if(tt.percentage_filter_) {
      percentage_filter(starts, tt.percent_for_filter_, fwd);
    }
    if(tt.weighted_filter_) {
      weighted_filter(starts, tt, tt.line_filter_, fwd);
    }
  }

};
} // namespace nigiri::routing
