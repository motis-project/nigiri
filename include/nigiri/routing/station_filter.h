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

  static void percentage_filter(std::vector<start>& starts, double percent) {
    printf("in percentage_filter\n");
    auto const min = [&](start const& a, start const& b) {
      return a.time_at_stop_ < b.time_at_stop_;
    };
    std::sort(starts.begin(), starts.end(), min);
    size_t percent_to_dismiss = static_cast<size_t>(starts.size() * percent);
    size_t new_size = starts.size() - percent_to_dismiss;
    if(starts.at(starts.size()-1).time_at_stop_ == starts.at(0).time_at_stop_) {
      return;
    }
    starts.resize(new_size);
    starts.shrink_to_fit();
  }

  static void weighted_filter(std::vector<start>& starts, timetable const& tt, bool linefilter, bool fwd) {
    printf("in weighted_filter\n");
    vector<weight_info> v_weights;
    int most = 1;
    auto const weighted = [&](start const& a) {
      location_idx_t l = a.stop_;
      for(auto const w : v_weights) {
        if(l == w.l_) {
          double percent = w.weight_ * 100.0 / most;
          if(percent > 22.0) {
            return false;
          }
          else return true;
        }
      }
      return true;
    };
    // Example:
    // dep_count = 3, local_count=2, slow_count=1*2=2,    o = 5   -> weight = 8,3
    // dep_count = 2, slow_count=1*2=2, fast_count=1*3=3, o = 10  -> weight = 7,2
    // 6 = 0-3min; 5 = 3-5min; 4 = 5-7min; 3 = 7-10min; 2 = 10-15min; 1 = 15-20min; 0 = > 20min
    for (auto const& s : starts) {
      auto const l = s.stop_;
      auto const o = fwd ? s.time_at_stop_ - s.time_at_start_ :
                           s.time_at_start_ - s.time_at_stop_;
      size_t dep_count = tt.depature_count_.at(l);
      int local_count = tt.get_groupclass_count(l, group::klocal);
      int slow_count = tt.get_groupclass_count(l, group::kslow) * 2;
      int fast_count = tt.get_groupclass_count(l, group::kfast) * 3;
      int weight = local_count + slow_count + fast_count + (dep_count/10);
      if(o.count() >= 15 && o.count() < 20) weight += 1;
      if(o.count() >= 10 && o.count() < 15) weight += 2;
      if(o.count() >= 7 && o.count() < 10) weight += 3;
      if(o.count() >= 5 && o.count() < 7) weight += 4;
      if(o.count() >= 3 && o.count() < 5) weight += 5;
      if(o.count() >= 0 && o.count() < 3) weight += 6;
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
    printf("in line_filter\n");
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
        if (o < dur_off) {
          weight_count++;
        }
      }
    }
    return weight_count;
  }

  static void filter_stations(std::vector<start>& starts, timetable const& tt, bool fwd) {
    printf("1 Anzahl starts vorher: %llu \n", starts.size());
    if(tt.percentage_filter_) {
      percentage_filter(starts, tt.percent_for_filter_);
    }
    if(tt.weighted_filter_) {
      weighted_filter(starts, tt, tt.line_filter_, fwd);
    }
    printf("nachher: %llu \n", starts.size());
  }

};
} // namespace nigiri::routing
