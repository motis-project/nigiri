#pragma once

#include "nigiri/timetable.h"
#include "nigiri/types.h"
#include "utl/erase_if.h"

// bool is_timepoint = std::holds_alternative<unixtime_t>(start_time);

namespace nigiri::routing {
struct station_filter {
  struct weight_info {
    location_idx_t l_;
    int weight_;
  };

  static void percentage_filter(std::vector<start>& starts, double percent) {
    auto const min = [&](start const& a, start const& b) {
      return a.time_at_stop_ <= b.time_at_stop_;
    };
    std::sort(starts.begin(), starts.end(), min);
    size_t percent_to_dismiss = static_cast<size_t>(starts.size() * percent); // percent 0.2
    size_t new_size = starts.size() - percent_to_dismiss;
    if(starts.at(starts.size()-1).time_at_stop_ == starts.at(0).time_at_stop_) {
      return;
    }
    starts.resize(new_size);
  }

  static void weighted_filter(std::vector<start>& starts, timetable const& tt) {
    vector<weight_info> v_weights;
    int most = 0;
    auto const weighted = [&](start const& a) {
      location_idx_t l = a.stop_;
      for(auto const w : v_weights) {
        if(l == w.l_) {
          double percent = w.weight_ * 100.0 / most;
          if(percent > 20.0) {
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
    // 6 = 0-3min; 5 = 3-5min; 4 = 5-7min; 3 = 7-10min; 2 = 10-15min; 1 = 15-20min; 0 = > 20
    for (auto const& s : starts) {
      auto const l = s.stop_;
      auto const o = s.time_at_stop_ - s.time_at_start_;
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
      weight_info wi = {l, weight};
      v_weights.emplace_back(wi);
      most = weight > most ? weight : most;
    }
    //auto it = std::remove_if(starts.begin(), starts.end(), weighted);
    utl::erase_if(starts, weighted);
  }

  static void line_filter(std::vector<start>& starts, timetable const& tt) {

  }

  static void filter_stations(std::vector<start>& starts, timetable const& tt) {
    // entweder percentage filter (einfachster)
    percentage_filter(starts, 0.2);
    // oder weighted -> da ist percentage ein element von
    weighted_filter(starts, tt);
    //
    line_filter(starts, tt);
  }


};
} // namespace nigiri::routing
