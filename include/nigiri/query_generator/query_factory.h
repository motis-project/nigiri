#pragma once

#include "nigiri/routing/query.h"
#include "generator_settings.h"

namespace nigiri::query_generation {

struct query_factory {
  routing::query make_query() {
    routing::query q;
    q.start_match_mode_ = settings_.start_match_mode_;
    q.dest_match_mode_ = settings_.dest_match_mode_;
    q.use_start_footpaths_ = settings_.use_start_footpaths_;
    q.max_transfers_ = settings_.max_transfers_;
    q.min_connection_count_ = settings_.min_connection_count_;
    q.extend_interval_earlier_ = settings_.extend_interval_earlier_;
    q.extend_interval_later_ = settings_.extend_interval_later_;
    q.prf_idx_ = settings_.prf_idx_;
    q.allowed_claszes_ = settings_.allowed_claszes_;
    return q;
  }

  generator_settings const& settings_;
};

}  // namespace nigiri::query_generation