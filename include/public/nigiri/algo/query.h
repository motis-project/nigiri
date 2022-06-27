#pragma once

#include "nigiri/types.h"

namespace nigiri::algo {

enum class OntripType : bool { Station, Train };

enum class SearchDir : bool { FWD, BWD };

enum class SourceMetas : bool { On, Off };

enum class TargetMetas : bool { On, Off };

enum class StartFootpaths : bool { On, Off };

enum class ResponseType : bool { Journey, Data };

namespace detail {

template <SearchDir Dir, SourceMetas SourceMetas, TargetMetas TargetMetas,
          StartFootpaths StartFootpaths, ResponseType ResponseType>
struct base_query {
  constexpr const static auto dir = Dir;
  constexpr const static auto source_metas = SourceMetas;
  constexpr const static auto target_metas = TargetMetas;
  constexpr const static auto start_footpaths = StartFootpaths;
  constexpr const static auto response_type = ResponseType;

  base_query(location_idx_t const source, location_idx_t const target,
             unixtime_t const begin)
      : source_{source}, target_{target}, begin_{begin} {}

  location_idx_t source_{};
  location_idx_t target_{};
  unixtime_t begin_{};
};

}  // namespace detail

template <OntripType Type, SearchDir Dir, SourceMetas SourceMetas,
          TargetMetas TargetMetas, StartFootpaths StartFootpaths,
          ResponseType ResponseType>
struct ontrip_query_gen : detail::base_query<Dir, SourceMetas, TargetMetas,
                                             StartFootpaths, ResponseType> {
  static_assert(SourceMetas == SourceMetas::Off);

  ontrip_query_gen(location_idx_t const source, location_idx_t const target,
                   unixtime_t const begin)
      : detail::base_query<Dir, SourceMetas, TargetMetas, StartFootpaths,
                           ResponseType>(source, target, begin) {}
};

template <SearchDir Dir, SourceMetas SourceMetas, TargetMetas TargetMetas,
          StartFootpaths StartFootpaths, ResponseType ResponseType>
struct pretrip_query_gen : detail::base_query<Dir, SourceMetas, TargetMetas,
                                              StartFootpaths, ResponseType> {

  pretrip_query_gen(location_idx_t const source, location_idx_t const target,
                    unixtime_t const begin, unixtime_t const end)
      : detail::base_query<Dir, SourceMetas, TargetMetas, StartFootpaths,
                           ResponseType>(source, target, begin),
        end_{end} {}

  unixtime_t end_;
};

using ontrip_query_fail =
    ontrip_query_gen<OntripType::Station, SearchDir::FWD, SourceMetas::On,
                     TargetMetas::On, StartFootpaths::On,
                     ResponseType::Journey>;

using ontrip_query =
    ontrip_query_gen<OntripType::Station, SearchDir::FWD, SourceMetas::Off,
                     TargetMetas::On, StartFootpaths::On,
                     ResponseType::Journey>;

using ontrip_train_query =
    ontrip_query_gen<OntripType::Train, SearchDir::FWD, SourceMetas::Off,
                     TargetMetas::On, StartFootpaths::On,
                     ResponseType::Journey>;

using pretrip_query =
    pretrip_query_gen<SearchDir::FWD, SourceMetas::On, TargetMetas::Off,
                      StartFootpaths::On, ResponseType::Journey>;

using pretrip_bwd_query =
    pretrip_query_gen<SearchDir::BWD, SourceMetas::On, TargetMetas::Off,
                      StartFootpaths::On, ResponseType::Journey>;

}  // namespace nigiri::algo
