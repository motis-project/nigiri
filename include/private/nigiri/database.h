#pragma once

#include <cinttypes>
#include <filesystem>
#include <vector>

#include "utl/verify.h"

#include "lmdb/lmdb.hpp"

#include "cista/containers/variant.h"
#include "cista/serialization.h"
#include "cista/type_hash/type_name.h"

namespace nigiri {

template <typename... Ts>
struct database {
  using handle_t = std::uint32_t;

  static constexpr auto const kBufMaxSize = 10'000;

  explicit database(std::filesystem::path const& path,
                    std::size_t const max_size) {
    env_.set_maxdbs(sizeof...(Ts));
    env_.set_mapsize(max_size);
    env_.open(path.c_str(),
              lmdb::env_open_flags::NOSUBDIR | lmdb::env_open_flags::NOTLS);

    lmdb::txn t{env_};
    (open_dbi<Ts>(t), ...);
    t.commit();
  }

  template <typename T>
  handle_t add(T&& el) {
    constexpr auto const idx = cista::index_of_type<T, Ts...>();
    std::get<idx>(insert_buf_).emplace_back(std::forward<T>(el));
    flush_if_full<idx>();
    return next_handle_[idx]++;
  }

  template <typename T>
  T get(handle_t const h) {
    using Type = std::decay_t<T>;
    lmdb::txn t{env_};
    auto dbi = t.dbi_open(cista::canonical_type_str<Type>().c_str());
    auto const entry = t.get(dbi, h);
    utl::verify(entry.has_value(),
                "nigiri::database type={}, key={} does not exist",
                cista::canonical_type_str<Type>(), h);
    return cista::copy_from_potentially_unaligned<T>(*entry);
  }

  void flush() { (flush<cista::index_of_type<Ts, Ts...>()>(), ...); }

private:
  template <typename T>
  void open_dbi(lmdb::txn& t) {
    t.dbi_open(cista::canonical_type_str<std::decay_t<T>>().c_str(),
               lmdb::dbi_flags::CREATE | lmdb::dbi_flags::INTEGERKEY);
  }

  template <handle_t Idx>
  void flush_if_full() {
    auto& buf = std::get<Idx>(insert_buf_);
    if (buf.size() < kBufMaxSize) {
      return;
    }

    flush<Idx>();
  }

  template <handle_t Idx>
  void flush() {
    using Type = std::decay_t<cista::type_at_index_t<Idx, Ts...>>;

    lmdb::txn t{env_};
    auto& buf = std::get<Idx>(insert_buf_);
    auto serialization_buf = cista::buf{};
    auto dbi = t.dbi_open(cista::canonical_type_str<Type>().c_str());
    for (auto const& el : buf) {
      cista::serialize(serialization_buf, el);
      t.put(dbi, next_handle_to_write_[Idx]++,
            std::string_view{
                reinterpret_cast<char const*>(serialization_buf.buf_.data()),
                serialization_buf.buf_.size()});
      serialization_buf.buf_.clear();
    }
    t.commit();
    buf.clear();
  }

  lmdb::env env_;
  std::tuple<std::vector<Ts>...> insert_buf_;
  std::array<handle_t, sizeof...(Ts)> next_handle_to_write_{};
  std::array<handle_t, sizeof...(Ts)> next_handle_{};
};

}  // namespace nigiri
