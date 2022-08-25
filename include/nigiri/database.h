#pragma once

#include <cinttypes>
#include <filesystem>
#include <vector>

#include "utl/verify.h"

#include "lmdb/lmdb.hpp"

#include "cista/containers/variant.h"
#include "cista/serialization.h"
#include "cista/type_hash/type_name.h"

#include "nigiri/types.h"

namespace nigiri {

template <typename Tag>
using db_index_t = cista::strong<std::uint32_t, Tag>;

template <typename... Ts>
struct database {
  using handle_t = std::uint32_t;

  static constexpr auto const kBufMaxSize = 10'000;

  enum class init_type { KEEP, CLEAR };

  template <typename T>
  struct iterator {
    explicit iterator(database& db)
        : t_{db.env_},
          dbi_{t_.dbi_open(
              cista::canonical_type_str<std::decay_t<T>>().c_str())},
          c_{t_, dbi_} {
      c_.get(lmdb::cursor_op::FIRST);
    }

    iterator& operator++() {
      c_.get(lmdb::cursor_op::NEXT);
      return *this;
    }

    T operator*() {
      auto const entry = c_.get(lmdb::cursor_op::GET_CURRENT);
      utl::verify(entry.has_value(), "invalid iterator access to {}",
                  cista::type_str<T>());
      return cista::copy_from_potentially_unaligned<T>(entry->second);
    }

    bool is_finished() {
      return c_.get(lmdb::cursor_op::GET_CURRENT) == std::nullopt;
    }

    lmdb::txn t_;
    lmdb::txn::dbi dbi_;
    lmdb::cursor c_;
  };

  template <typename T>
  struct iterator_wrapper {
    iterator_wrapper() : data_{end_t{}} {}
    explicit iterator_wrapper(database& db) : data_{iterator<T>{db}} {}

    iterator_wrapper& operator++() {
      utl::verify(!is_end(), "database {} increment end iterator",
                  cista::type_str<T>());
      cista::get<iterator<T>>(data_).operator++();
      return *this;
    }

    T operator*() {
      utl::verify(!is_end(), "database {} dereference end iterator",
                  cista::type_str<T>());
      return cista::get<iterator<T>>(data_).operator*();
    }

    bool operator==(iterator_wrapper const& o) {
      return o.is_end() &&
             (is_end() || cista::get<iterator<T>>(data_).is_finished());
    }

    bool operator!=(iterator_wrapper const& o) { return !operator==(o); }

    bool is_end() const { return cista::holds_alternative<end_t>(data_); }

    struct end_t {};
    variant<iterator<T>, end_t> data_;
  };

  template <typename T>
  struct range {
    explicit range(database& db) : db_{db} {}
    iterator_wrapper<T> begin() { return iterator_wrapper<T>{db_}; }
    iterator_wrapper<T> end() { return {}; }
    size_t size() const { return iterator<T>{db_}.dbi_.stat().ms_entries; }
    database& db_;
  };

  explicit database(std::filesystem::path const& path,
                    std::size_t const max_size,
                    init_type const init = init_type::KEEP) {
    env_.set_maxdbs(sizeof...(Ts));
    env_.set_mapsize(max_size);
    env_.open(path.string().c_str(),
              lmdb::env_open_flags::NOSUBDIR | lmdb::env_open_flags::NOTLS);

    lmdb::txn t{env_};
    (open_dbi<Ts>(t, init), ...);
    t.commit();
  }

  template <typename T1>
  handle_t add(T1&& el) {
    constexpr auto const idx = cista::index_of_type<std::decay_t<T1>, Ts...>();
    static_assert(idx != cista::TYPE_NOT_FOUND);
    std::get<idx>(insert_buf_).emplace_back(std::forward<T1>(el));
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

  template <typename T>
  struct writer {
    using Type = std::decay_t<T>;

    explicit writer(lmdb::env& env)
        : t_{env},
          dbi_{t_.dbi_open(cista::canonical_type_str<Type>().c_str())} {}

    ~writer() { t_.commit(); }

    void write(handle_t const key, T const& el) {
      cista::serialize(buf_, el);
      t_.put(dbi_, key,
             std::string_view{reinterpret_cast<char const*>(buf_.buf_.data()),
                              buf_.buf_.size()});
      buf_.buf_.clear();
    }

    lmdb::txn t_;
    lmdb::txn::dbi dbi_;
    cista::buf<> buf_{};
  };

  template <typename T>
  writer<T> get_writer() {
    return writer<T>{env_};
  }

  void flush() { (flush<cista::index_of_type<Ts, Ts...>()>(), ...); }

  template <typename T>
  size_t size() {
    return lmdb::txn{env_}
        .dbi_open(cista::canonical_type_str<std::decay_t<T>>().c_str())
        .stat()
        .ms_entries;
  }

  template <typename T>
  range<T> iterate() {
    return range<T>{*this};
  }

private:
  template <typename T>
  void open_dbi(lmdb::txn& t, init_type const init) {
    auto d = t.dbi_open(cista::canonical_type_str<std::decay_t<T>>().c_str(),
                        lmdb::dbi_flags::CREATE | lmdb::dbi_flags::INTEGERKEY);
    if (init == init_type::CLEAR) {
      d.clear();
    }
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
