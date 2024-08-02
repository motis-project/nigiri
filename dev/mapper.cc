#include <cista/mmap.h>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <format>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <ranges>
#include <string>

#include "cista/containers/mmap_vec.h"
#include "cista/containers/vecvec.h"
#include "cista/mmap.h"

constexpr std::string filepath{"dev-map.dat"};
constexpr std::string filepath2{"dev-map.v2.dat"};

template <typename T>
using mm_vec = cista::basic_mmap_vec<T, std::uint64_t>;

template <typename K, typename V, typename SizeType = cista::base_t<K>>
using mm_vecvec = cista::basic_vecvec<K, mm_vec<V>, mm_vec<SizeType>>;

void test(const std::string& path) {
  const auto& exists = std::filesystem::exists(path);
  std::cout << std::format("Path '{}' exists: {}", path, exists) << std::endl;
  if (exists) {
    std::cout << std::format("Size: {}", std::filesystem::file_size(path))
              << std::endl;
  }
}

// auto rand(auto& _) {
//     return
// }

auto setup_random(const int upper) {
  static std::random_device rd;
  static std::mt19937 mt(rd());
  static std::uniform_int_distribution<> dist(1, upper);
  return [](const auto&) { return static_cast<uint32_t>(dist(mt)); };
}

void fill(cista::mmap_vec<uint32_t>& map) {
  const std::size_t total{100u};
  map.reserve(total);
  // map.resize(15);
  test(filepath);
  auto rand = setup_random(65535);
  // const std::random_device rd;
  // const std::mt19937 mt(rd());
  // std::uniform_int_distribution<> dist(1u, 65535u);
  // const auto random = [&dist, mt](const auto&){ return dist(mt); };
  // auto show = [](const auto& x) { std::cout << x << std::endl; };
  auto size = [&map]() {
    std::cout << "map size: " << map.size() << ", " << map.used_size_
              << std::endl;
  };
  size();
  std::ranges::for_each(std::ranges::views::iota(0u, total)
                            // std::views::iota(1u, total)
                            // std::ranges::iota_view{1u, total}
                            // | std::views::transform(random)
                            | std::views::transform(rand),
                        [&map](const auto& r) { map.push_back(r); }
                        // , [](auto& x) { std::cout << x << std::endl; }
  );
  size();
  // map.resize(75);
  size();
}

void fill_map(auto& map, const std::vector<std::string>& entries) {
  for (const auto& entry : entries) {
    auto bucket = map.add_back_sized(0u);
    for (const char c : entry) {
      std::cout << std::format("Adding '{}'", c) << std::endl;
      bucket.push_back(c);
    }
  }
}

void write(const auto& path) {
  std::cout << "\nWRITING ...\n" << std::endl;
  const auto& mode = cista::mmap::protection::WRITE;
  using datatype = char;
  mm_vecvec<std::size_t, datatype> map{
      cista::basic_mmap_vec<datatype, std::size_t>{
          cista::mmap{(path + ".values").data(), mode}},
      cista::basic_mmap_vec<std::size_t, std::size_t>{
          cista::mmap{(path + ".metadata").data(), mode}}};
  std::cout << std::format("Is empty: {}", map.empty()) << std::endl;
  std::cout << std::format("Size: {}", map.size()) << std::endl;
  // map.emplace_back("ABC");
  fill_map(map, {"ABCD", "123", "Hello, world!"});
  // auto bucket = map.add_back_sized(0u);
  // for (char c : std::string{"ABCD"}) {
  //     std::cout << std::format("Adding '{}'", c) << std::endl;
  //     bucket.push_back(c);
  // }
  // // map.emplace_back(bucket);
  std::cout << std::format("Size: {}", map.size()) << std::endl;
  // map.at(0).push_back('E');
  // std::cout << std::format("Size: {}", map.size()) << std::endl;
  // map.
}
void read(const auto& path) {
  std::cout << "\nREADING ...\n" << std::endl;
  const auto& mode = cista::mmap::protection::READ;
  const mm_vecvec<std::size_t, char> map{
      cista::basic_mmap_vec<char, std::size_t>{
          cista::mmap{(path + ".values").data(), mode}},
      cista::basic_mmap_vec<std::size_t, std::size_t>{
          cista::mmap{(path + ".metadata").data(), mode}}};
  std::cout << std::format("Size: {}", map.size()) << std::endl;
  for (const auto bucket : map) {
    // for (auto& [bucket, index] : map | std::views::enumerate) {
    // for (const auto bucket : std::ranges::ref_view{map} ) {
    std::cout << "Printing bucket ..." << std::endl;
    for (auto c : bucket) {
      std::cout << std::format("Char: '{}'", c) << std::endl;
    }
  }
}

int main() {
  test(filepath);
  cista::mmap_vec<uint32_t> map{
      cista::mmap{filepath.data(), cista::mmap::protection::WRITE}};
  test(filepath);
  fill(map);
  test(filepath);
  map.resize(map.used_size_);
  test(filepath);
  // map.push_back(const unsigned int &t)
  using namespace std;
  cout << "Hello world!" << endl;
  write(filepath2);
  read(filepath2);
  return 0;
}
