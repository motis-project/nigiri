# Nigiri Development Guidelines

## Build & Test Commands
- Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build: `cmake --build build`
- Run all tests: `./build/bin/nigiri-test`
- Run specific test: `./build/bin/nigiri-test --gtest_filter=TestSuiteName.TestName`
- Lint: `cmake -S . -B build -DNIGIRI_LINT=ON && cmake --build build`
- Benchmark: `./build/bin/nigiri-benchmark`
- QA: `./build/bin/nigiri-qa`

## Code Style
- Use snake_case for everything except template parameters (CamelCase)
- Class/struct members have underscore suffix (`_`)
- 2-space indentation
- Standard C++23 features preferred
- Error handling: return values/error codes preferred over exceptions
- Braces required for control structures
- Comprehensive unit tests expected using GoogleTest
- Pass immutable data by const reference
- Use modern C++ features (auto, range-based for loops, etc.)
- Heavy use of templates and compile-time polymorphism
- Code should compile with no warnings (`-Werror` enabled)