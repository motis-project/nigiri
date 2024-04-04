#include <iostream>

int main(int argc, char* argv[]) {
  std::cout << "You have entered " << argc << " arguments:" << std::endl;

  // Using a while loop to iterate through arguments
  int i = 0;
  while (i < argc) {
    std::cout << "Argument " << i + 1 << ": " << argv[i] << std::endl;
    i++;
  }

  return 0;
}