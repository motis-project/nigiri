#include <cista/containers/mmap_vec.h>
#include <iostream>

#include "cista/mmap.h"

#include "nigiri/loader/gtfs/shape.h"

int main() {
    cista::mmap_vec<char> mmap{cista::mmap{"../dev/shapes.txt", cista::mmap::protection::READ}};
    std::string_view data{mmap};
    std::cout << "Head: " << data.substr(0, 10) << std::endl;
    nigiri::loader::gtfs::ShapeMap::Paths paths{
        "shape-writer2-id.dat",
        "shape-writer2-data.dat",
        "shape-writer2-metadata.dat",
    };
    nigiri::loader::gtfs::ShapeMap shapes{data, paths};
    std::cout << "Added " << shapes.size() << " shapes." << std::endl;
    return 0;
}