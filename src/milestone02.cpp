#include "milestone02.hpp"

// stl
#include <iostream>

// kokkos
#include <Kokkos_View.hpp>

namespace m2
{
    // velocities
    constexpr int cx[Q] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
    constexpr int cy[Q] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };

    KOKKOS_INLINE_FUNCTION
    int index(int x, int y)
    {
        return y * width + x;
    }

    double density()
    {

    }

    double velocity()
    {

    }

    void simulate()
    {
        Kokkos::View<double*> f[Q];

        // init f
        for(int i = 0; i < Q; ++i)
            f[i] = Kokkos::View<double*>("f" + std::to_string(i), N);
        std::cout << "ey jo ich simulate\n";
    }
}