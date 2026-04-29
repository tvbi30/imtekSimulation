#include "milestone02.hpp"

// stl
#include <iostream>
#include <array>

// kokkos
#include <Kokkos_View.hpp>

namespace m2
{
    // just a visualization for my brain to better understand the memory layout in the view
    // struct Laticce
    // {
    //     float f0[nx * ny];
    //     float f1[nx * ny];
    //     float f2[nx * ny];
    //     float f3[nx * ny];
    //     float f4[nx * ny];
    //     float f5[nx * ny];
    //     float f6[nx * ny];
    //     float f7[nx * ny];
    //     float f8[nx * ny];
    // };

    // velocities
    constexpr int cx[Q] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
    constexpr int cy[Q] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };

    // policy to traverse a 2d grid
    const Kokkos::MDRangePolicy<Kokkos::Rank<2>> gridTraversePolicy
    (
        {0, 0},
        {nx, ny}
    );
    
    KOKKOS_INLINE_FUNCTION
    int index(int x, int y)
    {
        return (y * nx) + x;
    }

    void simulate()
    {
        // indicies are: i,x,y -> direction, posX, posY
        // soa pattern is better for gpu performance
        Kokkos::View<double***, Kokkos::LayoutRight> f("f", Q, nx, ny);
        Kokkos::View<double***, Kokkos::LayoutRight> fn("fn", Q, nx, ny);
        
        // collison
        Kokkos::parallel_for
        (
            gridTraversePolicy,
            KOKKOS_LAMBDA (int x, int y)
            {
                // density of given cell
                double rho = 0;
                // velocity of given cell
                double ux = 0, uy = 0;
                
                // #pragma unroll(Q)
                for(int i = 0; i < Q; ++i)
                {
                    // value of current direction:
                    double val = f(x, y, i);
                    // calc velocity
                    rho += val;
                    // calc density
                    ux += val * cx[i];
                    uy += val * cy[i];
                }
                ux /= rho;
                uy /= rho;
            }
        );

        // streaming
        for(int x = 0; x < nx; ++x)
        {
            for(int y = 0; y < ny; y++)
            {
                int xn = 0;
                int yn = 0;
                #pragma unroll
                for(int i = 0; i < Q; ++i)
                {
                    xn = x - cx[i];
                    yn = y - cy[i];

                    fn(i, x, y) = f(i, xn, yn);
                }
            }
        }
    }
}