#include "milestone02.hpp"

// stl
#include <iostream>
#include <array>

// kokkos
#include <Kokkos_Core.hpp>

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

namespace m2
{
    struct D2Q9
    {
        // parameters
        static constexpr short Q = 9;
        static constexpr int nx = 15;
        static constexpr int ny = 10;
        static constexpr int N = nx * ny;
        static constexpr double omega = 1.0;

        // access functions to the velocites
        KOKKOS_INLINE_FUNCTION
        static int cx(int index)
        {
            constexpr int vx[Q] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
            return vx[index];
        }

        KOKKOS_INLINE_FUNCTION
        static int cy(int index)
        {
            constexpr int vy[Q] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
            return vy[index];
        }

        KOKKOS_INLINE_FUNCTION
        static double weight(int index)
        {
            constexpr double w[Q] = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0 };
            return w[index];
        }
    };

    struct StreamingFunctor
    {
        using GridBuffer = Kokkos::View<double***, Kokkos::LayoutRight>;
        GridBuffer f;
        GridBuffer fn;

        StreamingFunctor(GridBuffer _f, GridBuffer _fn) : f(_f), fn(_fn) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const int x, const int y) const 
        {
            int xn = 0;
            int yn = 0;
            #pragma unroll
            for(int i = 0; i < D2Q9::Q; ++i)
            {
                xn = x - D2Q9::cx(i);
                yn = y - D2Q9::cy(i);

                fn(i, x, y) = f(i, xn, yn);
            }
        }
    };

    struct CollisionFunctor
    {
        using GridBuffer = Kokkos::View<double***, Kokkos::LayoutRight>;
        GridBuffer f;

        CollisionFunctor(GridBuffer _f) : f(_f) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const int x, const int y) const
        {
            // density of given cell
            // velocity of given cell
            double ux = 0, uy = 0;
            double rho = 0;
            
            #pragma unroll(Q)
            for(int i = 0; i < D2Q9::Q; ++i)
            {
                // value of current direction:
                double val = f(i, x, y);
                // calc velocity
                rho += val;
                // calc density
                ux += val * D2Q9::cx(i);
                uy += val * D2Q9::cy(i);
            }

            // dont divide by zero
            if(rho <= 1e-12) return;

            ux /= rho;
            uy /= rho;

            double u2 = ux*ux + uy*uy;

            // bgk collision
            #pragma unroll(Q)
            for(int i = 0; i < D2Q9::Q; ++i)
            {
                int _cx = D2Q9::cx(i);
                int _cy = D2Q9::cy(i);

                double cu = _cx*ux + _cy*uy;
                
                // equi dist
                double feq = D2Q9::weight(i) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);

                // relaxation
                f(i, x, y) = f(i, x, y) - D2Q9::omega * (f(i, x, y) - feq);
            }
        }

    };

    void simulate()
    {
        using ExecSpace = Kokkos::DefaultExecutionSpace;

        // indicies are: i,x,y -> direction, posX, posY
        // soa pattern is better for gpu performance
        Kokkos::View<double***, Kokkos::LayoutRight> f("f", D2Q9::Q, D2Q9::nx, D2Q9::ny);
        Kokkos::View<double***, Kokkos::LayoutRight> fn("fn", D2Q9::Q, D2Q9::nx, D2Q9::ny);

        // policy to traverse a 2d grid
        using gridPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

        // collison
        Kokkos::parallel_for
        (
            "collision",
            gridPolicy({0, 0}, {D2Q9::nx, D2Q9::ny}),
            CollisionFunctor(f)
        );

        // streaming
        Kokkos::parallel_for
        (
            "streaming",
            gridPolicy({0, 0}, {D2Q9::nx, D2Q9::ny}),
            StreamingFunctor(f, fn)
        );

        std::swap(f, fn);
    }
}