#include "lbmD2Q9.hpp"

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

namespace lbm
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
        GridBuffer _fInput;
        GridBuffer _fOutput;

        StreamingFunctor(GridBuffer _fin, GridBuffer _fout) : _fInput(_fin), _fOutput(_fout) {}

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

                _fOutput(i, x, y) = _fInput(i, xn, yn);
            }
        }
    };

    struct CurrentCellState
    {
        float vx = 0;
        float vy = 0;
        float density = 0;
    };

    struct CollisionFunctor
    {
        using GridBuffer = Kokkos::View<double***, Kokkos::LayoutRight>;
        GridBuffer _fInput;
        GridBuffer _fOutput;

        CollisionFunctor(GridBuffer _fin, GridBuffer _fout) : _fInput(_fin), _fOutput(_fout) {}

        KOKKOS_INLINE_FUNCTION
        CurrentCellState calcCurrentCellState(const int x, const int y) const
        {
            CurrentCellState state;

            #pragma unroll(D2Q9::Q)
            for(int i = 0; i < D2Q9::Q; ++i)
            {
                // value of current direction:
                double val = _fInput(i, x, y);
                // calc velocity
                state.density += val;
                // calc density
                state.vx += val * D2Q9::cx(i);
                state.vy += val * D2Q9::cy(i);
            }

            return state;
        }

        KOKKOS_INLINE_FUNCTION
        float calcEquilibrium()
        {
            #pragma unroll(D2Q9::Q)
            for(int i = 0; i < D2Q9::Q; ++i)
            {
                
            }
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const int x, const int y) const
        {
            // calculate density and velocties
            CurrentCellState curr = calcCurrentCellState(x, y);

            // dont divide by zero
            if(curr.density <= 1e-12) return;

            // bgk collision
            #pragma unroll(D2Q9::Q)
            for(int i = 0; i < D2Q9::Q; ++i)
            {
                // int _cx = D2Q9::cx(i);
                // int _cy = D2Q9::cy(i);

                // double cu = _cx*ux + _cy*uy;
                
                // // equi dist
                // double feq = D2Q9::weight(i) * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);

                // // relaxation
                // _fOutput(i, x, y) = _fInput(i, x, y) - D2Q9::omega * (_fInput(i, x, y) - feq);

                _fOutput(i, x, y) = (1 - D2Q9::omega) * _fInput(i, x, y) + D2Q9::omega * equi;
                // gravity
                _fOutput(i, x, y) = curr.density * D2Q9::weight(i) * vec2dot(vel, grav);
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