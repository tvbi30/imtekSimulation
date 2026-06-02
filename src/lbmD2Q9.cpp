#include "lbmD2Q9.hpp"

// stl
#include <iostream>
#include <array>

// kokkos
#include <Kokkos_Core.hpp>

// just a visualization for my brain to better understand the memory layout in the view
    // struct Laticce
    // {
    //     double f0[nx * ny];
    //     double f1[nx * ny];
    //     double f2[nx * ny];
    //     double f3[nx * ny];
    //     double f4[nx * ny];
    //     double f5[nx * ny];
    //     double f6[nx * ny];
    //     double f7[nx * ny];
    //     double f8[nx * ny];
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
        static constexpr double gravX = 0.0;
        static constexpr double gravY = -1e-6;
        static constexpr int cx[Q] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
        static constexpr int cy[Q] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
        static constexpr double w[Q] = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0 };
    };

    struct LBMD2Q9Functor
    {
        using GridBuffer = Kokkos::View<double***, Kokkos::LayoutRight>;
        using KernelArray = Kokkos::View<double*, Kokkos::LayoutRight>;
        GridBuffer _fInput;
        GridBuffer _fOutput;
        KernelArray feq = KernelArray("feq", D2Q9::Q);

        LBMD2Q9Functor(GridBuffer _fin, GridBuffer _fout) : _fInput(_fin), _fOutput(_fout) {}

        KOKKOS_INLINE_FUNCTION
        void calcCurrentCellState(const int x, const int y, double& rho, double& ux, double& uy) const
        {
            #pragma unroll(D2Q9::Q)
            for(int i = 0; i < D2Q9::Q; ++i)
            {
                // value of current direction:
                double val = _fInput(i, x, y);
                // calc velocity
                rho += val;
                // calc density
                ux += val * D2Q9::cx[i];
                uy += val * D2Q9::cy[i];
            }

            // dont divide by zero
            if(rho <= 1e-12) [[unlikely]]
            {
                ux = uy = 0;
                return;
            }

            ux /= rho;
            uy /= rho;
        }

        KOKKOS_INLINE_FUNCTION
        void calcEquilibrium(const double rho, const double ux, const double uy) const
        {
            double cu = 0;
            double uu = ux * ux + uy * uy;
            
            #pragma unroll(D2Q9::Q)
            for(int i = 0; i < D2Q9::Q; ++i)
            {
                cu = D2Q9::cx[i] * ux + D2Q9::cy[i] * uy;
                feq(i) = D2Q9::w[i] * rho * (1.0 + 3 * cu + 4.5 * cu * cu - 1.5 * uu);
            }
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const int x, const int y) const
        {
            double rho = 0, ux = 0, uy = 0;
            calcCurrentCellState(x, y, rho, ux, uy);
            calcEquilibrium(rho, ux, uy);

            // bgk collision and streaming
            double value = 0;
            int nx = 0, ny = 0;
            #pragma unroll(D2Q9::Q)
            for(int i = 0; i < D2Q9::Q; ++i)
            {
                // COLLISION
                value = (1 - D2Q9::omega) * _fInput(i, x, y) + D2Q9::omega * feq[i];

                // STREAMING
                // get neighbor coordinates
                nx = (x + D2Q9::cx[i] + D2Q9::nx) % D2Q9::nx;
                ny = (y + D2Q9::cy[i] + D2Q9::ny) % D2Q9::ny;
                
                // we can push to the neighbor without a race condition because each cell get read from 
                // and written to only once per step. we can safe a read and a write doing this
                // push to new neighbor
                _fOutput(i, nx, ny) = value;
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
        using GridPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

        Kokkos::parallel_for
        (
            "init",
            GridPolicy({0, 0}, {D2Q9::nx, D2Q9::ny}),
            KOKKOS_LAMBDA(int x, int y)
            {
                const double rho = 1.0;
                const double ux = 0.0, uy = 0.01;
                const double uu = ux * ux + uy * uy;

                #pragma unroll(D2Q9::Q)
                for(int i = 0; i < D2Q9::Q; ++i)
                {
                    const double cu = D2Q9::cx[i] * ux + D2Q9::cy[i] * uy;
                    f(i, x, y) = D2Q9::w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
                }
            }
        );

        // collison
        Kokkos::parallel_for
        (
            "lbm",
            GridPolicy({0, 0}, {D2Q9::nx, D2Q9::ny}),
            LBMD2Q9Functor(f, fn)
        );

        std::swap(f, fn);
    }
}