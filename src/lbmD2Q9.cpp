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
        static constexpr double gravY = -9.81;
        static constexpr int cx[Q] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
        static constexpr int cy[Q] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
        static constexpr double w[Q] = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0 };
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
                xn = x - D2Q9::cx[i];
                yn = y - D2Q9::cy[i];

                _fOutput(i, x, y) = _fInput(i, xn, yn);
            }
        }
    };

    struct CollisionFunctor
    {
        using GridBuffer = Kokkos::View<double***, Kokkos::LayoutRight>;
        GridBuffer _fInput;
        GridBuffer _fOutput;

        CollisionFunctor(GridBuffer _fin, GridBuffer _fout) : _fInput(_fin), _fOutput(_fout) {}

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
                uy += val * D2Q9::cy[i]];
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
        void calcEquilibrium(double (&feq)[D2Q9::Q], const double rho, const double ux, const double uy) const
        {
            double cu = 0;
            double uu = ux * ux + uy * uy;
            
            #pragma unroll(D2Q9::Q)
            for(int i = 0; i < D2Q9::Q; ++i)
            {
                cu = D2Q9::cx[i] * ux + D2Q9::cy[i] * uy;
                feq[i] = D2Q9::w[i] * rho * (1.0 + 3 * cu + 4.5 * cu * cu - 1.5 * uu);
            }
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const int x, const int y) const
        {
            double rho = 0, ux = 0, uy = 0;
            calcCurrentCellState(x, y, rho, ux, uy);

            double feq[D2Q9::Q] = {0};
            calcEquilibrium(feq, rho, ux, uy);

            // bgk collision
            #pragma unroll(D2Q9::Q)
            for(int i = 0; i < D2Q9::Q; ++i)
                _fOutput(i, x, y) = (1 - D2Q9::omega) * _fInput(i, x, y) + D2Q9::omega * feq[i];
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
                const double ux = 0.0, uy = 1.0;
                const double uu = ux * ux + uy * uy;

                #pragma unroll(D2Q9::Q)
                for(int i = 0; i < D2Q9::Q; ++i)
                {
                    const double cu = D2Q9::cx[i] * ux + D2Q9::cy[i] * uy;
                    f(i, x, y) = D2Q9::w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu * 3.0 * uu);
                }
            }
        );

        // collison
        Kokkos::parallel_for
        (
            "collision",
            GridPolicy({0, 0}, {D2Q9::nx, D2Q9::ny}),
            CollisionFunctor(f, fn)
        );

        // streaming
        Kokkos::parallel_for
        (
            "streaming",
            GridPolicy({0, 0}, {D2Q9::nx, D2Q9::ny}),
            StreamingFunctor(f, fn)
        );

        std::swap(f, fn);
    }
}