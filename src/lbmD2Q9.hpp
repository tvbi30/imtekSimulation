// #pragma once

#include <Kokkos_Core.hpp>

#ifndef LBMD2Q9
#define LBMD2Q9

namespace lbm
{
    void simulate(int iterations = 100);
    
    class LBMD2Q9Simulator
    {
        using Kokkos3DArray = Kokkos::View<double***, Kokkos::LayoutRight>;
        using Kokkos2DArray = Kokkos::View<double**, Kokkos::LayoutRight>;
        using KokkosArray = Kokkos::View<double*, Kokkos::LayoutRight>;
        using GridPolicy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    private:
        struct D2Q9
        {
            // parameters
            static constexpr short Q = 9;
            static constexpr int nx = 200;
            static constexpr int ny = 150;
            static constexpr int N = nx * ny;
            static constexpr double omega = 1.0;
            static constexpr double gravX = 0.0;
            static constexpr double gravY = -1e-6;
            static constexpr double u = 0.01;
            static constexpr Kokkos::Array<int, Q> cx = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
            static constexpr Kokkos::Array<int, Q> cy = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
            static constexpr Kokkos::Array<double, Q> w = { 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0 };
            static constexpr Kokkos::Array<int, Q> o = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
        };

        // in buffer
        Kokkos3DArray _fin;
        // stream buffer
        Kokkos3DArray _fstream;
        // out buffer
        Kokkos3DArray _fout;
        // solid elements
        Kokkos2DArray _solids;
        // density
        Kokkos2DArray _rho;
        // velocities x
        Kokkos2DArray _vx;
        // velocites y
        Kokkos2DArray _vy;

        KOKKOS_INLINE_FUNCTION
        void calculateMacros(const int x, const int y) noexcept;

        KOKKOS_INLINE_FUNCTION
        void calculateEquilibrium(const int x, const int y, Kokkos::Array<double, D2Q9::Q>& feq) noexcept;

        void lbmStepCombined(const int x, const int y) noexcept;
        void lbmStepCollide(const int x, const int y) noexcept;
        void lbmStepStream(const int x, const int y) noexcept;
        void init() noexcept;

        void writeFile(Kokkos3DArray data, const int iteration) noexcept;
    public:
        LBMD2Q9Simulator() noexcept;

        void simulate(int iterations = 400) noexcept;
        const Kokkos3DArray getDistribution() const { return _fin; }
    };
}
#endif