#include "lbmD2Q9.hpp"

// stl
#include <iostream>
#include <array>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

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
    LBMD2Q9Simulator::LBMD2Q9Simulator() noexcept
    {
        _fin = Kokkos3DArray("f", D2Q9::Q, D2Q9::nx, D2Q9::ny);
        _fout = Kokkos3DArray("fout", D2Q9::Q, D2Q9::nx, D2Q9::ny);
        _fstream = Kokkos3DArray("fstream", D2Q9::Q, D2Q9::nx, D2Q9::ny);
        _solids = Kokkos2DArray("solids", D2Q9::nx, D2Q9::ny);
        _rho = Kokkos2DArray("rho", D2Q9::nx, D2Q9::ny);
        _vx = Kokkos2DArray("vx", D2Q9::nx, D2Q9::ny);
        _vy = Kokkos2DArray("vy", D2Q9::nx, D2Q9::ny);

        init();
    }


    // TODO: erweitern, dass eigene start parameter mitgegeben werden können
    void LBMD2Q9Simulator::init() noexcept
    {
        Kokkos::parallel_for
        (
            "init",
            GridPolicy({0, 0}, {D2Q9::nx, D2Q9::ny}),
            KOKKOS_LAMBDA(int x, int y)
            {
                double rho = 1.0;
                const double vx = 0.0, vy = 0.03;
                const double uu = vx * vx + vy * vy;
                _solids(x, y) = ( x==0 || x== D2Q9::nx - 1 || y == 0 || y == D2Q9::ny- 1);

                if ((x-D2Q9::nx/2)*(x-D2Q9::nx/2) + (y-D2Q9::ny/2)*(y-D2Q9::ny/2) < 25)
                    rho = 1.2;

                #pragma unroll
                for(int i = 0; i < D2Q9::Q; ++i)
                {
                    const double cu = D2Q9::cx[i] * vx + D2Q9::cy[i] * vy;
                    _fin(i, x, y) = D2Q9::w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
                }
            }
        );
    }

    void LBMD2Q9Simulator::calculateMacros(const int x, const int y) noexcept
    {
        // reset the values
        _rho(x, y) = 0;
        _vx(x, y) = 0;
        _vy(x, y) = 0;

        #pragma unroll
        for(int i = 0; i < D2Q9::Q; ++i)
        {
            // value of current direction:
            double val = _fstream(i, x, y);
            // calc velocity
            _rho(x, y) += val;
            // calc density
            _vx(x, y) += val * D2Q9::cx[i];
            _vy(x, y) += val * D2Q9::cy[i];
        }

        // dont divide by zero
        if(_rho(x, y)  <= 1e-12) [[unlikely]]
        {
            _vx(x, y) = _vy(x, y) = 0;
            return;
        }

        _vx(x, y) /= _rho(x, y);
        _vy(x, y) /= _rho(x, y);
    }

    void LBMD2Q9Simulator::calculateEquilibrium(const int x, const int y, Kokkos::Array<double, D2Q9::Q>& feq) noexcept
    {
        double cu = 0;
        double uu = _vx(x, y) * _vx(x, y) + _vy(x, y) * _vy(x, y);
        
        #pragma unroll
        for(int i = 0; i < D2Q9::Q; ++i)
        {
            cu = D2Q9::cx[i] * _vx(x, y) + D2Q9::cy[i] * _vy(x, y);
            feq[i] = D2Q9::w[i] * _rho(x, y) * (1.0 + 3 * cu + 4.5 * cu * cu - 1.5 * uu);
        }
    }

    void LBMD2Q9Simulator::lbmStepCollide(const int x, const int y) noexcept
    {
        calculateMacros(x, y);
        Kokkos::Array<double, D2Q9::Q> feq;
        calculateEquilibrium(x, y, feq);

        #pragma unroll
        for (int i = 0; i < 9; ++i)
            _fout(i, x, y) = _fstream(i, x, y) - D2Q9::omega * (_fstream(i, x, y) - feq[i]);
    }

    void LBMD2Q9Simulator::lbmStepStream(const int x, const int y) noexcept
    {
        #pragma unroll
        for(int i = 0; i < D2Q9::Q; ++i)
        {
            int xs = x - D2Q9::cx[i];
            int ys = y - D2Q9::cy[i];
        
            // periodic boundaries
            xs = (xs + D2Q9::nx) % D2Q9::nx;
            ys = (ys + D2Q9::ny) % D2Q9::ny;

            _fstream(i, x, y) = _fin(i, xs, ys);
        }
    }

    void LBMD2Q9Simulator::writeFile(Kokkos3DArray data, const int iteration) noexcept
    {
        auto rhoHost = Kokkos::create_mirror_view(_rho);
        Kokkos::deep_copy(rhoHost, _rho);

        auto vxHost = Kokkos::create_mirror_view(_vx);
        Kokkos::deep_copy(vxHost, _vx);

        auto vyHost = Kokkos::create_mirror_view(_vy);
        Kokkos::deep_copy(vyHost, _vy);

        std::string name = "/home/tobi/imtekSimulation/results/run1/lbmStep_" + std::to_string(iteration) + ".vtk";
        std::ofstream out(name);

        out << "# vtk DataFile Version 3.0\n";
        out << "LBM output\n";
        out << "ASCII\n";
        out << "DATASET STRUCTURED_POINTS\n";
        out << "DIMENSIONS " << D2Q9::nx << " " << D2Q9::ny << " 1\n";
        out << "ORIGIN 0 0 0\n";
        out << "SPACING 1 1 1\n";

        out << "POINT_DATA " << D2Q9::nx * D2Q9::ny << "\n";

        // density
        out << "SCALARS rho double 1\n";
        out << "LOOKUP_TABLE default\n";
        for (int x = 0; x < D2Q9::nx; x++)
        {
            for(int y = 0; y < D2Q9::ny; y++)
                out << rhoHost(x, y) << "\n";
        }

        // velocity x
        out << "VECTORS velocity double\n";
        for (int x = 0; x < D2Q9::nx; x++)
        {
            for(int y = 0; y < D2Q9::ny; y++)
                out << vxHost(x, y) << " " << vyHost(x, y) << " 0\n";
        }
    }

    void LBMD2Q9Simulator::simulate(int iterations /*= 200*/) noexcept
    {
        std::cout << "> starting simulation\n";
        std::cout << ">> settings:\n\titerations: " << iterations << "\n\tdimensions: " << D2Q9::nx << "x" << D2Q9::ny<<"\n";
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        const int writeOut = 5;
        for(int i = 0; i < iterations; ++i)
        {
            // Kokkos::parallel_for
            // (
            //     "LBM",
            //     GridPolicy({1, 1}, {D2Q9::nx - 1, D2Q9::ny - 1}),
            //     KOKKOS_LAMBDA(const int x, const int y)
            //     {
            //         lbmStepCombined(x, y);
            //     }
            // );
            Kokkos::parallel_for
            (
                "stream",
                GridPolicy({0, 0}, {D2Q9::nx, D2Q9::ny}),
                KOKKOS_LAMBDA(const int x, const int y)
                {
                    lbmStepStream(x, y);
                }
            );

            Kokkos::parallel_for
            (
                "collision",
                GridPolicy({0, 0}, {D2Q9::nx, D2Q9::ny}),
                KOKKOS_LAMBDA(const int x, const int y)
                {
                    lbmStepCollide(x, y);
                }
            );

            std::swap(_fin, _fout);
            Kokkos::fence();

            // write files
            if(i % writeOut == 0)
            {
                Kokkos3DArray host = Kokkos::create_mirror_view(_fin);
                Kokkos::deep_copy(host, _fin);
                writeFile(host, i);
            }
        }
        std::chrono::steady_clock::time_point finished = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finished - start).count();
        std::cout << "> simulation finished within " << duration << "ms\n";
    }
}