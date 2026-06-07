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
        _solids = Kokkos2DArray("solids", D2Q9::nx, D2Q9::ny);
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
                const double rho = 1.0;
                const double vx = 0.0, vy = 0.01;
                const double uu = vx * vx + vy * vy;
                _solids(x, y) = 0;

                #pragma unroll
                for(int i = 0; i < D2Q9::Q; ++i)
                {
                    const double cu = D2Q9::cx[i] * vx + D2Q9::cy[i] * vy;
                    _fin(i, x, y) = D2Q9::w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
                }
            }
        );
    }

    void LBMD2Q9Simulator::calculateMacros(const int x, const int y, double& rho, double& vx, double& vy) noexcept
    {
        // reset the values
        rho = 0;
        vx = 0;
        vy = 0;

        #pragma unroll
        for(int i = 0; i < D2Q9::Q; ++i)
        {
            // value of current direction:
            double val = _fin(i, x, y);
            // calc velocity
            rho += val;
            // calc density
            vx += val * D2Q9::cx[i];
            vy += val * D2Q9::cy[i];
        }

        // dont divide by zero
        if(rho  <= 1e-12) [[unlikely]]
        {
            vx = vy = 0;
            return;
        }

        vx /= rho;
        vy /= rho;
    }

    void LBMD2Q9Simulator::calculateEquilibrium(const int x, const int y, Kokkos::Array<double, D2Q9::Q>& feq, const double rho, const double vx, const double vy) noexcept
    {
        double cu = 0;
        double uu = vx * vx + vy * vy;
        
        #pragma unroll
        for(int i = 0; i < D2Q9::Q; ++i)
        {
            cu = D2Q9::cx[i] * vx + D2Q9::cy[i] * vy;
            feq[i] = D2Q9::w[i] * rho * (1.0 + 3 * cu + 4.5 * cu * cu - 1.5 * uu);
        }
    }

    void LBMD2Q9Simulator::lbmStep(const int x, const int y) noexcept
    {
        int idx = x;
        int idy = y;

        // bounce on solid nodes
        if(_solids(x, y))
        {
            #pragma unroll
            for(int i = 0; i < D2Q9::Q; ++i)
                _fout(i, x, y) = _fin(D2Q9::o[i], x, y);
            return;
        }

        // pull streaming
        double f0 = _fin(0, x, y);
        double f1 = _fin(1, x - 1, y);
        double f2 = _fin(2, x, y - 1);
        double f3 = _fin(3, x + 1, y);
        double f4 = _fin(4, x, y + 1);
        double f5 = _fin(5, x - 1, y - 1);
        double f6 = _fin(6, x + 1, y - 1);
        double f7 = _fin(7, x + 1, y + 1);
        double f8 = _fin(8, x - 1, y + 1);

        // calculate makros
        // density
        double rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;

        if(rho <= 0.09f) { return; }

        // velocites in each direction
        double vx = (f1 - f3 + f5 - f6 + f7 - f8) / rho;
        double vy = (f2 - f4 + f5 + f6 - f7 - f8) / rho;

        // zou/he inlet
        if(x == 1)
        {
            double f0b = f0;
            double f2b = f2;
            double f4b = f4;
            double f3b = f3;
            double f6b = f6;
            double f7b = f7;

            double rho_in = (f0b + f2b + f4b + 2*(f3b + f6b + f7b)) / (1.0 - D2Q9::u);

            vx = D2Q9::u;
            vy = 0.0;
            rho = rho_in;

            f1 = f3b + (2.0 / 3.0) * rho * D2Q9::u;
            f5 = f7b + 0.5*(f4b - f2b) + (1.0 / 6.0)* rho * D2Q9::u;
            f8 = f6b + 0.5*(f2b - f4b) + (1.0 / 6.0)* rho * D2Q9::u;
        }

        // velocity squared
        double v_sqrd = vx * vx + vy * vy;

        // calc equilibrium
        Kokkos::Array<double, D2Q9::Q> feq;
        calculateEquilibrium(x, y, feq, rho, vx, vy);

        // collide + write back
        _fout(0, x, y) = f0 - D2Q9::omega*(f0 - feq[0]);
        _fout(1, x, y) = f1 - D2Q9::omega*(f1 - feq[1]);
        _fout(2, x, y) = f2 - D2Q9::omega*(f2 - feq[2]);
        _fout(3, x, y) = f3 - D2Q9::omega*(f3 - feq[3]);
        _fout(4, x, y) = f4 - D2Q9::omega*(f4 - feq[4]);
        _fout(5, x, y) = f5 - D2Q9::omega*(f5 - feq[5]);
        _fout(6, x, y) = f6 - D2Q9::omega*(f6 - feq[6]);
        _fout(7, x, y) = f7 - D2Q9::omega*(f7 - feq[7]);
        _fout(8, x, y) = f8 - D2Q9::omega*(f8 - feq[8]);
    }

    void LBMD2Q9Simulator::writeFile(Kokkos3DArray data, const int iteration) noexcept
    {
        std::vector<double> rho(D2Q9::nx * D2Q9::ny);
        std::vector<double> ux(D2Q9::nx * D2Q9::ny);
        std::vector<double> uy(D2Q9::nx * D2Q9::ny);

        for (int x = 0; x < D2Q9::nx; x++) 
        {
            for (int y = 0; y < D2Q9::ny; y++) 
            {
                double r = 0, vx = 0, vy = 0;
            
                for (int i = 0; i < 9; i++) 
                {
                  double fi = data(i,x,y);
                  r += fi;
                  vx += D2Q9::cx[i] * fi;
                  vy += D2Q9::cy[i] * fi;
                }
            
                int idx = x + y * D2Q9::nx;
                rho[idx] = r;
                if(r > 1e-12)
                {
                    ux[idx] = vx / r;
                    uy[idx] = vy / r;
                }
                else
                {
                    ux[idx] = 0.0;
                    uy[idx] = 0.0;
                }
            }
        }

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
        for (auto v : rho) out << v << "\n";

        // velocity x
        out << "VECTORS velocity double\n";
        for (int i = 0; i < D2Q9::nx * D2Q9::ny; i++) 
          out << ux[i] << " " << uy[i] << " 0\n";
    }

    void LBMD2Q9Simulator::simulate(int iterations /*= 200*/) noexcept
    {
        std::cout << "> starting simulation\n";
        std::cout << ">> settings:\n\titerations: " << iterations << "\n\tdimensions: " << D2Q9::nx << "x" << D2Q9::ny<<"\n";
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        const int writeOut = 20;
        for(int i = 0; i < iterations; ++i)
        {
            Kokkos::parallel_for
            (
                "LBM",
                GridPolicy({1, 1}, {D2Q9::nx - 1, D2Q9::ny - 1}),
                KOKKOS_LAMBDA(const int x, const int y)
                {
                    lbmStep(x, y);
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