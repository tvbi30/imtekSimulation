// kokkos
#include <Kokkos_Core.hpp>
// mpi
#include <mpi.h>
// custom
#include "milestone02.hpp"

int main(int argc, char** argv)
{
    int rank = 0, size = 1;
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0)
        m2::simulate();

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}