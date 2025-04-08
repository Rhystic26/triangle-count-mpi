#include <gms/common/types.h>
#include <cassert>
#include <mpi.h>

namespace GMS::TriangleCount::Par {

template<class SGraph>
size_t count_total(const SGraph &graph) {
  int rank, gsize, csize, gtotal;
  size_t n = graph.num_nodes();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &gsize);
  csize = n / gsize;

    size_t total = 0;
    for (NodeId u = rank*csize; u < (rank+1)*csize; ++u) {
        const auto &neigh_u = graph.out_neigh(u);
        for (NodeId v : neigh_u) {
            if (u < v) {
                total += neigh_u.intersect_count(graph.out_neigh(v));
            }
        }
    }

    MPI_Reduce(&total, &gtotal, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      assert(gtotal % 3 == 0);
    }
    
    return gtotal / 3;
}

}
