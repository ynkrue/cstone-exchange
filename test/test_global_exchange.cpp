/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief CPU driver of the global sweep test
 */

#include "global_exchange.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    auto upsweepFunction = [](std::span<const TreeNodeIndex> ranges, const TreeNodeIndex* children, Q* q)
    { cstone::upsweep(ranges, children, q, CombineSum{}); };
    auto upsweepNodes = [](std::span<const TreeNodeIndex> nodes, const TreeNodeIndex* children, Q* q, bool)
    {
        for (TreeNodeIndex n : nodes)
        {
            q[n] = CombineSum{}(n, children[n], q);
        }
    };
    int      ranksPerNode = argc > 1 ? std::atoi(argv[1]) : 1;
    int      extraLevels  = argc > 2 ? std::atoi(argv[2]) : 0;
    int      steps        = argc > 3 ? std::atoi(argv[3]) : 10;
    MPI_Comm nodeComm;
    MPI_Comm_split(MPI_COMM_WORLD, rank / ranksPerNode, rank, &nodeComm);
    std::vector<Q> globQ;
    auto           allocGlobQ = [&](std::size_t numNodes)
    {
        globQ.assign(numNodes, Q{});
        return globQ.data();
    };
    auto loadNodePeer     = [](int, std::span<const TreeNodeIndex>) {};
    auto scatterNodePeers = [](std::span<const TreeNodeIndex>, const Q*) {};
    auto syncNode         = [] {};
    int  status =
        testGlobalExchange<std::vector>(execution::cpu, upsweepFunction, upsweepNodes, allocGlobQ, 0, loadNodePeer,
                                        scatterNodePeers, syncNode, rank, numRanks, nodeComm, steps, extraLevels);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return status;
}
