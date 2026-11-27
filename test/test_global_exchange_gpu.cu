/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GPU driver of the global sweep test
 */

#include <cstdint>
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/stream_holder.cuh"
#include "global_exchange.hpp"

namespace
{
constexpr int numComponents = Q{}.size();

//! @brief block b sets node first + b from its children, thread c sums component c, like the M2M kernel of exafmm
__global__ void sumParents(TreeNodeIndex first, const TreeNodeIndex* children, Q* q)
{
    TreeNodeIndex i = first + blockIdx.x;
    if (children[i] == 0) { return; }
    const double* qd  = reinterpret_cast<const double*>(q);
    double        sum = 0;
    for (int k = 0; k < 8; ++k)
    {
        sum += qd[std::size_t(children[i] + k) * numComponents + threadIdx.x];
    }
    reinterpret_cast<double*>(q)[std::size_t(i) * numComponents + threadIdx.x] = sum;
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    gpuAwareMpiEnabled = true;
    int status;
    {
        StreamHolder stream;
        auto         exec            = stream.exec();
        auto         upsweepFunction = [&](std::span<const TreeNodeIndex> ranges, const TreeNodeIndex* children, Q* q)
        {
            for (int l = int(ranges.size()) - 2; l >= 0; --l)
            {
                int n = ranges[l + 1] - ranges[l];
                if (n > 0)
                {
                    sumParents<<<n, numComponents, 0, exec>>>(ranges[l], children, q);
                    checkGpuErrors(cudaGetLastError());
                }
            }
        };
        auto upsweepNodes = [&](std::span<const TreeNodeIndex> nodes, const TreeNodeIndex* children, Q* q)
        {
            for (TreeNodeIndex n : nodes)
            {
                sumParents<<<1, numComponents, 0, exec>>>(n, children, q);
                checkGpuErrors(cudaGetLastError());
            }
        };
        int steps = argc > 1 ? std::atoi(argv[1]) : 10;
        status    = testGlobalExchange<DeviceVector>(exec, upsweepFunction, upsweepNodes, rank, numRanks, steps);
    }
    MPI_Finalize();
    return status;
}
