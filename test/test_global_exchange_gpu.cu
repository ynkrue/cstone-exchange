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
#include <nvshmem.h>
#include <nvshmemx.h>
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/stream_holder.cuh"
#include "global_exchange.hpp"

namespace
{
// one block per node and one thread per component of Q
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

//! @brief node nodes[i] of globQ = node nodes[i] of peerQ
__global__ void loadNodePeerKernel(const TreeNodeIndex* nodes, const Q* peerQ, Q* globQ)
{
    TreeNodeIndex n                                   = nodes[blockIdx.x];
    reinterpret_cast<double*>(globQ + n)[threadIdx.x] = reinterpret_cast<const double*>(peerQ + n)[threadIdx.x];
}

//! @brief node nodes[i] of every peerQ[m] = values[i]
__global__ void scatterNodePeersKernel(const TreeNodeIndex* nodes, const Q* values, Q* const* peerQ, int numPeers)
{
    double v = reinterpret_cast<const double*>(values + blockIdx.x)[threadIdx.x];
    for (int m = 0; m < numPeers; ++m)
    {
        reinterpret_cast<double*>(peerQ[m] + nodes[blockIdx.x])[threadIdx.x] = v;
    }
}

// nodes of one parent launch, passed by value
constexpr int maxListNodes = 256;
struct NodeList
{
    int           count;
    TreeNodeIndex nodes[maxListNodes];
};

//! @brief one block computes the parents of the list in order, thread c sums component c of the children
__global__ void sumParentsList(NodeList list, const TreeNodeIndex* children, Q* q, uint64_t* signal,
                               const int* peerPes, int myPos, int numPeers, uint64_t count)
{
    // count is 0 without sync, numComponents threads cover all node peers
    if (count != 0)
    {
        int m = threadIdx.x;
        if (m < numPeers && m != myPos)
        {
            __threadfence_system();
            nvshmemx_signal_op(signal + myPos, count, NVSHMEM_SIGNAL_SET, peerPes[m]);
            nvshmem_signal_wait_until(signal + m, NVSHMEM_CMP_GE, count);
        }
        __threadfence();
        __syncthreads();
    }
    for (int i = 0; i < list.count; ++i)
    {
        TreeNodeIndex n = list.nodes[i];
        if (children[n])
        {
            double* qd  = reinterpret_cast<double*>(q);
            double  sum = qd[std::size_t(children[n]) * numComponents + threadIdx.x];
            for (int k = 1; k < 8; ++k)
            {
                sum += qd[std::size_t(children[n] + k) * numComponents + threadIdx.x];
            }
            qd[std::size_t(n) * numComponents + threadIdx.x] = sum;
        }
    }
}

/*! @brief thread m signals node peer m that count syncs are done and waits until node peer m has done as many
 *
 * One signal slot per node peer, so that the next signal of a fast peer cannot stand in for a slow peer. The system
 * fence completes the stores of earlier kernels into the peers before the signal. nvshmem_fence would also fence the
 * remote proxy over libfabric, which aborts at P 128.
 */
__global__ void syncNodeKernel(uint64_t* signal, const int* peerPes, int myPos, uint64_t count)
{
    int m = threadIdx.x;
    if (m == myPos) { return; }
    __threadfence_system();
    nvshmemx_signal_op(signal + myPos, count, NVSHMEM_SIGNAL_SET, peerPes[m]);
    nvshmem_signal_wait_until(signal + m, NVSHMEM_CMP_GE, count);
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    gpuAwareMpiEnabled = true;

    // the node peers are the ranks of nodeComm, PE numbers equal the MPI world ranks, which are the SFC ranks
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int myPos, numNodePeers;
    MPI_Comm_rank(nodeComm, &myPos);
    MPI_Comm_size(nodeComm, &numNodePeers);
    std::vector<int> nodePeers(numNodePeers);
    MPI_Allgather(&rank, 1, MPI_INT, nodePeers.data(), 1, MPI_INT, nodeComm);

    nvshmemx_init_attr_t attr  = NVSHMEMX_INIT_ATTR_INITIALIZER;
    MPI_Comm             world = MPI_COMM_WORLD;
    nvshmemx_set_attr_mpi_comm_args(&world, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // symmetric allocation needs the same size on every PE, so every node must have the same number of ranks
    uint64_t* nodeSignal = static_cast<uint64_t*>(nvshmem_calloc(numNodePeers, sizeof(uint64_t)));
    uint64_t  numSyncs   = 0;
    Q*        globQ      = nullptr;
    Q**       nodePeerQ  = nullptr; // device array of the globQ pointers of the node peers
    int       status;
    {
        StreamHolder      stream;
        auto              exec = stream.exec();
        DeviceVector<int> nodePeerPes(nodePeers);
        auto upsweepFunction = [&](std::span<const TreeNodeIndex> ranges, const TreeNodeIndex* children, Q* q)
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
        auto syncNode = [&]
        {
            syncNodeKernel<<<1, numNodePeers, 0, exec>>>(nodeSignal, nodePeerPes.data(), myPos, ++numSyncs);
            checkGpuErrors(cudaGetLastError());
        };
        // the sync of a phase runs inside the first parent kernel
        auto upsweepNodes = [&](std::span<const TreeNodeIndex> nodes, const TreeNodeIndex* children, Q* q,
                                bool syncFirst)
        {
            if (syncFirst && nodes.empty()) { syncNode(); }
            for (std::size_t first = 0; first < nodes.size(); first += maxListNodes)
            {
                NodeList list;
                list.count = int(std::min<std::size_t>(maxListNodes, nodes.size() - first));
                std::copy_n(nodes.begin() + first, list.count, list.nodes);
                uint64_t count = (first == 0 && syncFirst) ? ++numSyncs : 0;
                sumParentsList<<<1, numComponents, 0, exec>>>(list, children, q, nodeSignal, nodePeerPes.data(),
                                                              myPos, numNodePeers, count);
                checkGpuErrors(cudaGetLastError());
            }
        };
        // globQ of the node peers, indexed by rank, and the device copy in the order of nodePeers
        std::vector<Q*> peerQ(numRanks, nullptr);
        auto            allocGlobQ = [&](std::size_t numNodes)
        {
            globQ = static_cast<Q*>(nvshmem_calloc(numNodes, sizeof(Q)));
            std::vector<Q*> orderedQ;
            for (int p : nodePeers)
            {
                peerQ[p] = static_cast<Q*>(nvshmem_ptr(globQ, p));
                if (!peerQ[p]) { MPI_Abort(MPI_COMM_WORLD, 1); }
                orderedQ.push_back(peerQ[p]);
            }
            checkGpuErrors(cudaMalloc(&nodePeerQ, numNodePeers * sizeof(Q*)));
            checkGpuErrors(cudaMemcpy(nodePeerQ, orderedQ.data(), numNodePeers * sizeof(Q*), cudaMemcpyHostToDevice));
            return globQ;
        };
        auto scatterNodePeers = [&](std::span<const TreeNodeIndex> nodes, const Q* values)
        {
            scatterNodePeersKernel<<<nodes.size(), numComponents, 0, exec>>>(nodes.data(), values, nodePeerQ,
                                                                             numNodePeers);
            checkGpuErrors(cudaGetLastError());
        };
        auto loadNodePeer = [&](int peer, std::span<const TreeNodeIndex> nodes)
        {
            loadNodePeerKernel<<<nodes.size(), numComponents, 0, exec>>>(nodes.data(), peerQ[peer], globQ);
            checkGpuErrors(cudaGetLastError());
        };
        int steps = argc > 1 ? std::atoi(argv[1]) : 10;
        status = testGlobalExchange<DeviceVector>(exec, upsweepFunction, upsweepNodes, rank, numRanks, steps,
                                                  nodeComm, allocGlobQ, nodeMask(rank, nodeComm), loadNodePeer,
                                                  scatterNodePeers, syncNode);
    }
    checkGpuErrors(cudaFree(nodePeerQ));
    nvshmem_free(globQ);
    nvshmem_free(nodeSignal);
    MPI_Comm_free(&nodeComm);
    nvshmem_finalize();
    MPI_Finalize();
    return status;
}
