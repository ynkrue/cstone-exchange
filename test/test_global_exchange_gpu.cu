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
#include <cstring>
#include <nccl.h>
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

// messages of the node block, passed by value, src[i] is the globQ of the peer of message i
constexpr int maxPullMessages = 32;
struct PullMessages
{
    int           count;
    TreeNodeIndex displ[maxPullMessages + 1];
    const Q*      src[maxPullMessages];
};

/*! @brief syncs with the node peers like syncNodeKernel, then block (x, y) copies nodes[displ[y] + x] of message y
 *
 * Every block signals before it waits. CUDA does not guarantee that a given block runs while others spin, and setting
 * the same count again does not change the slot.
 */
__global__ void pullNodePeersKernel(PullMessages msgs, const TreeNodeIndex* nodes, Q* globQ, uint64_t* signal,
                                    const int* peerPes, int myPos, int numPeers, uint64_t count)
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

    int y = blockIdx.y;
    if (y < msgs.count && msgs.displ[y] + int(blockIdx.x) < msgs.displ[y + 1])
    {
        TreeNodeIndex n = nodes[msgs.displ[y] + blockIdx.x];
        globQ[n][threadIdx.x] = msgs.src[y][n][threadIdx.x];
    }
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

//! @brief abort on an NCCL error
void checkNccl(ncclResult_t result)
{
    if (result != ncclSuccess)
    {
        std::fprintf(stderr, "NCCL error: %s\n", ncclGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
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

    bool       useNccl = argc > 2 && std::strcmp(argv[2], "nccl") == 0;
    ncclComm_t ncclComm = nullptr;
    if (useNccl)
    {
        ncclUniqueId id;
        if (rank == 0) { checkNccl(ncclGetUniqueId(&id)); }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        checkNccl(ncclCommInitRank(&ncclComm, numRanks, id, rank));
    }
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
        auto pullNodePeers = [&](std::span<const int> peers, std::span<const TreeNodeIndex> displ,
                                 std::span<const TreeNodeIndex> nodes)
        {
            if (peers.size() > maxPullMessages)
            {
                std::fprintf(stderr, "too many node peers for one pull\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            PullMessages  msgs;
            TreeNodeIndex maxNodes = 1;
            msgs.count             = int(peers.size());
            for (int i = 0; i < msgs.count; ++i)
            {
                msgs.displ[i] = displ[i];
                msgs.src[i]   = peerQ[peers[i]];
                maxNodes      = std::max(maxNodes, displ[i + 1] - displ[i]);
            }
            msgs.displ[msgs.count] = displ[msgs.count];
            pullNodePeersKernel<<<dim3(maxNodes, std::max(msgs.count, 1)), numComponents, 0, exec>>>(
                msgs, nodes.data(), globQ, nodeSignal, nodePeerPes.data(), myPos, numNodePeers, ++numSyncs);
            checkGpuErrors(cudaGetLastError());
        };
        int steps = argc > 1 ? std::atoi(argv[1]) : 10;
        auto run  = [&](auto streamPhase)
        {
            return testGlobalExchange<DeviceVector>(exec, upsweepFunction, upsweepNodes, rank, numRanks, steps,
                                                    nodeComm, allocGlobQ, nodeMask(rank, nodeComm), pullNodePeers,
                                                    scatterNodePeers, streamPhase);
        };
        if (rank == 0) { std::printf("transport: %s\n", useNccl ? "nccl" : "mpi"); }
        if (useNccl)
        {
            // grouped sends and receives of a phase on the stream of the sweep kernels
            auto ncclPhase = [&](std::span<const int> sendPeers, std::span<const TreeNodeIndex> sendDispl,
                                 Q* sendBuf, std::span<const int> recvPeers, std::span<const TreeNodeIndex> recvDispl,
                                 Q* recvBuf)
            {
                checkNccl(ncclGroupStart());
                for (std::size_t i = 0; i < recvPeers.size(); ++i)
                {
                    checkNccl(ncclRecv(recvBuf + recvDispl[i], (recvDispl[i + 1] - recvDispl[i]) * sizeof(Q),
                                       ncclUint8, recvPeers[i], ncclComm, exec));
                }
                for (std::size_t i = 0; i < sendPeers.size(); ++i)
                {
                    checkNccl(ncclSend(sendBuf + sendDispl[i], (sendDispl[i + 1] - sendDispl[i]) * sizeof(Q),
                                       ncclUint8, sendPeers[i], ncclComm, exec));
                }
                checkNccl(ncclGroupEnd());
            };
            status = run(ncclPhase);
        }
        else { status = run(nullptr); }
    }
    if (useNccl) { checkNccl(ncclCommDestroy(ncclComm)); }
    checkGpuErrors(cudaFree(nodePeerQ));
    nvshmem_free(globQ);
    nvshmem_free(nodeSignal);
    MPI_Comm_free(&nodeComm);
    nvshmem_finalize();
    MPI_Finalize();
    return status;
}
