#pragma once

/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test of the p2p global sweep against an Allgather and full upsweep on an equal-segment global tree
 *
 * Checks the values of the far nodes marked by markMacs and times the level round schedule.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <span>
#include <vector>
#include <mpi.h>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/common.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/octree.hpp"
#include "global_sweep.hpp"

using namespace global_sweep;
using Q = util::array<double, 98>;

//! @brief stand-in for the multipole combination, sum of the 8 children
struct CombineSum
{
    template<class T>
    HOST_DEVICE_FUN T operator()(TreeNodeIndex /*nodeIdx*/, TreeNodeIndex c, const T* q) const
    {
        T sum = q[c];
        for (int k = 1; k < 8; ++k)
        {
            sum += q[c + k];
        }
        return sum;
    }
};

//! @brief copy the host tree into the execution space, stands in for buildOctreeGpu in GROMACS DomainFixed
template<class Exec, class KeyType>
OctreeData<KeyType, Exec> uploadOctree(const OctreeData<KeyType, execution::Cpu>& host)
{
    OctreeData<KeyType, Exec> acc;
    acc.resize(host.numLeafNodes);
    acc.prefixes       = host.prefixes;
    acc.childOffsets   = host.childOffsets;
    acc.parents        = host.parents;
    acc.levelRange     = host.levelRange;
    acc.d_levelRange   = host.d_levelRange;
    acc.internalToLeaf = host.internalToLeaf;
    acc.leafToInternal = host.leafToInternal;
    return acc;
}

//! @brief leaf values of step @p step, different in every step so stale values fail the check
template<class T>
void populate(int step, std::span<T> leafQ)
{
    for (std::size_t i = 0; i < leafQ.size(); ++i)
    {
        // small integers, so that sums over the tree are exact; components differ by the factor k + 1
        for (std::size_t k = 0; k < leafQ[i].size(); ++k)
        {
            leafQ[i][k] = double((i + step + 1) * (k + 1));
        }
    }
}

/*! @brief run the sweep test
 *
 * @param[in] allocGlobQ   allocGlobQ(numNodes) returns the zeroed node values of this rank in the execution space
 * @param[in] direct       true if the node peers are reachable by direct access
 * @param[in] extraLevels  uniform tree levels below the deepest level of the spanning tree, 0 for the spanning tree
 *
 * direct, pullNodePeers, scatterNodePeers and streamPhase go to globalSweep.
 */
template<template<class...> class Vector, class Exec, class UpsweepFunction, class UpsweepNodes, class AllocQ,
         class PullNodePeers, class ScatterNodePeers, class StreamPhase>
int testGlobalExchange(Exec exec, UpsweepFunction upsweepFunction, UpsweepNodes upsweepNodes, int rank, int numRanks,
                       int steps, MPI_Comm nodeComm, AllocQ allocGlobQ, bool direct, PullNodePeers pullNodePeers,
                       ScatterNodePeers scatterNodePeers, StreamPhase streamPhase, int extraLevels = 0)
{
    using KeyType = unsigned;

    ///! global tree
    std::vector<KeyType> boundaries(numRanks + 1);
    KeyType              stride = nodeRange<KeyType>(0) / numRanks;
    for (int r = 0; r < numRanks; ++r)
    {
        boundaries[r] = KeyType(r) * stride;
    }
    boundaries.back()                          = nodeRange<KeyType>(0);
    auto                                leaves = computeSpanningTree<KeyType>(boundaries);
    if (extraLevels > 0)
    {
        // uniform tree at the deepest level of the spanning tree plus the extra levels
        unsigned level = 0;
        for (std::size_t i = 0; i + 1 < leaves.size(); ++i)
        {
            level = std::max(level, treeLevel(leaves[i + 1] - leaves[i]));
        }
        level += extraLevels;
        leaves.resize((std::size_t(1) << (3 * level)) + 1);
        for (std::size_t i = 0; i + 1 < leaves.size(); ++i)
        {
            leaves[i] = KeyType(i) * nodeRange<KeyType>(level);
        }
        leaves.back() = nodeRange<KeyType>(0);
    }
    std::vector<unsigned>               leafCounts(nNodes(leaves), 1);
    OctreeData<KeyType, execution::Cpu> tree;
    tree.resize(nNodes(leaves));
    updateInternalTree<KeyType>(leaves, tree.data());
    auto hostTree = tree.cdata();

    Box<double> box(0, 1, BoundaryType::periodic);
    float       invThetaEff = invThetaMinMac(1.1f);
    double      tol         = 1e-4;

    // distribute global tree on exec
    auto gOctreeAcc = uploadOctree<Exec>(tree);
    auto gOctree    = gOctreeAcc.cdata();

    auto          assignment = makeSfcAssignment(numRanks, leafCounts, leaves.data());
    auto          displ      = assignment.treeOffsetsConst();
    TreeNodeIndex firstLeaf  = displ[rank];
    TreeNodeIndex numLeaves  = displ[rank + 1] - displ[rank];

    // far nodes of this rank, outside its range with a parent failing the MAC, passing or leaves
    std::vector<Vec4<double>> macCenters(hostTree.numNodes);
    for (TreeNodeIndex i = 0; i < hostTree.numNodes; ++i)
    {
        macCenters[i] = computeMinMacR2(hostTree.prefixes[i], invThetaEff, box);
    }
    std::vector<uint8_t> macs(hostTree.numNodes, 0);
    markMacs(hostTree.prefixes, hostTree.childOffsets, hostTree.parents, macCenters.data(), box,
             leaves.data() + firstLeaf, numLeaves, false, macs.data());
    std::vector<TreeNodeIndex> far;
    for (TreeNodeIndex i = 1; i < hostTree.numNodes; ++i)
    {
        auto [nodeStart, nodeEnd] = decodePlaceholderBit2K(hostTree.prefixes[i]);
        if (nodeStart < assignment[rank + 1] && nodeEnd > assignment[rank]) { continue; }
        if (macs[hostTree.parents[(i - 1) / 8]] && (!macs[i] || hostTree.childOffsets[i] == 0)) { far.push_back(i); }
    }

    ///! Relay setup
    MPI_Barrier(MPI_COMM_WORLD);
    double start    = MPI_Wtime();
    int    unitMask = nodeMask(rank, nodeComm);
    auto   masks =
        unitMask ? roundMasks(numRanks, unitMask) : levelMasks(numRanks, hostTree.numLeafNodes / numRanks);
    auto   lists     = createSweepLists<KeyType>(hostTree, assignment, masks, invThetaEff, tol, box, rank);
    double setupTime = MPI_Wtime() - start;

    Vector<TreeNodeIndex> sendIdx(lists.sendIdx), recvIdx(lists.recvIdx);
    Q*                    globQ = allocGlobQ(std::size_t(hostTree.numNodes));
    Vector<Q>             sendBuf(lists.sendIdx.size()), recvBuf(lists.recvIdx.size());
    constexpr int         tag = 0;

    ///! Allgather setup
    Vector<Q> allLeaves(hostTree.numLeafNodes);
    Vector<Q> allQ(hostTree.numNodes);

    auto download = [](const Vector<Q>& v)
    {
        if constexpr (execution::HaveGpu<Exec>{}) { return toHost(v); }
        else
        {
            return v;
        }
    };

    std::vector<Q>      leafQ(hostTree.numLeafNodes), reference(hostTree.numNodes), sweptQ(hostTree.numNodes);
    std::vector<double> allgatherTime(steps);
    std::vector<double> p2pTime(steps), p2pHostTime(steps);
    int                 allgatherErrors = 0;
    int                 p2pErrors       = 0;
    for (int step = 0; step < steps; ++step)
    {
        populate(step, std::span<Q>(leafQ));
        cstone::scatter(hostTree.leafToInternalSpan(), leafQ.data(), reference.data());
        cstone::upsweep(hostTree.levelRangeSpan(), hostTree.childOffsets, reference.data(), CombineSum{});
        Vector<Q> localQ(leafQ.data() + firstLeaf, leafQ.data() + firstLeaf + numLeaves);
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }

        ///! Allgather + full upsweep
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        mpiAllgatherGpuDirect(exec, localQ.data(), allLeaves.data(), numLeaves, MPI_COMM_WORLD);
        scatter(exec, gOctree.leafToInternalSpan(), allLeaves.data(), allQ.data());
        upsweepFunction(gOctree.levelRangeSpan(), gOctree.childOffsets, allQ.data());
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        allgatherTime[step] = MPI_Wtime() - start;

        // check all nodes from Allgather
        auto fullQ = download(allQ);
        for (TreeNodeIndex i = 0; i < hostTree.numNodes; ++i)
        {
            if (fullQ[i] != reference[i]) { ++allgatherErrors; }
        }

        ///! distributed p2p upsweep
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        scatter(exec, gOctree.leafToInternalSpan().subspan(firstLeaf, numLeaves), localQ.data(), globQ);
        globalSweep(
            exec, lists, {sendIdx.data(), sendIdx.size()}, {recvIdx.data(), recvIdx.size()},
            std::span<Q>(globQ, hostTree.numNodes), {sendBuf.data(), sendBuf.size()},
            {recvBuf.data(), recvBuf.size()}, [&](std::span<const TreeNodeIndex> nodes, Q* q, bool syncFirst)
            { upsweepNodes(nodes, gOctree.childOffsets, q, syncFirst); }, direct, pullNodePeers, scatterNodePeers,
            streamPhase, tag, MPI_COMM_WORLD);
        p2pHostTime[step] = MPI_Wtime() - start;
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        p2pTime[step] = MPI_Wtime() - start;

        // check the far nodes and own leaves of this rank from the p2p sweep
        if constexpr (execution::HaveGpu<Exec>{})
        {
            memcpyD2HAsync(exec, globQ, sweptQ.size(), sweptQ.data());
            syncGpu(exec);
        }
        else
        {
            std::copy(globQ, globQ + sweptQ.size(), sweptQ.begin());
        }
        for (TreeNodeIndex n : far)
        {
            if (sweptQ[n] != reference[n]) { ++p2pErrors; }
        }
        for (TreeNodeIndex i = firstLeaf; i < firstLeaf + numLeaves; ++i)
        {
            TreeNodeIndex n = hostTree.leafToInternal[hostTree.numInternalNodes + i];
            if (sweptQ[n] != reference[n]) { ++p2pErrors; }
        }
    }

    // sum of errors, maximum of times over ranks
    MPI_Allreduce(MPI_IN_PLACE, &allgatherErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &p2pErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &setupTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    // max and sum over ranks of far nodes, send and receive messages
    long rankCounts[3] = {long(far.size()), long(lists.sendPeers.size()), long(lists.recvPeers.size())};
    long maxCounts[3], sumCounts[3];
    MPI_Reduce(rankCounts, maxCounts, 3, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(rankCounts, sumCounts, 3, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    // messages to peers outside the node and their node counts
    long netCounts[2] = {0, 0}, maxNet[2], sumNet[2];
    for (std::size_t m = 0; m < lists.sendPeers.size(); ++m)
    {
        if ((lists.sendPeers[m] ^ rank) & ~unitMask)
        {
            ++netCounts[0];
            netCounts[1] += lists.sendDispl[m + 1] - lists.sendDispl[m];
        }
    }
    MPI_Reduce(netCounts, maxNet, 2, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(netCounts, sumNet, 2, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    std::vector<double> gatheredTimes(std::size_t(numRanks) * steps), gatheredP2pTimes(std::size_t(numRanks) * steps),
        gatheredHostTimes(std::size_t(numRanks) * steps);
    MPI_Allgather(allgatherTime.data(), steps, MPI_DOUBLE, gatheredTimes.data(), steps, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(p2pTime.data(), steps, MPI_DOUBLE, gatheredP2pTimes.data(), steps, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(p2pHostTime.data(), steps, MPI_DOUBLE, gatheredHostTimes.data(), steps, MPI_DOUBLE, MPI_COMM_WORLD);

    // max, mean and median over ranks for one step
    auto rankStats = [&](const std::vector<double>& gathered, int step, double& max, double& mean, double& median)
    {
        std::vector<double> rankTimes(numRanks);
        mean = 0;
        for (int r = 0; r < numRanks; ++r)
        {
            rankTimes[r] = gathered[std::size_t(r) * steps + step];
            mean += rankTimes[r];
        }
        mean /= numRanks;
        std::sort(rankTimes.begin(), rankTimes.end());
        max    = rankTimes.back();
        median = numRanks % 2 ? rankTimes[numRanks / 2] : (rankTimes[numRanks / 2 - 1] + rankTimes[numRanks / 2]) / 2;
    };

    if (rank == 0)
    {
        std::printf("tree: P %d, %d leaves, %d leaves/rank, Q %zu B, theta 1.1, tol %g, periodic box\n", numRanks,
                    hostTree.numLeafNodes, numLeaves, sizeof(Q), tol);
        std::printf("round masks: ");
        for (std::size_t t = 0; t < masks.size(); ++t)
        {
            std::printf(t ? " %x" : "%x", masks[t]);
        }
        std::printf("\n");
        std::printf("p2p setup [ms]: %.3f\n", setupTime * 1e3);
        std::printf("far nodes, send msgs, recv msgs (max / mean): %ld / %.1f, %ld / %.1f, %ld / %.1f\n", maxCounts[0],
                    double(sumCounts[0]) / numRanks, maxCounts[1], double(sumCounts[1]) / numRanks, maxCounts[2],
                    double(sumCounts[2]) / numRanks);
        std::printf("network send msgs, values (max / mean): %ld / %.1f, %ld / %.1f\n", maxNet[0],
                    double(sumNet[0]) / numRanks, maxNet[1], double(sumNet[1]) / numRanks);
        std::printf("step | allgather max [ms] | mean [ms] | median [ms] | p2p max [ms] | mean [ms] | median [ms] | host max [ms] | host median [ms]\n");
        for (int step = 0; step < steps; ++step)
        {
            double max, mean, median, p2pMax, p2pMean, p2pMedian, hostMax, hostMean, hostMedian;
            rankStats(gatheredTimes, step, max, mean, median);
            rankStats(gatheredP2pTimes, step, p2pMax, p2pMean, p2pMedian);
            rankStats(gatheredHostTimes, step, hostMax, hostMean, hostMedian);
            std::printf("%4d | %18.3f | %9.3f | %11.3f | %12.3f | %9.3f | %11.3f | %13.3f | %16.3f\n", step + 1,
                        max * 1e3, mean * 1e3, median * 1e3, p2pMax * 1e3, p2pMean * 1e3, p2pMedian * 1e3,
                        hostMax * 1e3, hostMedian * 1e3);
        }
        std::printf("allgather upsweep: %s (%d errors)\n", allgatherErrors ? "FAIL" : "PASS", allgatherErrors);
        std::printf("p2p sweep: %s (%d errors)\n", p2pErrors ? "FAIL" : "PASS", p2pErrors);
    }
    return allgatherErrors != 0 || p2pErrors != 0;
}
