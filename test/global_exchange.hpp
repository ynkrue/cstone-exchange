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

//! @brief leaf values of repetition @p rep, different in every repetition so stale values fail the check
template<class T>
void populate(int rep, std::span<T> leafQ)
{
    for (std::size_t i = 0; i < leafQ.size(); ++i)
    {
        // small integers, so that sums over the tree are exact; components differ by the factor k + 1
        for (std::size_t k = 0; k < leafQ[i].size(); ++k)
        {
            leafQ[i][k] = double((i + rep + 1) * (k + 1));
        }
    }
}

template<template<class...> class Vector, class Exec, class UpsweepFunction, class UpsweepNodes>
int testGlobalExchange(Exec exec, UpsweepFunction upsweepFunction, UpsweepNodes upsweepNodes, int rank, int numRanks)
{
    using KeyType      = unsigned;
    constexpr int reps = 10;

    ///! global tree
    std::vector<KeyType> boundaries(numRanks + 1);
    KeyType              stride = nodeRange<KeyType>(0) / numRanks;
    for (int r = 0; r < numRanks; ++r)
    {
        boundaries[r] = KeyType(r) * stride;
    }
    boundaries.back()                          = nodeRange<KeyType>(0);
    auto                                leaves = computeSpanningTree<KeyType>(boundaries);
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
    double start     = MPI_Wtime();
    auto   lists     = createSweepLists<KeyType>(hostTree, assignment, invThetaEff, tol, box, rank);
    double setupTime = MPI_Wtime() - start;

    Vector<TreeNodeIndex> sendIdx(lists.sendIdx), recvIdx(lists.recvIdx);
    Vector<Q>             globQ(hostTree.numNodes), sendBuf(lists.sendIdx.size()), recvBuf(lists.recvIdx.size());
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

    std::vector<Q>      leafQ(hostTree.numLeafNodes), reference(hostTree.numNodes);
    std::vector<double> allgatherTime(reps);
    std::vector<double> p2pTime(reps);
    int                 allgatherErrors = 0;
    int                 p2pErrors       = 0;
    for (int rep = 0; rep < reps; ++rep)
    {
        populate(rep, std::span<Q>(leafQ));
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
        allgatherTime[rep] = MPI_Wtime() - start;

        // check all nodes from Allgather
        auto fullQ = download(allQ);
        for (TreeNodeIndex i = 0; i < hostTree.numNodes; ++i)
        {
            if (fullQ[i] != reference[i]) { ++allgatherErrors; }
        }

        ///! distributed p2p upsweep
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();
        scatter(exec, gOctree.leafToInternalSpan().subspan(firstLeaf, numLeaves), localQ.data(), globQ.data());
        globalSweep(
            exec, lists, {sendIdx.data(), sendIdx.size()}, {recvIdx.data(), recvIdx.size()},
            std::span<Q>(globQ.data(), globQ.size()), {sendBuf.data(), sendBuf.size()},
            {recvBuf.data(), recvBuf.size()}, [&](std::span<const TreeNodeIndex> nodes, Q* q)
            { upsweepNodes(nodes, gOctree.childOffsets, q); }, tag, MPI_COMM_WORLD);
        p2pTime[rep] = MPI_Wtime() - start;

        // check the far nodes and own leaves of this rank from the p2p sweep
        auto sweptQ = download(globQ);
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
    std::vector<double> gatheredTimes(std::size_t(numRanks) * reps), gatheredP2pTimes(std::size_t(numRanks) * reps);
    MPI_Allgather(allgatherTime.data(), reps, MPI_DOUBLE, gatheredTimes.data(), reps, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(p2pTime.data(), reps, MPI_DOUBLE, gatheredP2pTimes.data(), reps, MPI_DOUBLE, MPI_COMM_WORLD);

    // max, mean and median over ranks of repetition rep
    auto rankStats = [&](const std::vector<double>& gathered, int rep, double& max, double& mean, double& median)
    {
        std::vector<double> rankTimes(numRanks);
        mean = 0;
        for (int r = 0; r < numRanks; ++r)
        {
            rankTimes[r] = gathered[std::size_t(r) * reps + rep];
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
        std::printf("p2p setup [ms]: %.3f\n", setupTime * 1e3);
        std::printf("far nodes, send msgs, recv msgs (max / mean): %ld / %.1f, %ld / %.1f, %ld / %.1f\n", maxCounts[0],
                    double(sumCounts[0]) / numRanks, maxCounts[1], double(sumCounts[1]) / numRanks, maxCounts[2],
                    double(sumCounts[2]) / numRanks);
        std::printf("rep | allgather max [ms] | mean [ms] | median [ms] | p2p max [ms] | mean [ms] | median [ms]\n");
        for (int rep = 0; rep < reps; ++rep)
        {
            double max, mean, median, p2pMax, p2pMean, p2pMedian;
            rankStats(gatheredTimes, rep, max, mean, median);
            rankStats(gatheredP2pTimes, rep, p2pMax, p2pMean, p2pMedian);
            std::printf("%3d | %18.3f | %9.3f | %11.3f | %12.3f | %9.3f | %11.3f\n", rep + 1, max * 1e3, mean * 1e3,
                        median * 1e3, p2pMax * 1e3, p2pMean * 1e3, p2pMedian * 1e3);
        }
        std::printf("allgather upsweep: %s (%d errors)\n", allgatherErrors ? "FAIL" : "PASS", allgatherErrors);
        std::printf("p2p sweep: %s (%d errors)\n", p2pErrors ? "FAIL" : "PASS", p2pErrors);
    }
    return allgatherErrors != 0 || p2pErrors != 0;
}
