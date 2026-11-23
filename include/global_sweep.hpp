/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GROMACS isolated global exchange
 *
 * @author Yannik Rüfenacht <yannik.rue@gmail.com>
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <span>
#include <vector>
#include <mpi.h>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/common.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/tree/octree.hpp"

namespace global_sweep
{
using namespace cstone;

//! @brief one message to or from a peer rank
struct SweepMessage
{
    int           peer;
    TreeNodeIndex offset, count; // range in sendIdx or recvIdx
};

//! @brief nodes the rank owning [start, end) needs from other ranks, siblings of its ancestors refined
//! numExtraLevels deeper
template<class KeyType>
std::vector<TreeNodeIndex> siblingNodes(OctreeView<const KeyType> tree, KeyType start, KeyType end, int numExtraLevels)
{
    std::vector<TreeNodeIndex> nodes;
    for (TreeNodeIndex i = 1; i < tree.numNodes; ++i)
    {
        KeyType key   = decodePlaceholderBit(tree.prefixes[i]);
        int     level = decodePrefixLength(tree.prefixes[i]) / 3;

        // base is the level of the coarsest ancestor of i outside the range, a sibling of an ancestor of the range
        int base = 1;
        for (; base <= level; ++base)
        {
            KeyType a = enclosingBoxCode(key, base);
            if (a >= end || a + nodeRange<KeyType>(base) <= start) { break; }
        }
        // keep i if it is numExtraLevels below its base, or a leaf above that
        int depth = level - base;
        if (depth >= 0 && (tree.childOffsets[i] == 0 ? depth <= numExtraLevels : depth == numExtraLevels))
        {
            nodes.push_back(i);
        }
    }
    return nodes;
}

//! @brief number of rank bits exchanged after the given round
inline int lowBits(int leavesPerRank, int round) { return 3 * round - std::countr_zero(unsigned(leavesPerRank)); }

//! @brief true if rank computes the parent of the cells exchanged in this round
template<class KeyType>
bool computesParent(OctreeView<const KeyType> tree,
                    const SfcAssignment<KeyType>& assignment,
                    std::span<const TreeNodeIndex> needOffsets,
                    std::span<const TreeNodeIndex> needNodes,
                    int rank,
                    int round)
{
    int numRanks = assignment.numRanks();
    int level    = maxDepth(tree.levelRange, maxTreeLevel<KeyType>{} + 2) - round;
    int bits     = lowBits(tree.numLeafNodes / numRanks, round);

    // the parent is needed if it or an ancestor is needed by a rank that gets data from rank later on,
    // these are the ranks with the same low bits
    KeyType cell = encodePlaceholderBit(assignment[rank], 3 * level);
    for (int t = rank & ((1 << bits) - 1); t < numRanks; t += 1 << bits)
    {
        for (TreeNodeIndex i = needOffsets[t]; i < needOffsets[t + 1]; ++i)
        {
            auto [nodeStart, nodeEnd] = decodePlaceholderBit2K(tree.prefixes[needNodes[i]]);
            if (containedIn(cell, nodeStart, nodeEnd)) { return true; }
        }
    }
    return false;
}

//! @brief nodes src sends to dst in the given round, sorted and unique
template<class KeyType>
std::vector<TreeNodeIndex> messageNodes(OctreeView<const KeyType> tree,
                                        const SfcAssignment<KeyType>& assignment,
                                        std::span<const TreeNodeIndex> needOffsets,
                                        std::span<const TreeNodeIndex> needNodes,
                                        int src,
                                        int dst,
                                        int round)
{
    int numRanks = assignment.numRanks();
    int level    = maxDepth(tree.levelRange, maxTreeLevel<KeyType>{} + 2) - round + 1;
    int bits     = lowBits(tree.numLeafNodes / numRanks, round);

    // cell of src in this round, in round 1 its leaves (several nodes with 2 or 4 leaves per rank)
    KeyType cellStart = enclosingBoxCode(assignment[src], level);
    KeyType cellEnd   = std::max(assignment[src + 1], cellStart + nodeRange<KeyType>(level));

    // nodes inside the cell needed by dst and by the ranks that get data from dst later on
    std::vector<TreeNodeIndex> nodes;
    for (int t = dst & ((1 << bits) - 1); t < numRanks; t += 1 << bits)
    {
        for (TreeNodeIndex i = needOffsets[t]; i < needOffsets[t + 1]; ++i)
        {
            if (containedIn(tree.prefixes[needNodes[i]], cellStart, cellEnd)) { nodes.push_back(needNodes[i]); }
        }
    }

    // the cell itself if dst computes the parent
    if (computesParent(tree, assignment, needOffsets, needNodes, dst, round))
    {
        for (KeyType key = cellStart; key < cellEnd; key += nodeRange<KeyType>(level))
        {
            nodes.push_back(locateNode(encodePlaceholderBit(key, 3 * level), tree.prefixes, tree.levelRange));
        }
    }

    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return nodes;
}

//! @brief send, receive and compute lists of one rank, grouped by round
struct SweepLists
{
    std::vector<SweepMessage>  send, recv;               // messages of all rounds
    std::vector<TreeNodeIndex> sendOffsets, recvOffsets; // first message of each round, numRounds + 1 entries
    std::vector<TreeNodeIndex> sendIdx, recvIdx;         // node indices of all messages
    std::vector<TreeNodeIndex> computeOffsets;           // first compute node of each round, numRounds + 1 entries
    std::vector<TreeNodeIndex> computeNodes;             // parents computed at the end of each round
};

//! @brief relay shuffle lists of myRank, requires P a power of two and 1, 2 or 4 leaves per rank
template<class KeyType>
SweepLists createSweepLists(OctreeView<const KeyType> tree,
                            const SfcAssignment<KeyType>& assignment,
                            int numExtraLevels,
                            int myRank)
{
    int numRanks  = assignment.numRanks();
    int lpr       = tree.numLeafNodes / numRanks;
    int numRounds = maxDepth(tree.levelRange, maxTreeLevel<KeyType>{} + 2);
    assert(std::has_single_bit(unsigned(numRanks)) && (lpr == 1 || lpr == 2 || lpr == 4));

    // needed nodes of every rank
    std::vector<TreeNodeIndex> needOffsets(numRanks + 1, 0), needNodes;
    for (int r = 0; r < numRanks; ++r)
    {
        auto nodes = siblingNodes(tree, assignment[r], assignment[r + 1], numExtraLevels);
        needNodes.insert(needNodes.end(), nodes.begin(), nodes.end());
        needOffsets[r + 1] = needNodes.size();
    }

    SweepLists lists;
    lists.sendOffsets = lists.recvOffsets = lists.computeOffsets = {0};

    for (int round = 1; round <= numRounds; ++round)
    {
        // partners differ from myRank only in the rank bits of this round, at most 7 partners
        int first = std::max(0, lowBits(lpr, round - 1));
        int group = (1 << lowBits(lpr, round)) - (1 << first);
        int base  = myRank & ~group;

        for (int peer = base; peer <= (base | group); peer += 1 << first)
        {
            if (peer == myRank) { continue; }

            auto sendNodes = messageNodes(tree, assignment, needOffsets, needNodes, myRank, peer, round);
            auto recvNodes = messageNodes(tree, assignment, needOffsets, needNodes, peer, myRank, round);
            if (!sendNodes.empty())
            {
                lists.send.push_back({peer, TreeNodeIndex(lists.sendIdx.size()), TreeNodeIndex(sendNodes.size())});
                lists.sendIdx.insert(lists.sendIdx.end(), sendNodes.begin(), sendNodes.end());
            }
            if (!recvNodes.empty())
            {
                lists.recv.push_back({peer, TreeNodeIndex(lists.recvIdx.size()), TreeNodeIndex(recvNodes.size())});
                lists.recvIdx.insert(lists.recvIdx.end(), recvNodes.begin(), recvNodes.end());
            }
        }
        lists.sendOffsets.push_back(lists.send.size());
        lists.recvOffsets.push_back(lists.recv.size());

        if (computesParent(tree, assignment, needOffsets, needNodes, myRank, round))
        {
            // parent of the cells exchanged in this round
            KeyType cell = encodePlaceholderBit(assignment[myRank], 3 * (numRounds - round));
            lists.computeNodes.push_back(locateNode(cell, tree.prefixes, tree.levelRange));
        }
        lists.computeOffsets.push_back(lists.computeNodes.size());
    }
    return lists;
}

/*! @brief exchange the needed nodes of this rank round by round and compute the parents in between
 *
 * @param[inout] globQ           values of all global nodes, own leaves set by the caller
 * @param[in]    computeParents  computeParents(nodes, childOffsets, globQ) sets nodes from their children
 */
template<class Exec, class Q, class ComputeParents>
void globalSweep(Exec exec,
                 const SweepLists& lists,
                 std::span<const TreeNodeIndex> sendIdx,
                 std::span<const TreeNodeIndex> recvIdx,
                 const TreeNodeIndex* childOffsets,
                 std::span<Q> globQ,
                 std::span<Q> sendBuf,
                 std::span<Q> recvBuf,
                 ComputeParents&& computeParents,
                 int tag,
                 MPI_Comm comm)
{
    int numRounds = int(lists.sendOffsets.size()) - 1;

    // entries of round e of an array grouped by round
    auto roundSlice = [](const auto& v, const auto& offsets, int e)
    { return std::span(v.data() + offsets[e], v.data() + offsets[e + 1]); };

    // post all receives first, one request per message in lists.recv
    std::vector<MPI_Request> recvRequests, sendRequests;
    for (const auto& m : lists.recv)
    {
        mpiRecvAsync(recvBuf.data() + m.offset, m.count, m.peer, tag, recvRequests, comm);
    }

    for (int e = 0; e < numRounds; ++e)
    {
        // the messages of a round are contiguous, so one gather and one scatter per round
        auto sends = roundSlice(lists.send, lists.sendOffsets, e);
        if (!sends.empty())
        {
            TreeNodeIndex first = sends.front().offset;
            TreeNodeIndex count = sends.back().offset + sends.back().count - first;
            gather(exec, sendIdx.subspan(first, count), globQ.data(), sendBuf.data() + first);
            if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
            for (const auto& m : sends)
            {
                mpiSendAsync(sendBuf.data() + m.offset, m.count, m.peer, tag, sendRequests, comm);
            }
        }

        auto recvs = roundSlice(lists.recv, lists.recvOffsets, e);
        if (!recvs.empty())
        {
            TreeNodeIndex first = recvs.front().offset;
            TreeNodeIndex count = recvs.back().offset + recvs.back().count - first;
            MPI_Waitall(int(recvs.size()), recvRequests.data() + lists.recvOffsets[e], MPI_STATUSES_IGNORE);
            scatter(exec, recvIdx.subspan(first, count), recvBuf.data() + first, globQ.data());
        }

        computeParents(roundSlice(lists.computeNodes, lists.computeOffsets, e), childOffsets, globQ.data());
    }

    // the last scatter must finish before the next sweep receives into recvBuf
    if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUSES_IGNORE);
}

} // namespace global_sweep
