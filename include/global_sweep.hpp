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
#include <cmath>
#include <span>
#include <vector>
#include <mpi.h>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/common.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/macs.hpp"
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

//! @brief true if node prefix fails the min-distance MAC against target, with the squared radius scaled by macScale
template<class KeyType, class T>
bool macFails(KeyType prefix, const IBox& target, float invThetaEff, T macScale, const Box<T>& box)
{
    Vec4<T> c     = computeMinMacR2(prefix, invThetaEff, box);
    auto [tc, ts] = centerAndSize<KeyType>(target, box);
    return evaluateMacPbc(makeVec3(c), c[3] * macScale, tc, ts, box);
}

/*! @brief true if node prefix fails the MAC against some translate of target by multiples of spacing
 *
 * Spacing is in integer box units. The translates fold into the minimum image of a periodic box with the spacing as
 * edge. For an open global box this over-approximates, which is safe because it only controls the descent.
 */
template<class KeyType, class T>
bool latticeFails(KeyType prefix, const IBox& target, int spacing, float invThetaEff, T tol, const Box<T>& box)
{
    T      unit = T(spacing) / T(1 << maxTreeLevel<KeyType>{});
    Box<T> lattice(0, box.lx() * unit, 0, box.ly() * unit, 0, box.lz() * unit, BoundaryType::periodic,
                   BoundaryType::periodic, BoundaryType::periodic);

    Vec4<T> c     = computeMinMacR2(prefix, invThetaEff, box);
    auto [tc, ts] = centerAndSize<KeyType>(target, box);
    return evaluateMacPbc(makeVec3(c), c[3] * (T(1) + tol), tc, ts, lattice);
}

/*! @brief true if some translate of target by multiples of spacing needs node prefix
 *
 * The translate does not overlap the node, the parent of the node fails the MAC for it and the node passes or is a
 * leaf. Translates wrap around the global box, so each one is a rank box.
 */
template<class KeyType, class T>
bool latticeNeeds(KeyType prefix,
                  bool isLeaf,
                  const IBox& target,
                  int spacing,
                  float invThetaEff,
                  T tol,
                  const Box<T>& box)
{
    constexpr int maxCoord = 1 << maxTreeLevel<KeyType>{};

    unsigned level       = decodePrefixLength(prefix) / 3;
    KeyType  nodeStart   = decodePlaceholderBit(prefix);
    KeyType  parentStart = enclosingBoxCode(nodeStart, level - 1);
    KeyType  parent      = encodePlaceholderBit(parentStart, 3 * (level - 1));
    IBox     nodeBox     = sfcIBox(sfcKey(nodeStart), level);
    IBox     parentBox   = sfcIBox(sfcKey(parentStart), level - 1);

    int parentEdge   = parentBox.xmax() - parentBox.xmin();
    int nl           = maxCoord / spacing;
    int low[3]       = {target.xmin(), target.ymin(), target.zmin()};
    int size[3]      = {target.xmax() - target.xmin(), target.ymax() - target.ymin(), target.zmax() - target.zmin()};
    int parentMid[3] = {(parentBox.xmin() + parentBox.xmax()) / 2, (parentBox.ymin() + parentBox.ymax()) / 2,
                        (parentBox.zmin() + parentBox.zmax()) / 2};
    T   length[3]    = {box.lx(), box.ly(), box.lz()};

    // translate nearest to the parent centre and the window of translates around it
    int nearest[3], w[3];
    for (int d = 0; d < 3; ++d)
    {
        nearest[d] = std::lround((parentMid[d] - low[d] - size[d] / 2.0) / spacing);
        // acceptance radius of the parent in box units of dimension d
        double radius = invThetaEff * std::sqrt(1 + double(tol)) * parentEdge * (box.maxExtent() / length[d]);
        w[d]          = std::min(nl / 2, int(std::ceil(radius / spacing)) + 1);
    }

    // lower corner of the translate j in dimension d, wrapped into the global box
    auto shifted = [&](int d, int j)
    {
        int v = (low[d] + (nearest[d] + j) * spacing) % maxCoord;
        return v < 0 ? v + maxCoord : v;
    };

    for (int jx = -w[0]; jx <= w[0]; ++jx)
    {
        for (int jy = -w[1]; jy <= w[1]; ++jy)
        {
            for (int jz = -w[2]; jz <= w[2]; ++jz)
            {
                int  x = shifted(0, jx), y = shifted(1, jy), z = shifted(2, jz);
                IBox t(x, x + size[0], y, y + size[1], z, z + size[2]);
                if (overlap<KeyType>(nodeBox, t)) { continue; }
                if (!macFails(parent, t, invThetaEff, T(1) + tol, box)) { continue; }
                if (isLeaf || !macFails(prefix, t, invThetaEff, T(1) - tol, box)) { return true; }
            }
        }
    }
    return false;
}

/*! @brief far nodes of the rank owning [start, end), sorted
 *
 * Nodes outside the range whose parent fails the MAC against the range and that pass or are leaves.
 * The failing test is widened and the passing test narrowed by tol, so a node inside the band can be included.
 */
template<class KeyType, class T>
std::vector<TreeNodeIndex>
farNodes(OctreeView<const KeyType> tree, KeyType start, KeyType end, float invThetaEff, T tol, const Box<T>& box)
{
    // with the edge of the global box as spacing the only translate is the range itself
    constexpr int              spacing = 1 << maxTreeLevel<KeyType>{};
    IBox                       target  = assignmentIBox(start, end);
    std::vector<TreeNodeIndex> nodes;
    for (TreeNodeIndex i = 1; i < tree.numNodes; ++i)
    {
        if (latticeNeeds(tree.prefixes[i], tree.childOffsets[i] == 0, target, spacing, invThetaEff, tol, box))
        {
            nodes.push_back(i);
        }
    }
    return nodes;
}

/*! @brief true if rank computes its node of the given round
 *
 * That is its level L - round ancestor, needed by a rank behind it or implied by a needed ancestor.
 * The ranks behind it in this round are its box translated by multiples of 2^round leaves.
 */
template<class KeyType, class T>
bool computesParent(OctreeView<const KeyType> tree,
                    const SfcAssignment<KeyType>& assignment,
                    int rank,
                    int round,
                    float invThetaEff,
                    T tol,
                    const Box<T>& box)
{
    int  numLevels = maxDepth(tree.levelRange, maxTreeLevel<KeyType>{} + 2);
    IBox target    = assignmentIBox(assignment[rank], assignment[rank + 1]);
    int  spacing   = 1 << (int(maxTreeLevel<KeyType>{}) - numLevels + round);

    for (int level = numLevels - round; level >= 1; --level)
    {
        KeyType a = encodePlaceholderBit(assignment[rank], 3 * level);
        if (latticeNeeds(a, false, target, spacing, invThetaEff, tol, box)) { return true; }
    }
    return false;
}

//! @brief number of rank bits exchanged after the given round
inline int lowBits(int leavesPerRank, int round) { return 3 * round - std::countr_zero(unsigned(leavesPerRank)); }

/*! @brief nodes src sends to dst in the given round, sorted and unique
 *
 * The nodes of the cell of src needed by the ranks behind dst, plus the cell if dst computes the common parent.
 */
template<class KeyType, class T>
std::vector<TreeNodeIndex> messageNodes(OctreeView<const KeyType> tree,
                                        const SfcAssignment<KeyType>& assignment,
                                        int src,
                                        int dst,
                                        int round,
                                        float invThetaEff,
                                        T tol,
                                        const Box<T>& box)
{
    int  numLevels = maxDepth(tree.levelRange, maxTreeLevel<KeyType>{} + 2);
    int  level     = numLevels - round + 1;
    IBox target    = assignmentIBox(assignment[dst], assignment[dst + 1]);
    int  spacing   = 1 << (int(maxTreeLevel<KeyType>{}) - numLevels + round);

    // cell of src in this round, in round 1 its leaves (several nodes with 2 or 4 leaves per rank)
    KeyType cellStart = enclosingBoxCode(assignment[src], level);
    KeyType cellEnd   = std::max(assignment[src + 1], cellStart + nodeRange<KeyType>(level));
    std::vector<TreeNodeIndex> cell;
    for (KeyType key = cellStart; key < cellEnd; key += nodeRange<KeyType>(level))
    {
        cell.push_back(locateNode(encodePlaceholderBit(key, 3 * level), tree.prefixes, tree.levelRange));
    }

    // descend while the node fails for some rank behind dst
    std::vector<TreeNodeIndex> nodes, stack = cell;
    while (!stack.empty())
    {
        TreeNodeIndex i = stack.back();
        stack.pop_back();
        bool isLeaf = tree.childOffsets[i] == 0;
        if (latticeNeeds(tree.prefixes[i], isLeaf, target, spacing, invThetaEff, tol, box)) { nodes.push_back(i); }
        if (!isLeaf && latticeFails(tree.prefixes[i], target, spacing, invThetaEff, tol, box))
        {
            for (int c = 0; c < 8; ++c)
            {
                stack.push_back(tree.childOffsets[i] + c);
            }
        }
    }

    if (computesParent(tree, assignment, dst, round, invThetaEff, tol, box))
    {
        nodes.insert(nodes.end(), cell.begin(), cell.end());
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
template<class KeyType, class T>
SweepLists createSweepLists(OctreeView<const KeyType> tree,
                            const SfcAssignment<KeyType>& assignment,
                            float invThetaEff,
                            T tol,
                            const Box<T>& box,
                            int myRank)
{
    int numRanks  = assignment.numRanks();
    int lpr       = tree.numLeafNodes / numRanks;
    int numRounds = maxDepth(tree.levelRange, maxTreeLevel<KeyType>{} + 2);
    assert(std::has_single_bit(unsigned(numRanks)) && (lpr == 1 || lpr == 2 || lpr == 4));

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

            auto sendNodes = messageNodes(tree, assignment, myRank, peer, round, invThetaEff, tol, box);
            auto recvNodes = messageNodes(tree, assignment, peer, myRank, round, invThetaEff, tol, box);
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

        if (computesParent(tree, assignment, myRank, round, invThetaEff, tol, box))
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
