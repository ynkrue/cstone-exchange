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

/*! @brief true if node prefix fails the min-distance MAC against target, with the squared radius scaled by macScale */
template<class KeyType, class T>
bool macFails(KeyType prefix, const IBox& target, float invThetaEff, T macScale, const Box<T>& box)
{
    Vec4<T> c     = computeMinMacR2(prefix, invThetaEff, box);
    auto [tc, ts] = centerAndSize<KeyType>(target, box);
    return evaluateMacPbc(makeVec3(c), c[3] * macScale, tc, ts, box);
}

/*! @brief boxes of the ranks that get the data of rank after the round with low low bits, rank included */
template<class KeyType>
std::vector<IBox> receiverBoxes(const SfcAssignment<KeyType>& assignment, int rank, int low)
{
    std::vector<IBox> boxes;
    for (int k = 0; k < (assignment.numRanks() >> low); ++k)
    {
        int r = (rank & ((1 << low) - 1)) | (k << low);
        boxes.push_back(assignmentIBox(assignment[r], assignment[r + 1]));
    }
    return boxes;
}

/*! @brief true if some target needs node prefix
 *
 * The target does not overlap the node, the parent of the node fails the MAC for it and the node passes or is a leaf.
 */
template<class KeyType, class T>
bool needs(KeyType prefix, bool isLeaf, std::span<const IBox> targets, float invThetaEff, T tol, const Box<T>& box)
{
    unsigned level       = decodePrefixLength(prefix) / 3;
    KeyType  nodeStart   = decodePlaceholderBit(prefix);
    KeyType  parentStart = enclosingBoxCode(nodeStart, level - 1);
    KeyType  parent      = encodePlaceholderBit(parentStart, 3 * (level - 1));
    IBox     nodeBox     = sfcIBox(sfcKey(nodeStart), level);

    for (const IBox& t : targets)
    {
        if (overlap<KeyType>(nodeBox, t)) { continue; }
        if (!macFails(parent, t, invThetaEff, T(1) + tol, box)) { continue; }
        if (isLeaf || !macFails(prefix, t, invThetaEff, T(1) - tol, box)) { return true; }
    }
    return false;
}

/*! @brief number of rank bits exchanged after the given round
 *
 * 3 bits per round (radix 8) is the time optimum of the rank relay in the LogGP model and coincides with one tree
 * level per round (see .claude/docs/model.md Proposition 2).
 */
inline int lowBits(int leavesPerRank, int round) { return 3 * round - std::countr_zero(unsigned(leavesPerRank)); }

/*! @brief true if rank computes its node of the given round
 *
 * That is its level L - round ancestor, needed by a rank behind it or implied by a needed ancestor.
 * The receivers are the boxes of the ranks behind rank in this round (receiverBoxes).
 */
template<class KeyType, class T>
bool computesParent(OctreeView<const KeyType> tree, const SfcAssignment<KeyType>& assignment, int rank, int round,
                    std::span<const IBox> receivers, float invThetaEff, T tol, const Box<T>& box)
{
    int numLevels = maxDepth(tree.levelRange, maxTreeLevel<KeyType>{} + 2);

    for (int level = numLevels - round; level >= 1; --level)
    {
        KeyType a = encodePlaceholderBit(assignment[rank], 3 * level);
        if (needs(a, false, receivers, invThetaEff, tol, box)) { return true; }
    }
    return false;
}

/*! @brief nodes src sends to dst in the given round, sorted and unique
 *
 * The nodes of the cover of src needed by the receivers, which are the boxes of the ranks behind dst,
 * plus the cover if dst computes the common parent.
 */
template<class KeyType, class T>
std::vector<TreeNodeIndex> messageNodes(OctreeView<const KeyType> tree, const SfcAssignment<KeyType>& assignment,
                                        int src, int round, std::span<const IBox> receivers, bool dstComputesParent,
                                        float invThetaEff, T tol, const Box<T>& box)
{
    int numLevels = maxDepth(tree.levelRange, maxTreeLevel<KeyType>{} + 2);
    int level     = numLevels - round + 1;

    // cover of src in this round, the nodes it knows
    KeyType                    coverStart = enclosingBoxCode(assignment[src], level);
    KeyType                    coverEnd   = std::max(assignment[src + 1], coverStart + nodeRange<KeyType>(level));
    std::vector<TreeNodeIndex> cover;
    for (KeyType key = coverStart; key < coverEnd; key += nodeRange<KeyType>(level))
    {
        cover.push_back(locateNode(encodePlaceholderBit(key, 3 * level), tree.prefixes, tree.levelRange));
    }

    // descend while the node fails for some rank behind dst
    std::vector<TreeNodeIndex> nodes, stack = cover;
    while (!stack.empty())
    {
        TreeNodeIndex i = stack.back();
        stack.pop_back();
        bool isLeaf = tree.childOffsets[i] == 0;
        if (needs(tree.prefixes[i], isLeaf, receivers, invThetaEff, tol, box)) { nodes.push_back(i); }
        auto fails = [&](const IBox& t) { return macFails(tree.prefixes[i], t, invThetaEff, T(1) + tol, box); };
        if (!isLeaf && std::any_of(receivers.begin(), receivers.end(), fails))
        {
            for (int c = 0; c < 8; ++c)
            {
                stack.push_back(tree.childOffsets[i] + c);
            }
        }
    }

    if (dstComputesParent) { nodes.insert(nodes.end(), cover.begin(), cover.end()); }

    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return nodes;
}

//! @brief send, receive and compute lists of one rank, grouped by round
struct SweepLists
{
    std::vector<int>           sendPeers, recvPeers; // peer rank of each message, all rounds
    std::vector<TreeNodeIndex> sendDispl, recvDispl; // range of each message in sendIdx or recvIdx, numMsgs + 1 entries
    std::vector<TreeNodeIndex> sendOffsets, recvOffsets; // first message of each round, numRounds + 1 entries
    std::vector<TreeNodeIndex> sendIdx, recvIdx;         // node indices of all messages
    std::vector<TreeNodeIndex> computeOffsets;           // first compute node of each round, numRounds + 1 entries
    std::vector<TreeNodeIndex> computeIdx;               // parents computed at the end of each round
};

//! @brief relay shuffle lists of myRank, requires P a power of two and 1, 2 or 4 leaves per rank
template<class KeyType, class T>
SweepLists createSweepLists(OctreeView<const KeyType> tree, const SfcAssignment<KeyType>& assignment, float invThetaEff,
                            T tol, const Box<T>& box, int myRank)
{
    int numRanks  = assignment.numRanks();
    int lpr       = tree.numLeafNodes / numRanks;
    int numRounds = maxDepth(tree.levelRange, maxTreeLevel<KeyType>{} + 2);
    assert(std::has_single_bit(unsigned(numRanks)) && (lpr == 1 || lpr == 2 || lpr == 4));

    SweepLists lists;
    lists.sendOffsets = lists.recvOffsets = lists.computeOffsets = lists.sendDispl = lists.recvDispl = {0};

    for (int round = 1; round <= numRounds; ++round)
    {
        // partners differ from myRank only in the rank bits of this round, at most 7 partners
        int low   = lowBits(lpr, round);
        int first = std::max(0, lowBits(lpr, round - 1));
        int group = (1 << low) - (1 << first);
        int base  = myRank & ~group;

        auto myReceivers = receiverBoxes(assignment, myRank, low);
        bool myParent    = computesParent(tree, assignment, myRank, round, myReceivers, invThetaEff, tol, box);

        for (int peer = base; peer <= (base | group); peer += 1 << first)
        {
            if (peer == myRank) { continue; }

            auto peerReceivers = receiverBoxes(assignment, peer, low);
            bool peerParent    = computesParent(tree, assignment, peer, round, peerReceivers, invThetaEff, tol, box);
            auto sendNodes =
                messageNodes(tree, assignment, myRank, round, peerReceivers, peerParent, invThetaEff, tol, box);
            auto recvNodes = messageNodes(tree, assignment, peer, round, myReceivers, myParent, invThetaEff, tol, box);
            if (!sendNodes.empty())
            {
                lists.sendPeers.push_back(peer);
                lists.sendIdx.insert(lists.sendIdx.end(), sendNodes.begin(), sendNodes.end());
                lists.sendDispl.push_back(lists.sendIdx.size());
            }
            if (!recvNodes.empty())
            {
                lists.recvPeers.push_back(peer);
                lists.recvIdx.insert(lists.recvIdx.end(), recvNodes.begin(), recvNodes.end());
                lists.recvDispl.push_back(lists.recvIdx.size());
            }
        }
        lists.sendOffsets.push_back(lists.sendPeers.size());
        lists.recvOffsets.push_back(lists.recvPeers.size());

        if (myParent)
        {
            // parent of the covers exchanged in this round
            KeyType parent = encodePlaceholderBit(assignment[myRank], 3 * (numRounds - round));
            lists.computeIdx.push_back(locateNode(parent, tree.prefixes, tree.levelRange));
        }
        lists.computeOffsets.push_back(lists.computeIdx.size());
    }
    return lists;
}

/*! @brief exchange the needed nodes of this rank round by round and compute the parents in between
 *
 * @param[inout] globQ           values of all global nodes, own leaves set by the caller
 * @param[in]    upsweepNodes    upsweepNodes(nodes, globQ) sets nodes from their children
 */
template<class Exec, class Q, class UpsweepNodes>
void globalSweep(Exec exec, const SweepLists& lists, std::span<const TreeNodeIndex> sendIdx,
                 std::span<const TreeNodeIndex> recvIdx, std::span<Q> globQ, std::span<Q> sendBuf, std::span<Q> recvBuf,
                 UpsweepNodes&& upsweepNodes, int tag, MPI_Comm comm)
{
    int numRounds = int(lists.sendOffsets.size()) - 1;

    // post all receives first, one request per message
    std::vector<MPI_Request> recvRequests, sendRequests;
    for (std::size_t m = 0; m < lists.recvPeers.size(); ++m)
    {
        mpiRecvAsync(recvBuf.data() + lists.recvDispl[m], lists.recvDispl[m + 1] - lists.recvDispl[m],
                     lists.recvPeers[m], tag, recvRequests, comm);
    }

    for (int round = 0; round < numRounds; ++round)
    {
        // the messages of a round are contiguous, so one gather and one scatter per round
        TreeNodeIndex sendFirst = lists.sendOffsets[round], sendLast = lists.sendOffsets[round + 1];
        if (sendFirst != sendLast)
        {
            TreeNodeIndex first = lists.sendDispl[sendFirst];
            TreeNodeIndex count = lists.sendDispl[sendLast] - first;
            gather(exec, sendIdx.subspan(first, count), globQ.data(), sendBuf.data() + first);
            if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
            for (TreeNodeIndex m = sendFirst; m < sendLast; ++m)
            {
                mpiSendAsync(sendBuf.data() + lists.sendDispl[m], lists.sendDispl[m + 1] - lists.sendDispl[m],
                             lists.sendPeers[m], tag, sendRequests, comm);
            }
        }

        TreeNodeIndex recvFirst = lists.recvOffsets[round], recvLast = lists.recvOffsets[round + 1];
        if (recvFirst != recvLast)
        {
            TreeNodeIndex first = lists.recvDispl[recvFirst];
            TreeNodeIndex count = lists.recvDispl[recvLast] - first;
            MPI_Waitall(recvLast - recvFirst, recvRequests.data() + recvFirst, MPI_STATUSES_IGNORE);
            scatter(exec, recvIdx.subspan(first, count), recvBuf.data() + first, globQ.data());
        }

        std::span nodes(lists.computeIdx.data() + lists.computeOffsets[round],
                        lists.computeIdx.data() + lists.computeOffsets[round + 1]);
        upsweepNodes(nodes, globQ.data());
    }

    // the last scatter must finish before the next sweep receives into recvBuf
    if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUSES_IGNORE);
}

} // namespace global_sweep
