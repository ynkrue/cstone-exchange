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
#include <stdexcept>
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

/*! @brief boxes of the ranks that agree with rank in the bits of mask, rank included
 *
 * These are the ranks that get the data of rank after the round, with the unit bits removed from the round mask.
 */
template<class KeyType>
std::vector<IBox> receiverBoxes(const SfcAssignment<KeyType>& assignment, int rank, int mask)
{
    unsigned          free = (assignment.numRanks() - 1) & ~mask;
    std::vector<IBox> boxes;
    unsigned          sub = 0;
    do
    {
        int r = (rank & mask) | sub;
        boxes.push_back(assignmentIBox(assignment[r], assignment[r + 1]));
        sub = (sub - free) & free;
    } while (sub != 0);
    return boxes;
}

/*! @brief how many ranks owning leaves of node prefix agree with rank outside the bits of mask
 *
 * Returns 0 for none, 1 for some and 2 for all. The ranks of a node of at least one rank form an aligned
 * power of two block, which holds for equal SFC segments and a power of two number of ranks.
 */
template<class KeyType>
int knownRanks(const SfcAssignment<KeyType>& assignment, KeyType prefix, int rank, int mask)
{
    auto [start, end] = decodePlaceholderBit2K(prefix);
    int lo            = assignment.findRank(start);
    int block         = assignment.findRank(end - 1) - lo;
    if ((lo ^ rank) & ~mask & ~block) { return 0; }
    return (block & ~mask) ? 1 : 2;
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

/*! @brief cumulative rank bits known after each round of the level schedule, 3 bits per tree level
 *
 * The first mask is 0, so round 0 has no messages. 3 bits per round (radix 8) is the time optimum of the rank
 * relay in the LogGP model and coincides with one tree level per round (see .claude/docs/model.md Proposition 2).
 * The first group has the bits that complete the tree level above the nodes of a rank, 3 - log2(leavesPerRank) mod 3,
 * and 3 if that is 0. The last group is truncated. Requires numRanks and leavesPerRank to be powers of two.
 */
inline std::vector<int> levelMasks(int numRanks, int leavesPerRank)
{
    int              rankBits = std::countr_zero(unsigned(numRanks));
    std::vector<int> masks{0};
    for (int bits = 3 - std::countr_zero(unsigned(leavesPerRank)) % 3;; bits += 3)
    {
        masks.push_back((1 << std::min(bits, rankBits)) - 1);
        if (bits >= rankBits) { break; }
    }
    return masks;
}

/*! @brief cumulative rank bits known after each round for units of ranks that exchange first
 *
 * The first mask is unitMask. The other rank bits follow lowest first in as few rounds as the radix allows, with
 * group sizes as equal as possible, larger groups first. A rank talks to 8 partners per round and unit rank at
 * most (see .claude/docs/model.md Proposition 3).
 */
inline std::vector<int> roundMasks(int numRanks, int unitMask)
{
    int maxBits = std::bit_width(unsigned(8 * (1 << std::popcount(unsigned(unitMask))) + 1)) - 1;
    int rest    = (numRanks - 1) & ~unitMask;
    int q       = std::popcount(unsigned(rest));
    int R       = (q + maxBits - 1) / maxBits;

    std::vector<int> masks{unitMask};
    for (int g = 0; g < R; ++g)
    {
        int mask = masks.back();
        for (int b = 0; b < q / R + (g < q % R ? 1 : 0); ++b)
        {
            mask |= rest & -rest;
            rest &= rest - 1;
        }
        masks.push_back(mask);
    }
    return masks;
}

/*! @brief true if the node or one of its ancestors is needed by some receiver */
template<class KeyType, class T>
bool covered(KeyType prefix, bool isLeaf, std::span<const IBox> receivers, float invThetaEff, T tol, const Box<T>& box)
{
    if (needs(prefix, isLeaf, receivers, invThetaEff, tol, box)) { return true; }

    KeyType nodeStart = decodePlaceholderBit(prefix);
    for (int level = decodePrefixLength(prefix) / 3 - 1; level >= 1; --level)
    {
        KeyType ancestor = encodePlaceholderBit(enclosingBoxCode(nodeStart, level), 3 * level);
        if (needs(ancestor, false, receivers, invThetaEff, tol, box)) { return true; }
    }
    return false;
}

//! @brief maximal tree nodes known to rank when it agrees with the ranks of the nodes outside the bits of mask
template<class KeyType>
std::vector<TreeNodeIndex> coverNodes(OctreeView<const KeyType> tree, const SfcAssignment<KeyType>& assignment,
                                      int rank, int mask)
{
    std::vector<TreeNodeIndex> cover, stack{0};
    while (!stack.empty())
    {
        TreeNodeIndex i = stack.back();
        stack.pop_back();
        int known = knownRanks(assignment, tree.prefixes[i], rank, mask);
        if (known == 2) { cover.push_back(i); }
        else if (known == 1)
        {
            for (int c = 0; c < 8; ++c)
            {
                stack.push_back(tree.childOffsets[i] + c);
            }
        }
    }
    return cover;
}

/*! @brief internal nodes rank computes at the end of round t, deepest first
 *
 * The nodes of the cover of rank with masks[t] that no single partner knows before the round and that are needed
 * by the receivers of rank or implied by a needed ancestor. Round -1 is the local upsweep, it computes the internal
 * nodes inside the range of rank.
 */
template<class KeyType, class T>
std::vector<TreeNodeIndex> computeNodes(OctreeView<const KeyType> tree, const SfcAssignment<KeyType>& assignment,
                                        std::span<const int> masks, int t, int rank, std::span<const IBox> receivers,
                                        float invThetaEff, T tol, const Box<T>& box)
{
    int prevMask = t > 0 ? masks[t - 1] : 0;

    std::vector<TreeNodeIndex> nodes, stack = coverNodes(tree, assignment, rank, t < 0 ? 0 : masks[t]);
    while (!stack.empty())
    {
        TreeNodeIndex i = stack.back();
        stack.pop_back();
        KeyType prefix = tree.prefixes[i];
        if (tree.childOffsets[i] == 0) { continue; }
        int first = assignment.findRank(decodePlaceholderBit(prefix));
        if (t >= 0 && knownRanks(assignment, prefix, first, prevMask) == 2) { continue; }

        // the root has no parent for the MAC test and nobody needs it
        if (i != 0 && covered(prefix, false, receivers, invThetaEff, tol, box)) { nodes.push_back(i); }
        for (int c = 0; c < 8; ++c)
        {
            stack.push_back(tree.childOffsets[i] + c);
        }
    }

    // tree nodes are ordered by level, so descending indices are deepest first
    std::sort(nodes.rbegin(), nodes.rend());
    return nodes;
}

/*! @brief nodes src sends to dst in a round, sorted and unique
 *
 * The nodes of the cover of src with prevMask that the receivers of dst need or that are implied by a needed
 * ancestor, plus the descendants of the cover that the receivers need.
 */
template<class KeyType, class T>
std::vector<TreeNodeIndex> messageNodes(OctreeView<const KeyType> tree, const SfcAssignment<KeyType>& assignment,
                                        int src, int prevMask, std::span<const IBox> receivers, float invThetaEff,
                                        T tol, const Box<T>& box)
{
    auto cover = coverNodes(tree, assignment, src, prevMask);

    // descend while the node fails for some receiver
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

    for (TreeNodeIndex i : cover)
    {
        if (covered(tree.prefixes[i], tree.childOffsets[i] == 0, receivers, invThetaEff, tol, box))
        {
            nodes.push_back(i);
        }
    }

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

/*! @brief rank bits that differ within the node of rank, from the ranks of nodeComm in the SFC communicator
 *
 * Throws if the ranks of the node are not the 1 << popcount(mask) ranks that agree with rank outside the mask.
 */
inline int nodeMask(int rank, MPI_Comm nodeComm)
{
    int size;
    MPI_Comm_size(nodeComm, &size);
    std::vector<int> ranks(size);
    MPI_Allgather(&rank, 1, MPI_INT, ranks.data(), 1, MPI_INT, nodeComm);

    int mask = 0;
    for (int r : ranks)
    {
        mask |= r ^ rank;
    }
    std::sort(ranks.begin(), ranks.end());
    bool ok =
        size == (1 << std::popcount(unsigned(mask))) && std::adjacent_find(ranks.begin(), ranks.end()) == ranks.end();
    ok = ok && std::all_of(ranks.begin(), ranks.end(), [&](int r) { return (r & ~mask) == (rank & ~mask); });
    if (!ok) { throw std::runtime_error("ranks of the node are not a subset block of the SFC ranks"); }
    return mask;
}

/*! @brief sweep lists of myRank for the cumulative rank bit masks of the rounds, requires P a power of two
 *
 * Each network round has two phases when a node has several ranks. In the first the handler ranks send and receive
 * the bundles of the node pairs, in the second the received nodes are pooled to the other node peers and the round
 * computes.
 *
 * @param[in] masks  rank bits known after each round, the last one is P - 1, the first one the unit of ranks
 */
template<class KeyType, class T>
SweepLists createSweepLists(OctreeView<const KeyType> tree, const SfcAssignment<KeyType>& assignment,
                            std::span<const int> masks, float invThetaEff, T tol, const Box<T>& box, int myRank)
{
    assert(std::has_single_bit(unsigned(assignment.numRanks())));

    SweepLists lists;
    lists.sendOffsets = lists.recvOffsets = lists.computeOffsets = lists.sendDispl = lists.recvDispl = {0};

    // the ranks of the node of myRank in ascending order
    unsigned         unitMask = masks[0];
    std::vector<int> nodePeers;
    unsigned         unitSub = 0;
    do
    {
        nodePeers.push_back((myRank & ~unitMask) | int(unitSub));
        unitSub = (unitSub - unitMask) & unitMask;
    } while (unitSub != 0);
    int numNodePeers = int(nodePeers.size());

    auto addSend = [&](int peer, const std::vector<TreeNodeIndex>& nodes)
    {
        if (nodes.empty()) { return; }
        lists.sendPeers.push_back(peer);
        lists.sendIdx.insert(lists.sendIdx.end(), nodes.begin(), nodes.end());
        lists.sendDispl.push_back(lists.sendIdx.size());
    };
    auto addRecv = [&](int peer, const std::vector<TreeNodeIndex>& nodes)
    {
        if (nodes.empty()) { return; }
        lists.recvPeers.push_back(peer);
        lists.recvIdx.insert(lists.recvIdx.end(), nodes.begin(), nodes.end());
        lists.recvDispl.push_back(lists.recvIdx.size());
    };
    auto endPhase = [&]()
    {
        lists.sendOffsets.push_back(lists.sendPeers.size());
        lists.recvOffsets.push_back(lists.recvPeers.size());
    };

    // the local upsweep is its own phase when the ranges hold internal nodes, the same on all ranks
    auto ownCover = coverNodes(tree, assignment, myRank, 0);
    if (std::any_of(ownCover.begin(), ownCover.end(), [&](TreeNodeIndex i) { return tree.childOffsets[i] != 0; }))
    {
        lists.computeIdx = computeNodes(tree, assignment, masks, -1, myRank, receiverBoxes(assignment, myRank, 0),
                                        invThetaEff, tol, box);
        lists.computeOffsets.push_back(lists.computeIdx.size());
        endPhase();
    }

    for (int t = 0; t < int(masks.size()); ++t)
    {
        // partners differ from myRank only in the new bits of this round
        int prevMask = t ? masks[t - 1] : 0;
        int fresh    = masks[t] & ~prevMask;
        int recvMask = masks[t] & ~masks[0];

        auto myReceivers = receiverBoxes(assignment, myRank, recvMask);

        // calls f(j) for the partner offsets j of the handler, the node bundle j goes to node peer k % numNodePeers
        auto forHandled = [&](int handler, auto&& f)
        {
            unsigned k = 0;
            for (unsigned j = fresh & -fresh; j != 0; j = (j - fresh) & fresh)
            {
                if (t == 0 || nodePeers[k % numNodePeers] == handler) { f(int(j)); }
                ++k;
            }
        };

        std::vector<TreeNodeIndex> received;
        forHandled(
            myRank,
            [&](int j)
            {
                int  peer          = myRank ^ j;
                auto peerReceivers = receiverBoxes(assignment, peer, recvMask);
                addSend(peer, messageNodes(tree, assignment, myRank, prevMask, peerReceivers, invThetaEff, tol, box));
                auto recvNodes = messageNodes(tree, assignment, peer, prevMask, myReceivers, invThetaEff, tol, box);
                addRecv(peer, recvNodes);
                received.insert(received.end(), recvNodes.begin(), recvNodes.end());
            });
        endPhase();

        // the handlers pool the bundles of the node pairs to the other node peers, no compute in the first phase
        if (t > 0 && numNodePeers > 1)
        {
            lists.computeOffsets.push_back(lists.computeIdx.size());
            std::sort(received.begin(), received.end());
            received.erase(std::unique(received.begin(), received.end()), received.end());
            for (int nodePeer : nodePeers)
            {
                if (nodePeer == myRank) { continue; }
                std::vector<TreeNodeIndex> pooled;
                forHandled(nodePeer,
                           [&](int j)
                           {
                               auto nodes = messageNodes(tree, assignment, nodePeer ^ j, prevMask, myReceivers,
                                                         invThetaEff, tol, box);
                               pooled.insert(pooled.end(), nodes.begin(), nodes.end());
                           });
                std::sort(pooled.begin(), pooled.end());
                pooled.erase(std::unique(pooled.begin(), pooled.end()), pooled.end());
                addSend(nodePeer, received);
                addRecv(nodePeer, pooled);
            }
            endPhase();
        }

        auto parents = computeNodes(tree, assignment, masks, t, myRank, myReceivers, invThetaEff, tol, box);
        lists.computeIdx.insert(lists.computeIdx.end(), parents.begin(), parents.end());
        lists.computeOffsets.push_back(lists.computeIdx.size());
    }
    return lists;
}

/*! @brief exchange the needed nodes of this rank round by round and compute the parents in between
 *
 * A phase is either entirely on the node or entirely on the network, so the first message of a phase decides for
 * all of its messages. Messages to node peers are direct stores into globQ of the peer at the same node indices, so
 * the receiver only syncs. Network messages are gathered and sent by MPI.
 *
 * @param[inout] globQ           values of all global nodes, own leaves set by the caller
 * @param[in]    upsweepNodes    upsweepNodes(nodes, globQ) sets nodes from their children
 * @param[in]    directMask      rank bits of the node peers reachable by direct stores, 0 for none
 * @param[in]    storeNodePeer   storeNodePeer(peer, nodes) writes globQ[n] of this rank into globQ[n] of peer
 * @param[in]    syncNode        returns when all node peers have finished their stores of the phase
 */
template<class Exec, class Q, class UpsweepNodes, class StoreNodePeer, class SyncNode>
void globalSweep(Exec exec, const SweepLists& lists, std::span<const TreeNodeIndex> sendIdx,
                 std::span<const TreeNodeIndex> recvIdx, std::span<Q> globQ, std::span<Q> sendBuf, std::span<Q> recvBuf,
                 UpsweepNodes&& upsweepNodes, int directMask, StoreNodePeer&& storeNodePeer, SyncNode&& syncNode,
                 int tag, MPI_Comm comm)
{
    int numRounds = int(lists.sendOffsets.size()) - 1;
    int myRank;
    MPI_Comm_rank(comm, &myRank);
    auto isDirect = [&](int peer) { return ((peer ^ myRank) & ~directMask) == 0; };

    // post all receives first, one request per message, none for direct messages
    std::vector<MPI_Request> recvRequests, sendRequests;
    for (std::size_t m = 0; m < lists.recvPeers.size(); ++m)
    {
        if (isDirect(lists.recvPeers[m])) { recvRequests.push_back(MPI_REQUEST_NULL); }
        else
        {
            mpiRecvAsync(recvBuf.data() + lists.recvDispl[m], lists.recvDispl[m + 1] - lists.recvDispl[m],
                         lists.recvPeers[m], tag, recvRequests, comm);
        }
    }

    // the own leaves are set, node peers may start storing
    if (directMask != 0) { syncNode(); }

    for (int round = 0; round < numRounds; ++round)
    {
        // the messages of a round are contiguous, so one gather and one scatter per round
        TreeNodeIndex sendFirst = lists.sendOffsets[round], sendLast = lists.sendOffsets[round + 1];
        if (sendFirst != sendLast && isDirect(lists.sendPeers[sendFirst]))
        {
            for (TreeNodeIndex m = sendFirst; m < sendLast; ++m)
            {
                storeNodePeer(lists.sendPeers[m],
                              sendIdx.subspan(lists.sendDispl[m], lists.sendDispl[m + 1] - lists.sendDispl[m]));
            }
        }
        else if (sendFirst != sendLast)
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
        if (recvFirst != recvLast && !isDirect(lists.recvPeers[recvFirst]))
        {
            TreeNodeIndex first = lists.recvDispl[recvFirst];
            TreeNodeIndex count = lists.recvDispl[recvLast] - first;
            MPI_Waitall(recvLast - recvFirst, recvRequests.data() + recvFirst, MPI_STATUSES_IGNORE);
            scatter(exec, recvIdx.subspan(first, count), recvBuf.data() + first, globQ.data());
        }

        // all node peers have stored this phase and read the earlier values
        if (directMask != 0) { syncNode(); }

        std::span nodes(lists.computeIdx.data() + lists.computeOffsets[round],
                        lists.computeIdx.data() + lists.computeOffsets[round + 1]);
        upsweepNodes(nodes, globQ.data());
    }

    // the last scatter must finish before the next sweep receives into recvBuf
    if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUSES_IGNORE);
}

} // namespace global_sweep
