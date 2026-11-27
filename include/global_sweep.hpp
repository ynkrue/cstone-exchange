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
#include <type_traits>
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
 * The first mask is 0, so the node block is empty. 3 bits per round (radix 8) is the time optimum of the rank
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
 * most, which meets the round bound at P 4096 with 4 ranks per node (see .claude/docs/model.md Proposition 3).
 */
inline std::vector<int> roundMasks(int numRanks, int unitMask)
{
    int maxBits   = 3 + std::popcount(unsigned(unitMask));
    int rest      = (numRanks - 1) & ~unitMask;
    int numBits   = std::popcount(unsigned(rest));
    int numRounds = (numBits + maxBits - 1) / maxBits;

    std::vector<int> masks{unitMask};
    for (int g = 0; g < numRounds; ++g)
    {
        int mask = masks.back();
        for (int b = 0; b < numBits / numRounds + (g < numBits % numRounds ? 1 : 0); ++b)
        {
            mask |= rest & -rest;
            rest &= rest - 1;
        }
        masks.push_back(mask);
    }
    return masks;
}

/*! @brief rank bits that differ within the node of rank, from the ranks of nodeComm in the SFC communicator
 *
 * Throws if the ranks of the node are not the 1 << popcount(mask) ranks that agree with rank outside the mask.
 */
inline int nodeMask(int rank, MPI_Comm nodeComm)
{
    int size;
    MPI_Comm_size(nodeComm, &size);
    std::vector<int> ranks(size);
    mpiAllgather(&rank, ranks.data(), 1, nodeComm);

    int mask = 0;
    for (int r : ranks)
    {
        mask |= r ^ rank;
    }
    // the ranks are distinct and agree with rank outside the mask, so they are the whole block if the size matches
    if (size != 1 << std::popcount(unsigned(mask)))
    {
        throw std::runtime_error("ranks of the node are not a subset block of the SFC ranks");
    }
    return mask;
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

/*! @brief send, receive and compute lists of one rank, grouped by phase
 *
 * Phase 0 is the local upsweep, phase 1 the node block, phases 2t and 2t + 1 the network and pool phase of round t.
 */
struct SweepLists
{
    std::vector<int>           sendPeers, recvPeers; // peer rank of each message, all phases
    std::vector<TreeNodeIndex> sendDispl, recvDispl; // range of each message in sendIdx or recvIdx, numMsgs + 1 entries
    std::vector<TreeNodeIndex> sendOffsets, recvOffsets; // first message of each phase, numPhases + 1 entries
    std::vector<TreeNodeIndex> sendIdx, recvIdx;         // node indices of all messages
    std::vector<TreeNodeIndex> computeOffsets;           // first compute node of each phase, numPhases + 1 entries
    std::vector<TreeNodeIndex> computeIdx;               // parents computed at the end of each phase
};

/*! @brief sweep lists of myRank for the cumulative rank bit masks of the rounds, requires P a power of two
 *
 * The local upsweep comes first. The node block exchanges the needed nodes between the node peers and computes the
 * node parents. Every network round has a handler phase, in which the handler ranks exchange the bundles of the node
 * pairs, and a pool phase, in which the received nodes go to the other node peers and the parents are computed. With
 * nodes of one rank the node block and the pool phases are empty.
 *
 * @param[in] masks  rank bits known after each round, the last one is P - 1, the first one the node mask
 */
template<class KeyType, class T>
SweepLists createSweepLists(OctreeView<const KeyType> tree, const SfcAssignment<KeyType>& assignment,
                            std::span<const int> masks, float invThetaEff, T tol, const Box<T>& box, int myRank)
{
    assert(std::has_single_bit(unsigned(assignment.numRanks())));

    SweepLists lists;
    lists.sendOffsets = lists.recvOffsets = lists.computeOffsets = lists.sendDispl = lists.recvDispl = {0};

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
    auto endPhase = [&]
    {
        lists.sendOffsets.push_back(lists.sendPeers.size());
        lists.recvOffsets.push_back(lists.recvPeers.size());
        lists.computeOffsets.push_back(lists.computeIdx.size());
    };

    // the local upsweep and the node block have all ranks as receivers, so myRank sends the same nodes to all peers
    auto allReceivers = receiverBoxes(assignment, myRank, 0);
    lists.computeIdx  = computeNodes(tree, assignment, masks, -1, myRank, allReceivers, invThetaEff, tol, box);
    endPhase();

    unsigned unitMask = masks[0];
    auto     ownNodes = messageNodes(tree, assignment, myRank, 0, allReceivers, invThetaEff, tol, box);
    for (unsigned j = unitMask & -unitMask; j != 0; j = (j - unitMask) & unitMask)
    {
        int peer = myRank ^ int(j);
        addSend(peer, ownNodes);
        addRecv(peer, messageNodes(tree, assignment, peer, 0, allReceivers, invThetaEff, tol, box));
    }
    auto nodeParents = computeNodes(tree, assignment, masks, 0, myRank, allReceivers, invThetaEff, tol, box);
    lists.computeIdx.insert(lists.computeIdx.end(), nodeParents.begin(), nodeParents.end());
    endPhase();

    // the ranks of the node of myRank in ascending order
    std::vector<int> nodePeers;
    unsigned         unitSub = 0;
    do
    {
        nodePeers.push_back((myRank & ~unitMask) | int(unitSub));
        unitSub = (unitSub - unitMask) & unitMask;
    } while (unitSub != 0);
    int numNodePeers = int(nodePeers.size());
    int myPos        = int(std::find(nodePeers.begin(), nodePeers.end(), myRank) - nodePeers.begin());

    for (int t = 1; t < int(masks.size()); ++t)
    {
        // partners differ from myRank only in the new bits of this round
        int prevMask = masks[t - 1];
        int fresh    = masks[t] & ~prevMask;
        int recvMask = masks[t] & ~masks[0];

        auto myReceivers = receiverBoxes(assignment, myRank, recvMask);

        // partner offsets of the round, node peer p handles the offsets p, p + numNodePeers, ...
        std::vector<int> offsets;
        for (unsigned j = fresh & -fresh; j != 0; j = (j - fresh) & fresh)
        {
            offsets.push_back(int(j));
        }

        std::vector<TreeNodeIndex> received;
        for (std::size_t k = myPos; k < offsets.size(); k += numNodePeers)
        {
            int  peer          = myRank ^ offsets[k];
            auto peerReceivers = receiverBoxes(assignment, peer, recvMask);
            addSend(peer, messageNodes(tree, assignment, myRank, prevMask, peerReceivers, invThetaEff, tol, box));
            auto recvNodes = messageNodes(tree, assignment, peer, prevMask, myReceivers, invThetaEff, tol, box);
            addRecv(peer, recvNodes);
            received.insert(received.end(), recvNodes.begin(), recvNodes.end());
        }
        endPhase();

        // the handlers pool the bundles of the node pairs to the other node peers
        std::sort(received.begin(), received.end());
        received.erase(std::unique(received.begin(), received.end()), received.end());
        for (int p = 0; p < numNodePeers; ++p)
        {
            if (p == myPos) { continue; }
            std::vector<TreeNodeIndex> pooled;
            for (std::size_t k = p; k < offsets.size(); k += numNodePeers)
            {
                auto nodes = messageNodes(tree, assignment, nodePeers[p] ^ offsets[k], prevMask, myReceivers,
                                          invThetaEff, tol, box);
                pooled.insert(pooled.end(), nodes.begin(), nodes.end());
            }
            std::sort(pooled.begin(), pooled.end());
            pooled.erase(std::unique(pooled.begin(), pooled.end()), pooled.end());
            addSend(nodePeers[p], received);
            addRecv(nodePeers[p], pooled);
        }
        auto parents = computeNodes(tree, assignment, masks, t, myRank, myReceivers, invThetaEff, tol, box);
        lists.computeIdx.insert(lists.computeIdx.end(), parents.begin(), parents.end());
        endPhase();
    }
    return lists;
}

/*! @brief exchange the needed nodes of this rank round by round and compute the parents in between
 *
 * The sweep starts with the local upsweep. With direct node peers the node block then syncs once, pulls the needed
 * nodes from the node peers and computes the node parents. Every network round has a handler phase that sends the
 * bundles between node pairs and scatters them into all node peers, and a pool phase whose sync is fused into the
 * parent computation. Without direct node peers every phase after the local upsweep exchanges messages.
 *
 * @param[inout] globQ             values of all global nodes, own leaves set by the caller
 * @param[in]    upsweepNodes      upsweepNodes(nodes, globQ, syncFirst) sets nodes from their children, if syncFirst
 *                                 it first waits until all node peers have finished the scatters of the phase, also
 *                                 for empty nodes
 * @param[in]    direct            true if the node peers are reachable by direct access
 * @param[in]    pullNodePeers     pullNodePeers(peers, displ, nodes) waits until the own range of all node peers is
 *                                 final, then writes globQ[n] of peers[i] into globQ[n] of this rank for n in
 *                                 nodes[displ[i], displ[i+1])
 * @param[in]    scatterNodePeers  scatterNodePeers(nodes, values) writes values[i] into globQ[nodes[i]] of all node
 *                                 peers and this rank
 * @param[in]    streamPhase       streamPhase(sendPeers, sendDispl, sendBuf, recvPeers, recvDispl, recvBuf) enqueues the
 *                                 messages of a phase on the stream of exec, nullptr selects MPI
 */
template<class Exec, class Q, class UpsweepNodes, class PullNodePeers, class ScatterNodePeers, class StreamPhase>
void globalSweep(Exec exec, const SweepLists& lists, std::span<const TreeNodeIndex> sendIdx,
                 std::span<const TreeNodeIndex> recvIdx, std::span<Q> globQ, std::span<Q> sendBuf, std::span<Q> recvBuf,
                 UpsweepNodes&& upsweepNodes, bool direct, PullNodePeers&& pullNodePeers,
                 ScatterNodePeers&& scatterNodePeers, StreamPhase&& streamPhase, int tag, MPI_Comm comm)
{
    int  numPhases = int(lists.sendOffsets.size()) - 1;
    auto parents   = [&](int phase)
    {
        return std::span(lists.computeIdx.data() + lists.computeOffsets[phase],
                         lists.computeIdx.data() + lists.computeOffsets[phase + 1]);
    };

    constexpr bool viaStream = !std::is_null_pointer_v<std::decay_t<StreamPhase>>;

    // post all receives first, one request per message, none for the node block and pool phases of direct peers
    std::vector<MPI_Request> recvRequests, sendRequests;
    if constexpr (!viaStream)
    {
        for (int phase = 0; phase < numPhases; ++phase)
        {
            for (TreeNodeIndex m = lists.recvOffsets[phase]; m < lists.recvOffsets[phase + 1]; ++m)
            {
                if (direct && phase % 2) { recvRequests.push_back(MPI_REQUEST_NULL); }
                else
                {
                    mpiRecvAsync(recvBuf.data() + lists.recvDispl[m], lists.recvDispl[m + 1] - lists.recvDispl[m],
                                 lists.recvPeers[m], tag, recvRequests, comm);
                }
            }
        }
    }

    upsweepNodes(parents(0), globQ.data(), false);

    if (direct)
    {
        // the pulled nodes lie in the ranges of the peers and are final after the sync, later scatters write outside
        // them, and the next step starts after the last sync of this one, so the pull needs no second sync
        TreeNodeIndex first = lists.recvOffsets[1], last = lists.recvOffsets[2];
        pullNodePeers(std::span(lists.recvPeers).subspan(first, last - first),
                      std::span(lists.recvDispl).subspan(first, last - first + 1), recvIdx);
        upsweepNodes(parents(1), globQ.data(), false);
    }

    // without direct node peers the node block is the first message phase
    for (int phase = direct ? 2 : 1; phase < numPhases; ++phase)
    {
        // the messages of a phase are contiguous, so one gather and one scatter per phase
        // with direct node peers the network phase scatters into all node peers, so the pool phase has no messages
        bool          pool      = direct && phase % 2;
        TreeNodeIndex sendFirst = lists.sendOffsets[phase], sendLast = lists.sendOffsets[phase + 1];
        TreeNodeIndex recvFirst = lists.recvOffsets[phase], recvLast = lists.recvOffsets[phase + 1];
        if (!pool && sendFirst != sendLast)
        {
            TreeNodeIndex first = lists.sendDispl[sendFirst];
            TreeNodeIndex count = lists.sendDispl[sendLast] - first;
            gather(exec, sendIdx.subspan(first, count), globQ.data(), sendBuf.data() + first);
            if constexpr (!viaStream)
            {
                if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
                for (TreeNodeIndex m = sendFirst; m < sendLast; ++m)
                {
                    mpiSendAsync(sendBuf.data() + lists.sendDispl[m], lists.sendDispl[m + 1] - lists.sendDispl[m],
                                 lists.sendPeers[m], tag, sendRequests, comm);
                }
            }
        }
        if constexpr (viaStream)
        {
            if (!pool)
            {
                streamPhase(std::span(lists.sendPeers).subspan(sendFirst, sendLast - sendFirst),
                            std::span(lists.sendDispl).subspan(sendFirst, sendLast - sendFirst + 1), sendBuf.data(),
                            std::span(lists.recvPeers).subspan(recvFirst, recvLast - recvFirst),
                            std::span(lists.recvDispl).subspan(recvFirst, recvLast - recvFirst + 1), recvBuf.data());
            }
        }

        if (!pool && recvFirst != recvLast)
        {
            TreeNodeIndex first = lists.recvDispl[recvFirst];
            TreeNodeIndex count = lists.recvDispl[recvLast] - first;
            if constexpr (!viaStream)
            {
                MPI_Waitall(recvLast - recvFirst, recvRequests.data() + recvFirst, MPI_STATUSES_IGNORE);
            }
            if (direct) { scatterNodePeers(recvIdx.subspan(first, count), recvBuf.data() + first); }
            else
            {
                scatter(exec, recvIdx.subspan(first, count), recvBuf.data() + first, globQ.data());
            }
        }

        // bundles of different partner nodes are disjoint and none lies in the known block of the own node, which
        // the node peers gather from, so the scatters need no sync before the one of the pool phase
        upsweepNodes(parents(phase), globQ.data(), pool);
    }

    // without stream phases the last scatter must finish before the next sweep receives into recvBuf
    if constexpr (!viaStream)
    {
        if constexpr (execution::HaveGpu<Exec>{}) { syncGpu(exec); }
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUSES_IGNORE);
    }
}

} // namespace global_sweep
