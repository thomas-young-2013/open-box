# License: MIT

from typing import List
import numpy as np


class Hypervolume:
    r"""Hypervolume computation dimension sweep algorithm from [Fonseca2006]_.

    Adapted from Simon Wessing's implementation of the algorithm
    (Variant 3, Version 1.2) in [Fonseca2006]_ in PyMOO:
    https://github.com/msu-coinlab/pymoo/blob/master/pymoo/vendor/hv.py

    Minimization is assumed.
    """

    def __init__(self, ref_point) -> None:
        r"""Initialize hypervolume object.

        Args:
            ref_point: `m`-dim numpy array containing the reference point.

        """
        assert ref_point is not None
        self.ref_point = np.asarray(ref_point)

    @property
    def ref_point(self):
        r"""Get reference point.

        Returns:
            A `m`-dim numpy array containing the reference point.
        """
        return self._ref_point

    @ref_point.setter
    def ref_point(self, ref_point) -> None:
        r"""Set the reference point

        Args:
            ref_point:  A `m`-dim numpy array containing the reference point.
        """
        assert ref_point is not None
        self._ref_point = np.asarray(ref_point)

    def compute(self, pareto_Y) -> float:
        r"""Compute hypervolume.

        Args:
            pareto_Y: A `n x m`-dim array of pareto optimal outcomes

        Returns:
            The hypervolume.
        """
        pareto_Y = np.atleast_2d(pareto_Y)
        if pareto_Y.shape[-1] != self._ref_point.shape[0]:
            raise Exception(
                "pareto_Y must have the same number of objectives as ref_point. "
                "Got %d, expected %d." % (pareto_Y.shape[-1], self._ref_point.shape[0])
            )
        elif pareto_Y.ndim != 2:
            raise Exception(
                "pareto_Y must have exactly two dimensions, got %d." % (pareto_Y.ndim)
            )

        better_than_ref = np.all(pareto_Y <= self._ref_point, axis=-1)
        pareto_Y = pareto_Y[better_than_ref]

        # Shift the pareto front so that reference point is all zeros
        pareto_Y = pareto_Y - self._ref_point
        self._initialize_multilist(pareto_Y)
        bounds = np.full_like(self._ref_point, float("-inf"))
        return self._hv_recursive(
            i=self._ref_point.shape[0] - 1, n_pareto=pareto_Y.shape[0], bounds=bounds
        )

    def _hv_recursive(self, i: int, n_pareto: int, bounds) -> float:
        r"""Recursive method for hypervolume calculation.

        This assumes minimization.

        In contrast to the paper, this code assumes that the reference point
        is the origin. This enables pruning a few operations.

        Args:
            i: objective index
            n_pareto: number of pareto points
            bounds: objective bounds

        Returns:
            The hypervolume.
        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if n_pareto == 0:
            # base case: one dimension
            return hvol
        elif i == 0:
            # base case: one dimension
            return -sentinel.next[0].data[0].item()
        elif i == 1:
            # two dimensions, end recursion
            q = sentinel.next[1]
            h = q.data[0]
            p = q.next[1]
            while p is not sentinel:
                hvol += h * (q.data[1] - p.data[1])
                if p.data[0] < h:
                    h = p.data[0]
                q = p
                p = q.next[1]
            hvol += h * q.data[1]
            return hvol
        else:
            p = sentinel
            q = p.prev[i]
            while q.data is not None:
                if q.ignore < i:
                    q.ignore = 0
                q = q.prev[i]
            q = p.prev[i]
            while n_pareto > 1 and (
                q.data[i] > bounds[i] or q.prev[i].data[i] >= bounds[i]
            ):
                p = q
                self.list.remove(p, i, bounds)
                q = p.prev[i]
                n_pareto -= 1
            q_prev = q.prev[i]
            if n_pareto > 1:
                hvol = q_prev.volume[i] + q_prev.area[i] * (q.data[i] - q_prev.data[i])
            else:
                q.area[0] = 1
                q.area[1: i + 1] = q.area[:i] * -(q.data[:i])
            q.volume[i] = hvol
            if q.ignore >= i:
                q.area[i] = q_prev.area[i]
            else:
                q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                if q.area[i] <= q_prev.area[i]:
                    q.ignore = i
            while p is not sentinel:
                p_data = p.data[i]
                hvol += q.area[i] * (p_data - q.data[i])
                bounds[i] = p_data
                self.list.reinsert(p, i, bounds)
                n_pareto += 1
                q = p
                p = p.next[i]
                q.volume[i] = hvol
                if q.ignore >= i:
                    q.area[i] = q.prev[i].area[i]
                else:
                    q.area[i] = self._hv_recursive(i - 1, n_pareto, bounds)
                    if q.area[i] <= q.prev[i].area[i]:
                        q.ignore = i
            hvol -= q.area[i] * q.data[i]
            return hvol

    def _initialize_multilist(self, pareto_Y) -> None:
        r"""Sets up the multilist data structure needed for calculation.

        Note: this assumes minimization.

        Args:
            pareto_Y: A `n x m`-dim tensor of pareto optimal objectives.

        """
        m = pareto_Y.shape[-1]
        nodes = [
            Node(m=m, data=point)
            for point in pareto_Y
        ]
        self.list = MultiList(m=m)
        for i in range(m):
            sort_by_dimension(nodes, i)
            self.list.extend(nodes, i)


class Node:
    r"""Node in the MultiList data structure."""

    def __init__(
        self,
        m: int,
        data=None,
    ) -> None:
        r"""Initialize MultiList.

        Args:
            m: The number of objectives
            data: The tensor data to be stored in this Node.
        """
        self.data = data
        self.next = [None] * m
        self.prev = [None] * m
        self.ignore = 0
        self.area = np.zeros(m)
        self.volume = np.zeros_like(self.area)


class MultiList:
    r"""A special data structure used in hypervolume computation.

    It consists of several doubly linked lists that share common nodes.
    Every node has multiple predecessors and successors, one in every list.
    """

    def __init__(self, m: int) -> None:
        r"""Initialize `m` doubly linked lists.

        Args:
            m: number of doubly linked lists
        """
        self.m = m
        self.sentinel = Node(m=m)
        self.sentinel.next = [self.sentinel] * m
        self.sentinel.prev = [self.sentinel] * m

    def append(self, node: Node, index: int) -> None:
        r"""Appends a node to the end of the list at the given index.

        Args:
            node: the new node
            index: the index where the node should be appended.
        """
        last = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = last
        # set the last element as the new one
        self.sentinel.prev[index] = node
        last.next[index] = node

    def extend(self, nodes: List[Node], index: int) -> None:
        r"""Extends the list at the given index with the nodes.

        Args:
            nodes: list of nodes to append at the given index.
            index: the index where the nodes should be appended.
        """
        for node in nodes:
            self.append(node=node, index=index)

    def remove(self, node: Node, index: int, bounds) -> Node:
        r"""Removes and returns 'node' from all lists in [0, 'index'].

        Args:
            node: The node to remove
            index: The upper bound on the range of indices
            bounds: A `2 x m`-dim array bounds on the objectives
        """
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor

        bounds.data = np.minimum(bounds, node.data)
        return node

    def reinsert(self, node: Node, index: int, bounds) -> None:
        r"""Re-inserts the node at its original position.

        Re-inserts the node at its original position in all lists in [0, 'index']
        before it was removed. This method assumes that the next and previous
        nodes of the node that is reinserted are in the list.

        Args:
            node: The node
            index: The upper bound on the range of indices
            bounds: A `2 x m`-dim tensor bounds on the objectives

        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
        bounds.data = np.minimum(bounds, node.data)


def sort_by_dimension(nodes: List[Node], i: int) -> None:
    r"""Sorts the list of nodes in-place by the specified objective.

    Args:
        nodes: A list of Nodes
        i: The index of the objective to sort by

    """
    # build a list of tuples of (point[i], node)
    decorated = [(node.data[i], index, node) for index, node in enumerate(nodes)]
    # sort by this value
    decorated.sort()
    # write back to original list
    nodes[:] = [node for (_, _, node) in decorated]
