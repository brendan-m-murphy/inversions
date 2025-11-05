from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from queue import PriorityQueue
from typing import Any

import numpy as np
import xarray as xr


Node = tuple[int, int]
NodeList = list[Node]


def node_list_from_mask(mask: np.ndarray) -> NodeList:
    return list(zip(*np.where(mask)))


def partition_remainder(partition: list[NodeList], nx: int, ny: int) -> NodeList:
    part_sets = [set(part) for part in partition]  # faster lookup
    remainder = []
    for i in range(nx):
        for j in range(ny):
            if all((i, j) not in part for part in part_sets):
                remainder.append((i, j))
    return remainder


def split_node_list(nodes: NodeList) -> tuple[list[int], list[int]]:
    return tuple(map(list, zip(*nodes)))  # type: ignore


def node_list_weight(nodes: NodeList, weights: np.ndarray) -> int | float:
    return np.sum(weights[split_node_list(nodes)])


def partition_data_array(partition: list[NodeList], lat_coord, lon_coord) -> xr.DataArray:
    """Make DataArray based on partition."""
    comp_arr = np.zeros((len(lat_coord), len(lon_coord)), dtype="int")

    for i, part in enumerate(partition):
        comp_arr[split_node_list(part)] = i + 1

    return xr.DataArray(comp_arr, dims=["lat", "lon"], coords=[lat_coord, lon_coord])


def idx_of_half_cumsum(weights: Sequence) -> int:
    """Return index idx so that sum(weights[:idx]) and sum(weights[idx:]) are as close as possible."""
    sum1, sum2 = 0, sum(weights)
    idx = 0

    while sum1 < sum2:
        sum1 += weights[idx]
        sum2 -= weights[idx]
        idx += 1

    if idx > 0:
        old_sum1 = sum1 - weights[idx - 1]
        old_sum2 = sum2 + weights[idx - 1]
        if (sum1 - sum2) > (old_sum2 - old_sum1):
            return idx - 1
    return idx


# ------------------------------
# Greedy partitioning based on repeated bisection
# ------------------------------


@dataclass(order=True)
class PrioritizedItem:
    """Items with a priority.

    This allows us to specify a "priority", which will be used to sort
    the items.

    This is an idea from the Python docs for using a PriorityQueue with objects
    that aren't naturally comparable.
    """
    priority: int | float
    item: Any = field(compare=False)


class NodeListPriorityQueue:
    """Priority queue that stores NodeLists based on a ranking function.

    The `.get` method retrieves the NodeList with highest ranking.

    This allows use to implement a greedy algorithm by getting the highest ranking
    list of nodes, splitting it, and putting the subnodes back into the queue. This means
    we are always splitting the highest ranking node.
    """

    def __init__(self, ranking_func: Callable[[NodeList], float | int] | None = None) -> None:
        """Create NodeListPriorityQueue.

        Args:
            ranking_func: function ranking NodeLists. For instance, the total
              "weight" in the region defined by the NodeList. Defaults to the
              length of the list. Larger values have higher priority.

        """
        self.queue = PriorityQueue()
        self.rank = ranking_func if ranking_func is not None else len

    def __bool__(self) -> bool:
        return not self.queue.empty()

    def insert(self, nodes: NodeList) -> None:
        # use negative of rank since PriorityQueue returns min priority item
        # and we want to return max priority item
        self.queue.put_nowait(PrioritizedItem(-self.rank(nodes), nodes))

    def get(self) -> NodeList:
        return self.queue.get_nowait().item


SplittingFunction = Callable[[NodeList], tuple[NodeList]]


def partitioning(
    init_partition: list[NodeList],
    n_parts: int,
    split_function: SplittingFunction,
    ranking_function: Callable[[NodeList], int | float] | None = None,
) -> list[NodeList]:
    q = NodeListPriorityQueue(ranking_func=ranking_function)

    for part in init_partition:
        q.insert(part)

    done = []
    n = len(init_partition)

    # split highest ranking component until we reach the target number of components
    # or we can no longer split components (i.e. the queue is empty)
    while n < n_parts and q:
        part = q.get()
        subparts = split_function(part)

        # if split is degenerate, put part on done list
        # TODO: add hook for other "done" conditions
        if any(len(sp) == 0 for sp in subparts):
            done.append(part)
            continue

        n -= 1  # 'part' is no longer a component
        for sp in subparts:
            n += 1  # count each new component
            if len(sp) > 1:
                q.insert(sp)
            else:
                done.append(sp)

    # collect parts still on queue
    while q:
        done.append(q.get())

    return done


# ------------------------------
# Axis Parallel Split
# ------------------------------


# TODO: better axis choice for balanced case
def long_axis(nodes: NodeList) -> int:
    """Return 0 if projection of nodes onto x-axis longer, otherwise return 1."""
    if not nodes:
        return 0
    arr = np.array(nodes)
    return np.argmax(arr.max(axis=0) - arr.min(axis=0))  # type: ignore


def long_axis_weighted(nodes: NodeList, weights: np.ndarray) -> int:
    """Return 0 if spread along x-axis is longer, otherwise return 1."""
    if not nodes:
        return 0

    if len(weights) != len(nodes):
        weights = weights[split_node_list(nodes)]

    parr = np.array(nodes)
    warr = np.array(weights).reshape(-1, 1)

    c = (parr * warr).sum(axis=0) / warr.sum()

    spread = (warr * np.abs(parr - c)).sum(axis=0) / warr.sum()

    return np.argmax(spread)  # type: ignore


def _axis_parallel_split(
        nodes: NodeList, axis: int, weights: np.ndarray | None = None, balanced: bool = True, clean_splits: bool = False,
) -> tuple[NodeList, NodeList]:
    """Split nodes into two parts by dividing along a vertical or horizontal line."""
    nodes = sorted(nodes, key=lambda n: (n[axis], n[1 - axis]))

    if balanced and weights is not None:
        node_weights = weights[split_node_list(nodes)]
        idx = idx_of_half_cumsum(node_weights)  # type: ignore
    elif len(nodes) == 2:
        idx = 1
    else:
        idx = len(nodes) // 2

    if clean_splits:
        split1 = [n for n in nodes if n[axis] <= nodes[idx][axis]]
        split2 = [n for n in nodes if n[axis] > nodes[idx][axis]]
        return split1, split2

    return nodes[:idx], nodes[idx:]


def axis_parallel_split(
    nodes: NodeList, weights: np.ndarray | None = None, balanced: bool = True, clean_splits: bool = False,
) -> tuple[NodeList, NodeList]:
    axis = long_axis(nodes) if (weights is None or not balanced) else long_axis_weighted(nodes, weights)
    return _axis_parallel_split(nodes, axis, weights, balanced, clean_splits)


def axis_parallel_partitioning(
        init_partition: list[NodeList], n_parts: int, weights: np.ndarray | None = None, balanced: bool = True, clean_splits: bool = False,
) -> list[NodeList]:
    splitting_func = partial(axis_parallel_split, weights=weights, balanced=balanced, clean_splits=clean_splits)
    ranking_func = partial(node_list_weight, weights=weights) if weights is not None else None
    return partitioning(
        init_partition, n_parts=n_parts, split_function=splitting_func, ranking_function=ranking_func  # type: ignore
    )


# ------------------------------
# Inertial Split
# ------------------------------


def ortho_reg_cost(points: list[tuple], weights: np.ndarray, a: float, b: float) -> float:
    """Moment of inertia of points about axis with equation y = ax + b."""
    cost = 0

    for p, w in zip(points, weights):
        cost += w * (p[1] - a * p[0] - b)**2

    return cost / (1 + a**2)


def centroid(points: NodeList, weights: np.ndarray):
    parr = np.array(points)
    warr = np.array(weights).reshape(-1, 1)
    return (parr * warr).sum(axis=0) / warr.sum()


# TODO: add lat/lon coords to orthog regression (or generalise NodeList to list[tuple] or np.ndarray)
# TODO: accept list of weights rather than array
# TODO: allow equal weights for all points


def orthogonal_regression_coeff(points: NodeList, weights: np.ndarray) -> tuple[float, float]:
    """Return coefficients for orthogonal regression line."""
    parr = np.array(points)
    warr = np.array(weights).reshape(-1, 1)

    wavgs = (parr * warr).sum(axis=0) / warr.sum()

    m2 = (warr * (parr - wavgs)**2).sum(axis=0) # / warr.sum()
    mxy = ((warr * (parr - wavgs))[:, 0] * (parr - wavgs)[:, 1]).sum() # / warr.sum()

    diff = m2[1] - m2[0]
    a = (diff + np.sign(diff) * np.sqrt(diff**2 + 4 * mxy**2)) / (2 * mxy)  # slope
    b = wavgs[1] - a * wavgs[0]  # intercept

    # there are two solutions for the slope a, we'll compute both and see which line
    # has the lowest cost
    a2 = -1 / a
    b2 = wavgs[1] - a2 * wavgs[0]

    if ortho_reg_cost(points, weights, a, b) < ortho_reg_cost(points, weights, a2, b2):
        return a, b

    # alternative: if mxy > 0, we want positive root, and if mxy < 0, we want negative root
    # but in either case, we want to compute the larger root first, then get the smaller root
    # using that r_1 * r_2 = -1

    return a2, b2


def ortho_reg_x_value(a: float, b: float, point: tuple) -> float:
    """x value so that (x, ax + b) is the nearest point on line y = ax + b to the given 'point'."""
    x, y = point
    return x + a / (1 + a**2) * (y - a * x - b)


def inertial_split(nodes: NodeList, weights: np.ndarray, balanced: bool = True) -> tuple[NodeList, NodeList]:
    a, b = orthogonal_regression_coeff(nodes, weights[split_node_list(nodes)])

    # sort along "primary axis"
    nodes = sorted(nodes, key=lambda n: ortho_reg_x_value(a, b, n))

    if balanced and weights is not None:
        weights = weights[split_node_list(nodes)]
        idx = idx_of_half_cumsum(weights)  # type: ignore
    elif len(nodes) == 2:
        idx = 1
    else:
        idx = len(nodes) // 2

    # TODO: add option to "break ties"

    return nodes[:idx], nodes[idx:]


def inertial_partitioning(
    init_partition: list[NodeList], n_parts: int, weights: np.ndarray, balanced: bool = True
) -> list[NodeList]:
    splitting_func = partial(inertial_split, weights=weights, balanced=balanced)
    ranking_func = partial(node_list_weight, weights=weights)
    return partitioning(
        init_partition, n_parts=n_parts, split_function=splitting_func, ranking_function=ranking_func  # type: ignore
    )
