"""Group-contiguity repair for Voronoi cartogram results."""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from typing import Any

import geopandas as gpd
import numpy as np

__all__ = ["make_groups_contiguous"]

# _compose_topology_permutation is intentionally private (leading underscore)
# but imported by api.py and result.py.


def _repair_contiguity(
    cells: list,
    groups: list,
    *,
    max_passes: int = 20,
    show_progress: bool = False,
    debug: bool = False,
    min_shared_length: float | None = None,
) -> tuple[np.ndarray, list[tuple[Any, list[int]]]]:
    """Core contiguity repair algorithm operating on raw lists.

    Parameters
    ----------
    cells : list
        Shapely geometry objects — the Voronoi cells, one per row.
    groups : list
        Group label for each row (same length as *cells*).
    max_passes : int
        Maximum number of repair passes (default 20).
    show_progress : bool
        Print per-pass progress summary.
    debug : bool
        Print one line per satellite repair attempt.

    Returns
    -------
    slot_of : np.ndarray of int, shape (n,)
        Permutation array: ``slot_of[d]`` is the slot (index into *cells*) that
        district *d* should occupy.  Apply directly as a numpy index:
        ``new_cells = cells[slot_of]`` or ``new_points = old_points[slot_of]``.
    discontiguous : list[tuple[Any, list[int]]]
        Remaining satellite components that could not be repaired: each entry is
        ``(group_id, slot_indices)`` where *slot_indices* are the slots belonging
        to that satellite.  Empty when fully converged.
    """
    from carto_flow.geo_utils.adjacency import find_adjacent_pairs

    n = len(cells)

    # --- Build slot adjacency graph (adj[s] = set of adjacent slots) ---
    raw_pairs = find_adjacent_pairs(cells, min_shared_length=min_shared_length)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j, _ in raw_pairs:
        adj[i].add(j)
        adj[j].add(i)

    # --- Initial permutation: identity ---
    slot_of = np.arange(n, dtype=np.intp)  # district d → slot (numpy for direct indexing)
    dist_at = list(range(n))  # slot s → district

    group_districts: dict = defaultdict(list)
    for d, g in enumerate(groups):
        group_districts[g].append(d)

    # --- Helpers ---

    def _connected_components(slots: list[int]) -> list[list[int]]:
        slot_set = set(slots)
        visited: set[int] = set()
        components: list[list[int]] = []
        for s in slots:
            if s in visited:
                continue
            comp: list[int] = []
            q: deque[int] = deque([s])
            visited.add(s)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb in slot_set and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            components.append(comp)
        return components

    def _enumerate_paths(
        src_slots: set[int],
        dst_slots: set[int],
        forbidden_nodes: set[int],
        max_len: int = 10,
        max_k: int = 20,
    ):
        """Yield simple paths from src_slots to dst_slots in non-decreasing length order.

        Nodes in forbidden_nodes are never included.  Yields at most max_k complete
        paths and only explores partial paths up to length max_len.

        Note: the heap stores full path tuples, so heap size is O(degree^max_len).
        Keep max_len small (≤ 12) to avoid memory blowup.
        """
        heap: list[tuple[int, tuple[int, ...]]] = []
        for s in src_slots:
            if s not in forbidden_nodes:
                heapq.heappush(heap, (1, (s,)))
        k = 0
        while heap and k < max_k:
            length, path_tup = heapq.heappop(heap)
            cur = path_tup[-1]
            if cur in dst_slots:
                yield list(path_tup)
                k += 1
                continue
            if length >= max_len:
                continue
            path_set = set(path_tup)
            for nb in adj[cur]:
                if nb not in path_set and nb not in forbidden_nodes:
                    heapq.heappush(heap, (length + 1, (*path_tup, nb)))

    def _do_swap(d1: int, d2: int) -> None:
        s1, s2 = slot_of[d1], slot_of[d2]
        slot_of[d1], slot_of[d2] = s2, s1
        dist_at[s1], dist_at[s2] = d2, d1

    # --- Initial progress report ---
    if show_progress:
        n_discontig = 0
        n_satellites = 0
        for dists in group_districts.values():
            if len(dists) < 2:
                continue
            comps = _connected_components([slot_of[d] for d in dists])
            if len(comps) > 1:
                n_discontig += 1
                n_satellites += len(comps) - 1
        print(f"[contiguity]  {n_discontig} discontiguous groups, {n_satellites} satellites total")

    # --- Main loop ---
    passes_done = 0
    for _pass in range(max_passes):
        any_swap = False
        pass_swaps = 0
        # Slots that changed hands earlier in this pass.  Excluded only as
        # *landing destinations* (not as traversal intermediates) to prevent
        # immediately re-displacing a district just placed there, while still
        # allowing other groups to route through those slots.
        pass_locked: set[int] = set()

        for grp, dists in group_districts.items():
            if len(dists) < 2:
                continue

            current_slots = [slot_of[d] for d in dists]
            comps = _connected_components(current_slots)
            if len(comps) == 1:
                continue

            comps.sort(key=lambda c: -len(c))
            main_slots = set(comps[0])

            # Sort satellites by BFS distance to main body (nearest first) so
            # that close satellites claim short paths and leave more of the
            # graph available to farther-away ones.
            def _sat_dist(sat_comp: list[int], _main_slots: set[int] = main_slots) -> int:
                visited: set[int] = set(sat_comp)
                frontier: list[int] = list(sat_comp)
                dist = 0
                while frontier:
                    dist += 1
                    nxt: list[int] = []
                    for s in frontier:
                        for nb in adj[s]:
                            if nb in _main_slots:
                                return dist
                            if nb not in visited:
                                visited.add(nb)
                                nxt.append(nb)
                    frontier = nxt
                return dist  # unreachable from main — sort last

            sat_comps_sorted = sorted(comps[1:], key=_sat_dist)

            for sat_comp in sat_comps_sorted:
                sat_slots = set(sat_comp)

                # Skip satellites that landed at a locked slot this pass — they
                # were created by a swap we just did and retrying now would undo
                # that swap (the cycle root cause).
                if sat_slots & pass_locked:
                    if debug:
                        print(
                            f"[contiguity debug] pass {_pass + 1}  group={grp}"
                            f"  sat={set(sat_comp)}  SKIPPED (created this pass)"
                        )
                    continue

                # Traversal-forbidden: only other satellites of the same group, to
                # prevent circular routes through the group's own territory.
                # pass_locked slots are excluded only as *destinations* (see below)
                # so BFS may still route through them as intermediates.
                other_sat_slots: set[int] = set()
                for c in comps[2:]:
                    other_sat_slots.update(c)

                # Landing destinations: main body minus pass_locked slots.  Routing
                # to a pass_locked slot would immediately re-displace the district
                # just placed there; routing *through* it is safe and handled by
                # Conditions A/B below.
                open_main = main_slots - pass_locked

                # Try paths in increasing length order.  When simulation rejects a
                # candidate (it would create a new satellite in another group), skip
                # to the next path rather than growing a forbidden-node set — a
                # growing forbidden set can block valid longer paths that share a
                # node with a failing shorter one.
                path = None
                for candidate in _enumerate_paths(sat_slots, open_main, other_sat_slots):
                    # Unified chain simulation: compute each displaced district's
                    # final slot, then check all affected groups for new satellites.
                    #
                    # After the chain: path[i] gets the district initially at
                    # path[i+1] for i=0..n-3; path[n-2] gets the satellite district.
                    n_cand = len(candidate)
                    affected_final: dict[int, int] = {}
                    for i in range(n_cand - 1):
                        d = dist_at[candidate[i]]
                        affected_final[d] = candidate[i - 1] if i > 0 else candidate[n_cand - 2]

                    bad_slot = -1
                    bad_grp = None
                    checked: set = set()
                    for i in range(n_cand - 1):
                        grp_b = groups[dist_at[candidate[i]]]
                        if grp_b in checked:
                            continue
                        checked.add(grp_b)
                        grp_b_dists = group_districts[grp_b]
                        if len(grp_b_dists) < 2:
                            continue
                        current_b_slots = [slot_of[dd] for dd in grp_b_dists]
                        if len(_connected_components(current_b_slots)) > 1:
                            continue  # already discontiguous — multi-pass will handle
                        final_b_slots = [affected_final.get(dd, slot_of[dd]) for dd in grp_b_dists]
                        if len(_connected_components(final_b_slots)) > 1:
                            for k in range(1, n_cand - 1):
                                if groups[dist_at[candidate[k]]] == grp_b:
                                    bad_slot = candidate[k]
                                    break
                            else:
                                bad_slot = candidate[n_cand - 2]
                            bad_grp = grp_b
                            break

                    if bad_slot != -1:
                        if debug:
                            print(
                                f"[contiguity debug] pass {_pass + 1}  group={grp}"
                                f"  sat={set(sat_comp)}  SKIP"
                                f"  sim-blocked at slot {bad_slot} (group={bad_grp})"
                            )
                        continue

                    path = candidate
                    break
                else:
                    if debug:
                        lock_note = f"  locked={len(pass_locked)}" if pass_locked else ""
                        print(
                            f"[contiguity debug] pass {_pass + 1}  group={grp}"
                            f"  sat={set(sat_comp)}  NO PATH"
                            f"  (candidates exhausted{lock_note})"
                        )

                if path is None:
                    continue

                # Execute the chain of swaps along the path.
                for k in range(1, len(path) - 1):
                    d_moving = dist_at[path[k - 1]]
                    d_displaced = dist_at[path[k]]
                    _do_swap(d_moving, d_displaced)
                    any_swap = True
                    pass_swaps += 1

                # Record all slots that changed hands so they can't be used as
                # landing destinations by subsequent satellites in this pass
                # (routing through them is still allowed; the simulation check
                # against cascading breaks).  The SKIPPED check above uses
                # pass_locked to detect satellites that were created this pass.
                pass_locked.update(path[:-1])

                if debug:
                    print(
                        f"[contiguity debug] pass {_pass + 1}  group={grp}"
                        f"  sat={set(sat_comp)}  SWAPPED {len(path) - 2}"
                        f"  path={path}"
                    )

                # Refresh main_slots after swaps.
                current_slots = [slot_of[d] for d in dists]
                comps = _connected_components(current_slots)
                comps.sort(key=lambda c: -len(c))
                main_slots = set(comps[0])

        passes_done = _pass + 1
        if show_progress and pass_swaps > 0:
            remaining = sum(
                1
                for dists in group_districts.values()
                if len(dists) >= 2 and len(_connected_components([slot_of[d] for d in dists])) > 1
            )
            print(f"[contiguity]  pass {_pass + 1:2d}  swaps={pass_swaps:<4d}  discontiguous={remaining}")

        if not any_swap:
            break

    if show_progress:
        remaining = sum(
            1
            for dists in group_districts.values()
            if len(dists) >= 2 and len(_connected_components([slot_of[d] for d in dists])) > 1
        )
        if remaining == 0:
            print(f"[contiguity]  converged after {passes_done} pass{'es' if passes_done != 1 else ''}")
        else:
            print(f"[contiguity]  stopped at max_passes={max_passes}  discontiguous={remaining}")

    # Collect remaining satellites: one (group_id, slot_indices) entry per component
    discontiguous: list[tuple[Any, list[int]]] = []
    for grp, dists in group_districts.items():
        if len(dists) < 2:
            continue
        comps = _connected_components([slot_of[d] for d in dists])
        if len(comps) <= 1:
            continue
        comps.sort(key=lambda c: -len(c))  # comps[0] = main body
        for sat_comp in comps[1:]:
            discontiguous.append((grp, sorted(sat_comp)))

    return slot_of, discontiguous


def _repair_compactness(
    cells: list,
    groups: list,
    *,
    max_passes: int = 10,
    min_shared_length: float | None = None,
) -> np.ndarray:
    """Permute slots to improve spatial compactness of each group.

    For each pair of adjacent slots belonging to different groups, a swap is
    accepted when the following dimensionless criterion is negative:

        delta_inertia / I_0  -  delta_shared / S_0  <  0

    where *I_0* is the total inertia of the two groups (sum of squared
    distances to group centroid) and *S_0* is the total intra-group shared
    edge length of the two groups.  Both terms are signed fractions: a
    negative inertia fraction means cells move closer to their group centroid;
    a negative shared-edge fraction means less internal cohesion.  A small
    loss in one metric can be accepted when offset by a larger gain in the
    other.  If *S_0 = 0* (groups have no internal shared edges), only inertia
    counts.

    Contiguity of both groups is verified before each swap is applied.

    Parameters
    ----------
    cells : list
        Shapely geometry objects — Voronoi cells, district-indexed
        (``cells[d]`` is the cell currently assigned to district *d*).
    groups : list
        Group label for each district (same length as *cells*).
    max_passes : int
        Maximum improvement passes (default 10).
    min_shared_length : float or None
        Minimum shared border length for two cells to be considered adjacent.
        Must match the value used by the rest of the pipeline to ensure
        consistent adjacency.

    Returns
    -------
    slot_of : np.ndarray[int], shape (n,)
        Permutation array: ``slot_of[d]`` is the slot (index into *cells*) that
        district *d* should occupy.  Identity when no improvement was found.
    """
    from carto_flow.geo_utils.adjacency import find_adjacent_pairs

    n = len(cells)
    slot_of = np.arange(n, dtype=np.intp)
    dist_at = list(range(n))

    raw_pairs = find_adjacent_pairs(cells, min_shared_length=min_shared_length)
    adj: list[set[int]] = [set() for _ in range(n)]
    # edge_len[(min(a,b), max(a,b))] = shared border length
    edge_len: dict[tuple[int, int], float] = {}
    for s1, s2, length in raw_pairs:
        adj[s1].add(s2)
        adj[s2].add(s1)
        edge_len[(min(s1, s2), max(s1, s2))] = length

    group_districts: dict = defaultdict(list)
    for d, g in enumerate(groups):
        group_districts[g].append(d)

    cell_xy = np.array([[c.centroid.x, c.centroid.y] for c in cells])

    def _connected(slots: set[int]) -> bool:
        if len(slots) <= 1:
            return True
        start = next(iter(slots))
        visited = {start}
        q: deque[int] = deque([start])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb in slots and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        return len(visited) == len(slots)

    def _connected_components(slots: list[int]) -> list[list[int]]:
        slot_set = set(slots)
        visited: set[int] = set()
        components: list[list[int]] = []
        for s in slots:
            if s in visited:
                continue
            comp: list[int] = []
            q: deque[int] = deque([s])
            visited.add(s)
            while q:
                cur = q.popleft()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb in slot_set and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            components.append(comp)
        return components

    def _enum_paths(
        src_slots: set[int],
        forbidden_nodes: set[int],
        min_len: int = 3,
        max_len: int = 6,
        max_k: int = 50,
    ):
        """Yield simple paths from src_slots of length in [min_len, max_len].

        All paths are extended up to max_len; those reaching min_len are yielded.
        There are no destination slots — the caller filters by content.
        """
        heap: list[tuple[int, tuple[int, ...]]] = []
        for s in src_slots:
            if s not in forbidden_nodes:
                heapq.heappush(heap, (1, (s,)))
        k = 0
        while heap and k < max_k:
            length, path_tup = heapq.heappop(heap)
            if length >= min_len:
                yield list(path_tup)
                k += 1
            if length < max_len:
                path_set = set(path_tup)
                cur = path_tup[-1]
                for nb in adj[cur]:
                    if nb not in path_set and nb not in forbidden_nodes:
                        heapq.heappush(heap, (length + 1, (*path_tup, nb)))

    def _elen(a: int, b: int) -> float:
        return edge_len.get((min(a, b), max(a, b)), 0.0)

    # Cell areas (slot-indexed) used as per-district weights.
    cell_area = np.array([c.area for c in cells], dtype=np.float64)

    for _pass in range(max_passes):
        # Recompute area-weighted group centroids from current slot assignments.
        centroid: dict = {}
        for g, dists in group_districts.items():
            slots_g = [slot_of[d] for d in dists]
            pts = cell_xy[slots_g]
            w_g = cell_area[slots_g]
            w_sum = w_g.sum()
            centroid[g] = np.average(pts, axis=0, weights=w_g) if w_sum > 0 else pts.mean(axis=0)

        locked: set[int] = set()
        improved = False

        for s1 in range(n):
            if s1 in locked:
                continue
            d1 = dist_at[s1]
            g1 = groups[d1]
            for s2 in adj[s1]:
                if s2 <= s1 or s2 in locked:
                    continue
                d2 = dist_at[s2]
                g2 = groups[d2]
                if g1 == g2:
                    continue

                # --- Metric 1: inertia delta with fixed centroids ---
                # delta = 2 * dot(pos[s2] - pos[s1], c_g2 - c_g1)
                # delta < 0 means each cell moves closer to its group centroid.
                delta_inertia = 2.0 * float(
                    np.dot(
                        cell_xy[s2] - cell_xy[s1],
                        centroid[g2] - centroid[g1],
                    )
                )

                # --- Metric 2: intra-group shared edge length delta ---
                # After the swap s1→g2, s2→g1:
                #   g1 loses s1's edges to g1_remaining, gains s2's edges to g1_remaining
                #   g2 loses s2's edges to g2_remaining, gains s1's edges to g2_remaining
                # Compute incrementally using the current slot assignment.
                g1_slots = {slot_of[d] for d in group_districts[g1]}
                g2_slots = {slot_of[d] for d in group_districts[g2]}
                g1_rem = g1_slots - {s1}
                g2_rem = g2_slots - {s2}
                delta_shared = (
                    sum(_elen(s2, nb) for nb in adj[s2] if nb in g1_rem)
                    - sum(_elen(s1, nb) for nb in adj[s1] if nb in g1_rem)
                    + sum(_elen(s1, nb) for nb in adj[s1] if nb in g2_rem)
                    - sum(_elen(s2, nb) for nb in adj[s2] if nb in g2_rem)
                )

                # Dimensionless criterion: accept when
                #   delta_inertia / I_0  -  delta_shared / S_0  <  0
                # Both terms are signed fractions of the combined group totals.
                # I_0 = total inertia of g1 U g2; S_0 = total intra-group
                # shared edge length of g1 U g2.
                g1_list = list(g1_slots)
                g2_list = list(g2_slots)
                g1_arr = cell_xy[g1_list]
                g2_arr = cell_xy[g2_list]
                w1 = cell_area[g1_list]
                w1 = w1 / w1.sum() if w1.sum() > 0 else np.ones(len(g1_list)) / len(g1_list)
                w2 = cell_area[g2_list]
                w2 = w2 / w2.sum() if w2.sum() > 0 else np.ones(len(g2_list)) / len(g2_list)
                i0_g1 = float(np.sum(w1 * np.sum((g1_arr - centroid[g1]) ** 2, axis=1)))
                i0_g2 = float(np.sum(w2 * np.sum((g2_arr - centroid[g2]) ** 2, axis=1)))
                I_0 = i0_g1 + i0_g2
                if I_0 == 0.0:
                    continue
                S_0 = sum(_elen(a, b) for a in g1_slots for b in adj[a] if b in g1_slots and b > a) + sum(
                    _elen(a, b) for a in g2_slots for b in adj[a] if b in g2_slots and b > a
                )
                criterion = delta_inertia / I_0 - (delta_shared / S_0 if S_0 > 0.0 else 0.0)
                if criterion >= 0.0:
                    continue

                # Both groups must remain connected after the swap.
                if not _connected(g1_rem | {s2}):
                    continue
                if not _connected(g2_rem | {s1}):
                    continue

                # Apply swap.
                slot_of[d1], slot_of[d2] = s2, s1
                dist_at[s1], dist_at[s2] = d2, d1
                locked.add(s1)
                locked.add(s2)
                improved = True
                break

        # --- BFS-chain pass ---
        # For each group, find the most outlying cell and try to improve it via
        # a short displacement chain.  Processing per group (not per district)
        # keeps this O(n_groups * max_k) rather than O(n_districts * max_k).
        for g0, g0_dists in group_districts.items():
            if len(g0_dists) < 2:
                continue
            cg0 = centroid[g0]

            # Most outlying unlocked district in this group (area-weighted inertia).
            d0 = max(
                (d for d in g0_dists if slot_of[d] not in locked),
                key=lambda d: cell_area[slot_of[d]] * float(np.sum((cell_xy[slot_of[d]] - cg0) ** 2)),
                default=None,
            )
            if d0 is None:
                continue
            s0 = slot_of[d0]
            r_sq0 = cell_area[s0] * float(np.sum((cell_xy[s0] - cg0) ** 2))
            if r_sq0 == 0.0:
                continue

            for candidate in _enum_paths({s0}, locked, min_len=3, max_len=5, max_k=20):
                n_cand = len(candidate)

                # The intermediate slots (indices 1..n_cand-2) are displaced.
                # Skip chains where any displaced intermediate district belongs
                # to g0 — that would shuffle districts within the same group
                # without crossing a group boundary, which is incorrect for
                # per-district cartograms (e.g. NY-1 shown in NY-2's cell).
                if any(groups[dist_at[candidate[i]]] == g0 for i in range(1, n_cand - 1)):
                    continue

                # The outlier moves to candidate[n_cand - 2].  Skip if no closer (area-weighted).
                s_final = candidate[n_cand - 2]
                if cell_area[s_final] * float(np.sum((cell_xy[s_final] - centroid[g0]) ** 2)) >= r_sq0:
                    continue

                # Compute final slot for each displaced district.
                affected_final: dict[int, int] = {}
                for i in range(n_cand - 1):
                    d = dist_at[candidate[i]]
                    affected_final[d] = candidate[i - 1] if i > 0 else candidate[n_cand - 2]

                # Reject if any currently-contiguous group would be split.
                bad = False
                checked_grps: set = set()
                for i in range(n_cand - 1):
                    grp_b = groups[dist_at[candidate[i]]]
                    if grp_b in checked_grps:
                        continue
                    checked_grps.add(grp_b)
                    grp_b_dists = group_districts[grp_b]
                    if len(grp_b_dists) < 2:
                        continue
                    cur_slots_b = [slot_of[dd] for dd in grp_b_dists]
                    if len(_connected_components(cur_slots_b)) > 1:
                        continue
                    fin_slots_b = [affected_final.get(dd, slot_of[dd]) for dd in grp_b_dists]
                    if len(_connected_components(fin_slots_b)) > 1:
                        bad = True
                        break
                if bad:
                    continue

                # Compute normalised metric change across all affected groups.
                affected_groups = {groups[d] for d in affected_final}
                delta_I = 0.0
                delta_S = 0.0
                I_0_total = 0.0
                S_0_total = 0.0
                for g in affected_groups:
                    g_dists = group_districts[g]
                    cur_slots_g = [slot_of[dd] for dd in g_dists]
                    fin_slots_g = [affected_final.get(dd, slot_of[dd]) for dd in g_dists]
                    cg = centroid[g]
                    w_ag = cell_area[cur_slots_g]
                    w_ag = w_ag / w_ag.sum() if w_ag.sum() > 0 else np.ones(len(cur_slots_g)) / len(cur_slots_g)
                    w_ag_fin = cell_area[fin_slots_g]
                    w_ag_fin = (
                        w_ag_fin / w_ag_fin.sum()
                        if w_ag_fin.sum() > 0
                        else np.ones(len(fin_slots_g)) / len(fin_slots_g)
                    )
                    i_cur = float(np.sum(w_ag * np.sum((cell_xy[cur_slots_g] - cg) ** 2, axis=1)))
                    i_fin = float(np.sum(w_ag_fin * np.sum((cell_xy[fin_slots_g] - cg) ** 2, axis=1)))
                    delta_I += i_fin - i_cur
                    I_0_total += i_cur
                    cur_set = set(cur_slots_g)
                    fin_set = set(fin_slots_g)
                    s_cur_g = sum(_elen(a, b) for a in cur_set for b in adj[a] if b in cur_set and b > a)
                    s_fin_g = sum(_elen(a, b) for a in fin_set for b in adj[a] if b in fin_set and b > a)
                    delta_S += s_fin_g - s_cur_g
                    S_0_total += s_cur_g

                if I_0_total == 0.0:
                    continue
                criterion = delta_I / I_0_total - (delta_S / S_0_total if S_0_total > 0.0 else 0.0)
                if criterion >= 0.0:
                    continue

                # Execute the chain.
                for k in range(1, n_cand - 1):
                    d_mv = dist_at[candidate[k - 1]]
                    d_dp = dist_at[candidate[k]]
                    sa, sb = slot_of[d_mv], slot_of[d_dp]
                    slot_of[d_mv], slot_of[d_dp] = sb, sa
                    dist_at[sa], dist_at[sb] = d_dp, d_mv
                locked.update(candidate[:-1])
                improved = True
                break  # next outlier

        if not improved:
            break

    return slot_of


def _swap_preserves_contiguity(
    d1: int,
    d2: int,
    groups: list,
    adj_dict: dict,
) -> bool:
    """Return True if swapping districts d1 and d2 keeps both groups connected.

    Parameters
    ----------
    d1, d2 : int
        District indices to swap.
    groups : list
        Current group label for each district (index = district, value = group).
    adj_dict : dict[int, set[int]]
        Voronoi adjacency as an adjacency-list dict (district-indexed after
        applying the current slot_of permutation).
    """
    g1, g2 = groups[d1], groups[d2]
    if g1 == g2:
        return True  # same group: swap can't split it

    for target_d, source_d, grp in ((d1, d2, g1), (d2, d1, g2)):
        members = {i for i, g in enumerate(groups) if g == grp}
        new_members = (members - {target_d}) | {source_d}
        if len(new_members) <= 1:
            continue  # singleton: trivially connected
        start = next(iter(new_members))
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj_dict.get(node, ()):
                if nb in new_members and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if visited != new_members:
            return False
    return True


def _repair_adjacency(
    cells: list,
    adj_pairs: list[tuple[int, int]],
    *,
    max_passes: int = 3,
    groups: list | None = None,
    min_shared_length: float | None = None,
) -> np.ndarray:
    """Permute slots so that input-adjacent geometry pairs have adjacent Voronoi cells.

    A swap of districts *a* and *b* is applied only when the net count of
    newly-satisfied adj_pairs minus newly-violated adj_pairs is strictly positive.
    This guarantees monotonic improvement and avoids oscillation (e.g. gaining
    one adjacency while losing another yields net 0 → skip).

    Parameters
    ----------
    cells : list
        Shapely geometry objects — the Voronoi cells, one per slot (row index).
    adj_pairs : list of (i, j)
        Pairs of district indices that are adjacent in the input geometries.
    max_passes : int
        Maximum repair passes.

    Returns
    -------
    slot_of : np.ndarray[int], shape (n,)
        Permutation array.  Identity if no improvement was found.
    """
    from carto_flow.geo_utils.adjacency import find_adjacent_pairs

    n = len(cells)
    slot_of = np.arange(n, dtype=np.intp)
    dist_at = list(range(n))

    if not adj_pairs:
        return slot_of

    # Build Voronoi adjacency set and dict (for contiguity guard)
    raw = find_adjacent_pairs(cells, min_shared_length=min_shared_length)
    voronoi_adj: set[tuple[int, int]] = set()
    voronoi_adj_dict: dict[int, set[int]] = defaultdict(set)
    for s1, s2, _ in raw:
        voronoi_adj.add((min(s1, s2), max(s1, s2)))
        voronoi_adj_dict[s1].add(s2)
        voronoi_adj_dict[s2].add(s1)

    def _v_adj(s1: int, s2: int) -> bool:
        return (min(s1, s2), max(s1, s2)) in voronoi_adj

    # adj_pair index lists per district
    district_pairs: defaultdict[int, list[int]] = defaultdict(list)
    for k, (i, j) in enumerate(adj_pairs):
        district_pairs[i].append(k)
        district_pairs[j].append(k)

    def _net_gain_swap(a: int, b: int) -> int:
        sa, sb = slot_of[a], slot_of[b]
        gain = 0
        seen: set[int] = set()
        for d in (a, b):
            for k in district_pairs[d]:
                if k in seen:
                    continue
                seen.add(k)
                pi, pj = adj_pairs[k]
                si, sj = slot_of[pi], slot_of[pj]
                # After swap: slot_of[a]↔slot_of[b]
                new_si = sb if pi == a else (sa if pi == b else si)
                new_sj = sb if pj == a else (sa if pj == b else sj)
                before = _v_adj(si, sj)
                after = _v_adj(new_si, new_sj)
                gain += int(after) - int(before)
        return gain

    def _do_swap(a: int, b: int) -> None:
        sa, sb = slot_of[a], slot_of[b]
        slot_of[a], slot_of[b] = sb, sa
        dist_at[sa], dist_at[sb] = b, a

    # Working copy of groups (district-indexed) for contiguity guard
    cur_groups = list(groups) if groups is not None else None

    for _ in range(max_passes):
        violated_districts: set[int] = set()
        for i, j in adj_pairs:
            if not _v_adj(slot_of[i], slot_of[j]):
                violated_districts.add(i)
                violated_districts.add(j)
        if not violated_districts:
            break

        locked: set[int] = set()
        improved = False
        for d in list(violated_districts):
            if d in locked:
                continue
            best_gain, best_b = 0, -1
            for b in range(n):
                if b == d or b in locked:
                    continue
                gain = _net_gain_swap(d, b)
                if gain > best_gain:
                    if cur_groups is not None and not _swap_preserves_contiguity(d, b, cur_groups, voronoi_adj_dict):
                        continue
                    best_gain, best_b = gain, b
            if best_b >= 0:
                _do_swap(d, best_b)
                if cur_groups is not None:
                    cur_groups[d], cur_groups[best_b] = cur_groups[best_b], cur_groups[d]
                locked.add(d)
                locked.add(best_b)
                improved = True
        if not improved:
            break

    return slot_of


def _repair_orientation(
    cells: list,
    geom_positions: np.ndarray,
    adj_pairs: list[tuple[int, int]],
    slot_of: np.ndarray,
    *,
    max_passes: int = 3,
    groups: list | None = None,
    min_shared_length: float | None = None,
) -> np.ndarray:
    """Permute slots to align output cell directions with input geometry directions.

    For each pair in *adj_pairs* that is currently Voronoi-adjacent, computes the
    cosine similarity between the input direction vector (geom j - geom i) and the
    output direction vector (cell[slot_of[j]].centroid - cell[slot_of[i]].centroid).
    Swaps are applied only when (1) the net adjacency count does not decrease and
    (2) the total orientation score (sum of cosines for adjacent pairs) strictly
    improves.

    Parameters
    ----------
    cells : list
        Shapely geometry objects — the Voronoi cells, one per slot.
    geom_positions : np.ndarray, shape (G, 2)
        Input geometry centroids (x, y), district-indexed.
    adj_pairs : list of (i, j)
        Pairs of district indices adjacent in the input geometries.
    slot_of : np.ndarray[int], shape (G,)
        Current permutation array (district → slot), typically from
        ``_repair_adjacency()``.  Modified in-place and returned.
    max_passes : int
        Maximum repair passes.

    Returns
    -------
    slot_of : np.ndarray[int], shape (G,)
        Updated permutation array.
    """
    from carto_flow.geo_utils.adjacency import find_adjacent_pairs

    n = len(cells)
    dist_at = list(np.argsort(slot_of))  # slot → district (inverse of slot_of)
    # Recompute dist_at from slot_of (slot_of may differ from identity)
    dist_at = [-1] * n
    for d, s in enumerate(slot_of):
        dist_at[s] = d

    if not adj_pairs:
        return slot_of

    # Build Voronoi adjacency set and dict (for contiguity guard)
    raw = find_adjacent_pairs(cells, min_shared_length=min_shared_length)
    voronoi_adj: set[tuple[int, int]] = set()
    voronoi_adj_dict: dict[int, set[int]] = defaultdict(set)
    for s1, s2, _ in raw:
        voronoi_adj.add((min(s1, s2), max(s1, s2)))
        voronoi_adj_dict[s1].add(s2)
        voronoi_adj_dict[s2].add(s1)

    def _v_adj(s1: int, s2: int) -> bool:
        return (min(s1, s2), max(s1, s2)) in voronoi_adj

    # Precompute cell centroids (slot-indexed) and input direction vectors
    cell_xy = np.array([[c.centroid.x, c.centroid.y] for c in cells])
    adj_i = np.array([i for i, j in adj_pairs], dtype=np.intp)
    adj_j = np.array([j for i, j in adj_pairs], dtype=np.intp)
    d_in = geom_positions[adj_j] - geom_positions[adj_i]  # fixed

    # adj_pair index lists per district
    district_pairs: defaultdict[int, list[int]] = defaultdict(list)
    for k, (i, j) in enumerate(adj_pairs):
        district_pairs[i].append(k)
        district_pairs[j].append(k)

    def _cosine(s_i: int, s_j: int, k: int) -> float:
        d_out = cell_xy[s_j] - cell_xy[s_i]
        n_in = float(np.linalg.norm(d_in[k]))
        n_out = float(np.linalg.norm(d_out))
        if n_in < 1e-10 or n_out < 1e-10:
            return 1.0  # degenerate → treat as satisfied
        return float(np.dot(d_in[k], d_out) / (n_in * n_out))

    def _current_score() -> float:
        total = 0.0
        for k, (pi, pj) in enumerate(adj_pairs):
            si, sj = slot_of[pi], slot_of[pj]
            if _v_adj(si, sj):
                total += _cosine(si, sj, k)
        return total

    def _delta_swap(a: int, b: int):
        """Return (adjacency_net_gain, orientation_delta) for swapping a and b."""
        sa, sb = slot_of[a], slot_of[b]
        adj_gain = 0
        ori_delta = 0.0
        seen: set[int] = set()
        for d in (a, b):
            for k in district_pairs[d]:
                if k in seen:
                    continue
                seen.add(k)
                pi, pj = adj_pairs[k]
                si, sj = slot_of[pi], slot_of[pj]
                new_si = sb if pi == a else (sa if pi == b else si)
                new_sj = sb if pj == a else (sa if pj == b else sj)
                was_adj = _v_adj(si, sj)
                now_adj = _v_adj(new_si, new_sj)
                adj_gain += int(now_adj) - int(was_adj)
                if was_adj:
                    ori_delta -= _cosine(si, sj, k)
                if now_adj:
                    ori_delta += _cosine(new_si, new_sj, k)
        return adj_gain, ori_delta

    def _do_swap(a: int, b: int) -> None:
        sa, sb = slot_of[a], slot_of[b]
        slot_of[a], slot_of[b] = sb, sa
        dist_at[sa], dist_at[sb] = b, a

    # Working copy of groups for contiguity guard
    cur_groups = list(groups) if groups is not None else None

    for _ in range(max_passes):
        # Find pairs with negative cosine that are currently adjacent
        violated: set[int] = set()
        for k, (pi, pj) in enumerate(adj_pairs):
            si, sj = slot_of[pi], slot_of[pj]
            if _v_adj(si, sj) and _cosine(si, sj, k) <= 0:
                violated.add(pi)
                violated.add(pj)
        if not violated:
            break

        locked: set[int] = set()
        improved = False
        for d in list(violated):
            if d in locked:
                continue
            best_ori, best_b = 0.0, -1
            for b in range(n):
                if b == d or b in locked:
                    continue
                adj_gain, ori_delta = _delta_swap(d, b)
                if adj_gain >= 0 and ori_delta > best_ori:
                    if cur_groups is not None and not _swap_preserves_contiguity(d, b, cur_groups, voronoi_adj_dict):
                        continue
                    best_ori, best_b = ori_delta, b
            if best_b >= 0:
                _do_swap(d, best_b)
                if cur_groups is not None:
                    cur_groups[d], cur_groups[best_b] = cur_groups[best_b], cur_groups[d]
                locked.add(d)
                locked.add(best_b)
                improved = True
        if not improved:
            break

    return slot_of


def make_groups_contiguous(
    gdf: gpd.GeoDataFrame,
    group_by: str,
    *,
    max_passes: int = 20,
    show_progress: bool = False,
    debug: bool = False,
) -> tuple[gpd.GeoDataFrame, list[tuple[Any, list[int]]]]:
    """Permute rows of *gdf* so that each group forms a contiguous Voronoi region.

    After lloyd relaxation, rows belonging to the same group (e.g. congressional
    districts within a state) may occupy non-adjacent Voronoi slots.  This
    function permutes the row-to-slot mapping — without moving any cell geometry
    — so that each group's slots form a connected subgraph of the Voronoi
    adjacency graph.

    The returned GeoDataFrame has the same rows as *gdf* but reordered:
    ``result.iloc[i]`` is the row that should be placed at slot *i* (i.e. the
    geometry at ``gdf.geometry.iloc[i]`` is now owned by a potentially different
    original row).

    Algorithm
    ---------
    1. Build a slot-adjacency graph from ``gdf.geometry`` using
       :func:`carto_flow.geo_utils.adjacency.find_adjacent_pairs`.
    2. Maintain a permutation ``slot_of[d]`` (district → slot) and its inverse
       ``dist_at[s]`` (slot → district).
    3. Repeat up to *max_passes* times:

       a. For each multi-district group, find connected components of its
          current slots (BFS restricted to that group's slots).
       b. For each satellite component (all but the largest), BFS through
          other-group slots to find the shortest path to the main body.
          Slots that changed hands earlier in the same pass are forbidden,
          preventing within-pass oscillation.
       c. Walk the path: swap each intermediate district with the one moving
          forward along the path.  Before each swap, check that the swap would
          not disconnect the displaced district's group (articulation-point
          check).  If it would, skip this path.
       d. Stop when all groups are contiguous or no swaps were made in a pass.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame whose ``geometry`` column contains the Voronoi cells (one
        cell per row).  All other columns are carried along unchanged.
    group_by : str
        Column in *gdf* that identifies which group each row belongs to.
    max_passes : int
        Maximum number of repair passes (default 20).
    show_progress : bool
        If True, print a summary of discontiguous groups before the first pass
        and a status line after each pass.
    debug : bool
        If True, print one line per satellite repair attempt showing the group,
        satellite slots, and outcome (SWAPPED / REJECTED / NO PATH).

    Returns
    -------
    GeoDataFrame
        Copy of *gdf* with only the geometry column permuted.  Row *i* of the
        result keeps all original attribute values but is assigned the Voronoi
        cell geometry that was determined to belong to it.
    list[tuple[Any, list[int]]]
        One entry per remaining satellite component: ``(group_id, row_indices)``
        where *group_id* is the value from the *group_by* column and *row_indices*
        is the sorted list of original gdf row indices belonging to that satellite.
        The main body of each group is excluded.  Empty when fully converged.
    """
    cells = list(gdf.geometry)
    groups = list(gdf[group_by])
    slot_of, discontiguous = _repair_contiguity(
        cells,
        groups,
        max_passes=max_passes,
        show_progress=show_progress,
        debug=debug,
    )
    result = gdf.copy()
    result = result.set_geometry([cells[s] for s in slot_of])
    return result, discontiguous


def _compose_topology_permutation(
    cells: list,
    groups: list | None,
    adj_pairs: list[tuple[int, int]] | None,
    *,
    group_contiguity: bool | None = None,
    compactness: bool | None = None,
    adjacency: bool | None = None,
    orientation: bool | None = None,
    max_passes: int = 3,
    geom_positions: np.ndarray | None = None,
    min_shared_length: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Run topology repair stages and return the composed permutation.

    This is the shared core used by both the online ``api.py`` pipeline and
    the post-hoc :meth:`~carto_flow.voronoi_cartogram.result.VoronoiCartogram.repair_topology`
    method.

    Parameters
    ----------
    cells : list
        Voronoi cell geometries (one per district/slot).
    groups : list or None
        Group label for each district.  Required for stage 1; ignored when
        ``group_contiguity=False`` or ``None``.
    adj_pairs : list of (i, j) or None
        Input-adjacent district index pairs.  Required for stages 2 & 3;
        those stages are skipped when ``None``.
    group_contiguity : bool
        Enable stage 1 (group contiguity repair).
    compactness : bool or None
        Enable stage 1.5 (compactness enhancement).  ``None`` → ``True`` when
        *groups* is not ``None``, ``False`` otherwise.
    adjacency : bool or None
        Enable stage 2 (adjacency permutation).  ``None`` → ``True`` when
        *groups* is ``None``, ``False`` otherwise.
    orientation : bool or None
        Enable stage 3 (orientation alignment).  ``None`` → ``True`` when
        *groups* is ``None``, ``False`` otherwise.
    max_passes : int
        Maximum repair passes per active stage.
    geom_positions : np.ndarray of shape (n, 2) or None
        Input geometry centroids (x, y), district-indexed.  Required for
        stage 3; stage 3 is skipped when ``None``.

    Returns
    -------
    slot_of : np.ndarray[int], shape (n,)
        Composed permutation across all active stages.
        ``slot_of[d]`` is the original slot whose cell district *d* should
        now occupy.  Identity array when no stages were run or no swaps
        were made.
    stages_run : list[str]
        Names of stages that were executed (subset of
        ``["group_contiguity", "compactness", "adjacency", "orientation"]``).
    """
    # Auto-resolve stage flags
    if group_contiguity is None:
        group_contiguity = groups is not None
    if compactness is None:
        compactness = groups is not None
    if adjacency is None:
        adjacency = groups is None
    if orientation is None:
        orientation = groups is None

    n = len(cells)
    slot_of = np.arange(n, dtype=np.intp)
    stages_run: list[str] = []

    # Stage 1: group contiguity
    if group_contiguity and groups is not None:
        _s, _ = _repair_contiguity(cells, groups, max_passes=max_passes, min_shared_length=min_shared_length)
        slot_of = slot_of[_s]
        cells = [cells[s] for s in _s]
        stages_run.append("group_contiguity")

    # Stage 1.5: compactness enhancement — boundary swaps between adjacent groups
    # to reduce inertia without introducing satellites.  Runs on the already-permuted
    # (district-indexed) cells produced by stage 1.
    if compactness and groups is not None:
        _s = _repair_compactness(cells, groups, max_passes=max_passes, min_shared_length=min_shared_length)
        slot_of = slot_of[_s]
        cells = [cells[s] for s in _s]
        stages_run.append("compactness")

    # Stage 2: adjacency
    if adjacency and adj_pairs:
        _s = _repair_adjacency(
            cells, adj_pairs, max_passes=max_passes, groups=groups, min_shared_length=min_shared_length
        )
        slot_of = slot_of[_s]
        cells = [cells[s] for s in _s]
        stages_run.append("adjacency")

    # Stage 3: orientation
    if orientation and adj_pairs and geom_positions is not None:
        _s = _repair_orientation(
            cells,
            geom_positions,
            adj_pairs,
            np.arange(len(cells), dtype=np.intp),
            max_passes=max_passes,
            groups=groups,
            min_shared_length=min_shared_length,
        )
        slot_of = slot_of[_s]
        stages_run.append("orientation")

    return slot_of, stages_run
