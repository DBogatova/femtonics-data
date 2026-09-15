#!/usr/bin/env python
"""
segment_skeleton_napari.py - Segment a dendrite mask by CLICKING, not drawing.

The mask is skeletonized and automatically broken into many small pieces (one per
inter-bifurcation branch). Each piece is inflated back to fill the mask (every mask
voxel is assigned to its geodesically nearest piece, measured THROUGH the mask, so
branches that pass close in space but are far along the dendrite never bleed together).

You then just CLICK pieces in the 3D view to group them:
  * every piece starts as its own distinct part (already a valid segmentation);
  * pick a segment number, then click the pieces that belong to that segment to merge
    them; pieces you never click stay separate.

This avoids both hard 3D drawing and depth-ambiguous 2D painting.

Keys
----
  1..9      : set the current segment number (what a click assigns)
  m         : start a NEW segment number (max+1)
  click     : assign the clicked piece to the current segment (rotate = drag, no assign)
  u         : undo the last click
  r         : reset ALL groupings (back to one-piece-per-part)
  [ / ]     : coarser / finer decomposition (fewer / more pieces); resets grouping
  n         : renumber/preview -> recolor by final contiguous segment ids
  s         : save -> *_segments_labelmap.tif  (grouped pieces share a label; ungrouped
              pieces each keep their own; labels renumbered 1..N, clamped to the mask)

Usage
-----
  python code/STEP4_segments/segment_skeleton_napari.py run_clean.tif run_labelmap.tif \
      --voxel 0.8 0.9 0.9 [--min-branch 4] [--agg max] [--ndisplay 3]
Then:
  python code/STEP5_traces/extract_segment_traces.py run_clean.tif *_segments_labelmap.tif --voxel 0.8 0.9 0.9
"""
import argparse
import numpy as np
import tifffile


def decompose(mask, voxel, min_branch=4):
    """Skeletonize -> split at bifurcations into pieces -> inflate each piece to fill the
    mask by geodesic (through-mask, anisotropic) nearest-seed. Returns uint labelmap of
    pieces (1..K), same shape as mask; 0 outside the mask."""
    from skimage.morphology import skeletonize
    from scipy.ndimage import convolve, label as cc_label
    from skimage.graph import MCP_Geometric

    skel = skeletonize(mask)
    if not skel.any():
        raise SystemExit("skeleton is empty - is the mask non-empty?")
    # neighbour count on the skeleton (26-connectivity); degree>=3 -> bifurcation
    nb = convolve(skel.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant") - skel
    nodes = skel & (nb * skel >= 3)
    edges = skel & ~nodes                                  # cut junctions -> disjoint branch pieces
    piece_skel, npc = cc_label(edges, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(piece_skel.ravel())
    keep = [i for i in range(1, npc + 1) if sizes[i] >= min_branch]  # drop short spurs
    if not keep:                                           # mask too small to prune - keep all
        keep = [i for i in range(1, npc + 1)]
    ps = np.zeros_like(piece_skel, np.int32)
    for new, old in enumerate(keep, start=1):
        ps[piece_skel == old] = new
    K = len(keep)

    costs = np.where(mask, 1.0, np.inf).astype(float)
    best = np.full(mask.shape, np.inf)
    pieces = np.zeros(mask.shape, np.uint16)
    for l in range(1, K + 1):
        mcp = MCP_Geometric(costs, sampling=tuple(voxel))
        cc, _ = mcp.find_costs([tuple(p) for p in np.argwhere(ps == l)])
        take = cc < best
        best[take] = cc[take]
        pieces[take] = l
    pieces[~mask] = 0
    pieces[np.isinf(best)] = 0                             # unreachable (disconnected, no seed)
    return pieces, K


def final_labelmap(pieces, K, group_of):
    """Map pieces -> contiguous segment ids. Pieces sharing a positive group id merge;
    ungrouped pieces (group 0) each become their own segment. Returns uint16 labelmap."""
    # build a per-piece 'key': grouped pieces share ('g', gid); ungrouped are unique ('p', pid)
    keys = {}
    for pid in range(1, K + 1):
        g = int(group_of[pid])
        keys[pid] = ("g", g) if g > 0 else ("p", pid)
    # order segments: grouped ids first (by gid), then ungrouped (by piece id) - deterministic
    uniq = sorted(set(keys.values()), key=lambda k: (k[0] != "g", k[1]))
    remap = {k: i + 1 for i, k in enumerate(uniq)}
    lut = np.zeros(K + 1, np.uint16)
    for pid in range(1, K + 1):
        lut[pid] = remap[keys[pid]]
    return lut[pieces]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack")
    ap.add_argument("labelmap")
    ap.add_argument("--voxel", nargs=3, type=float, default=[1.0, 1.0, 1.0], metavar=("Z", "Y", "X"))
    ap.add_argument("--min-branch", type=int, default=4,
                    help="drop skeleton pieces shorter than this many voxels (absorbed into a neighbour)")
    ap.add_argument("--agg", choices=["max", "mean"], default="max",
                    help="temporal projection for the anatomy background (max matches refine)")
    ap.add_argument("--ndisplay", type=int, default=3, choices=[2, 3],
                    help="3 = rotatable 3D (recommended for clicking branches); 2 = slice view")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import napari

    stack = tifffile.imread(args.stack)
    mask = tifffile.imread(args.labelmap) > 0
    bgvol = (stack.max(0) if args.agg == "max" else stack.mean(0)).astype(np.float32) \
        if stack.ndim == 4 else stack.astype(np.float32)
    vox = tuple(args.voxel)
    out = args.out or args.labelmap.rsplit(".", 1)[0].replace("_labelmap", "") + "_segments_labelmap.tif"

    print(f"[1/2] decomposing mask ({int(mask.sum())} voxels) into skeleton pieces ...", flush=True)
    pieces, K = decompose(mask, vox, args.min_branch)
    print(f"[2/2] {K} pieces; mask coverage {100*(pieces[mask]>0).mean():.1f}%. Opening napari ...",
          flush=True)

    # mutable state (K/pieces change when the granularity is re-dialled live)
    S = {"pieces": pieces, "K": K, "group_of": np.zeros(K + 1, np.int32),
         "cur": 1, "history": [], "min_branch": args.min_branch}
    OFF = 1000                                # ungrouped pieces get distinct display colours

    def display_array():
        K = S["K"]
        lut = np.zeros(K + 1, np.uint32)
        for pid in range(1, K + 1):
            g = S["group_of"][pid]
            lut[pid] = g if g > 0 else OFF + pid
        return lut[S["pieces"]]

    v = napari.Viewer(ndisplay=args.ndisplay)
    clim = (float(np.percentile(bgvol, 2)), float(np.percentile(bgvol, 99.7)))
    v.add_image(bgvol, name="anatomy", colormap="gray", scale=vox,
                contrast_limits=clim, rendering="attenuated_mip", opacity=0.5)
    pieces_layer = v.add_labels(S["pieces"], name="pieces (pick source)", scale=vox, visible=False)
    seg_layer = v.add_labels(display_array(), name="segments (click to group)",
                             scale=vox, opacity=0.7)
    seg_layer.mode = "pan_zoom"                              # drag = rotate; we detect clicks below
    v.scale_bar.visible = True
    v.scale_bar.unit = "um"

    def refresh():
        seg_layer.data = display_array()

    def recompute(delta):
        """Re-decompose at a coarser (+) / finer (-) granularity; resets any grouping."""
        mb = max(2, S["min_branch"] + delta)
        S["min_branch"] = mb
        pcs, k = decompose(mask, vox, mb)
        S["pieces"], S["K"] = pcs, k
        S["group_of"] = np.zeros(k + 1, np.int32)
        S["history"].clear(); S["cur"] = 1
        pieces_layer.data = pcs
        refresh()
        print(f"re-decomposed: min-branch={mb} -> {k} pieces (grouping reset)", flush=True)

    def piece_at(event):
        """Piece id under the cursor via the (hidden) pieces layer's ray pick."""
        try:
            val = pieces_layer.get_value(
                event.position,
                view_direction=getattr(event, "view_direction", None),
                dims_displayed=getattr(event, "dims_displayed", None),
                world=True,
            )
        except Exception:
            val = None
        return int(val) if val else 0

    @seg_layer.mouse_drag_callbacks.append
    def on_click(layer, event):
        dragged = False
        yield
        while event.type == "mouse_move":                   # rotating / panning, not a click
            dragged = True
            yield
        if dragged:
            return
        pid = piece_at(event)
        if pid <= 0 or pid > S["K"]:
            return
        S["history"].append((pid, int(S["group_of"][pid])))
        S["group_of"][pid] = S["cur"]
        refresh()
        n_in = int((S["group_of"] == S["cur"]).sum())
        print(f"piece {pid} -> segment {S['cur']}  ({n_in} piece(s) in segment {S['cur']})", flush=True)

    def set_group(g):
        S["cur"] = g
        print(f"current segment = {g} (click pieces to add them here; 'm' = new segment)", flush=True)

    for d in range(1, 10):
        v.bind_key(str(d), lambda vw, d=d: set_group(d), overwrite=True)

    @v.bind_key("m", overwrite=True)
    def _new(vw):
        set_group(int(S["group_of"].max()) + 1 if S["group_of"].max() > 0 else 1)

    @v.bind_key("u", overwrite=True)
    def _undo(vw):
        if not S["history"]:
            print("nothing to undo", flush=True); return
        pid, prev = S["history"].pop()
        S["group_of"][pid] = prev
        refresh()
        print(f"undo: piece {pid} -> segment {prev if prev else 'ungrouped'}", flush=True)

    @v.bind_key("r", overwrite=True)
    def _reset(vw):
        S["group_of"][:] = 0
        S["history"].clear()
        refresh()
        print("reset: every piece is its own part again", flush=True)

    v.bind_key("BracketRight", lambda vw: recompute(-1), overwrite=True)   # ] finer (more pieces)
    v.bind_key("BracketLeft", lambda vw: recompute(+1), overwrite=True)    # [ coarser (fewer pieces)

    @v.bind_key("n", overwrite=True)
    def _preview(vw):
        fm = final_labelmap(S["pieces"], S["K"], S["group_of"])
        seg_layer.data = fm.astype(np.uint32)
        ids = sorted(int(i) for i in np.unique(fm) if i > 0)
        print(f"preview final segmentation: {len(ids)} segments {ids} "
              "(press 's' to save, any click/number to keep editing)", flush=True)

    @v.bind_key("s", overwrite=True)
    def _save(vw):
        fm = final_labelmap(S["pieces"], S["K"], S["group_of"]).astype(np.uint16)
        fm[~mask] = 0
        tifffile.imwrite(out, fm)
        ids = [int(i) for i in np.unique(fm) if i > 0]
        vx = {i: int((fm == i).sum()) for i in ids}
        print(f"saved -> {out}  segments={ids}  voxels={vx}", flush=True)

    print(f"{K} auto pieces. KEYS: 1..9 = pick segment #, m = new segment, click a branch to add "
          "it, u = undo, r = reset, [ / ] = coarser/finer pieces, n = preview final ids, s = save. "
          "Drag = rotate.")
    napari.run()


if __name__ == "__main__":
    main()

