"""Distance reporting using the already accepted cuRobo-map collision model."""
import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
path = REPO / "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp/umi_sphere_geometry.py"
spec = importlib.util.spec_from_file_location("thunder_map_geometry", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
UmiSphereGeometry = module.UmiSphereGeometry


def clearances(adapter, positions, rotations, origins):
    """Report signed separation and remaining map clearance, including all links."""
    output = {key: [] for key in ("self_margin_m", "world_margin_m", "world_gap_m")}
    for start in range(0, len(positions), 128):
        sl = slice(start, start+128)
        centers = (rotations[sl][:, adapter.body_ids] @ adapter.centers[None, :, :, None]).squeeze(-1)
        centers += positions[sl][:, adapter.body_ids] - origins[sl, None, :]
        self_gap = (centers[:, adapter.pair_a]-centers[:, adapter.pair_b]).norm(dim=-1)-adapter.pair_radius
        delta = centers[:, adapter.world_ids, None]-adapter.box_p
        local = torch.einsum("bski,kij->bskj", delta, adapter.box_r)
        d = local.abs()-adapter.box_half
        signed = d.clamp_min(0).norm(dim=-1)+d.amax(dim=-1).clamp_max(0)
        gap = signed-adapter.radii[adapter.world_ids][None, :, None]
        output["self_margin_m"].append(self_gap.amin(dim=1))
        output["world_margin_m"].append((gap-adapter.world_padding).amin(dim=(1, 2)))
        output["world_gap_m"].append((gap+0.001).amin(dim=(1, 2)))
    return {key: torch.cat(value) for key, value in output.items()}
