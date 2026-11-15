"""Frame conventions and differentiable-free geometry for HGO (docs/METHOD.md).

Everything a model sees is in the t0 ego (rear-axle) frame, NAVSIM's planning frame: x forward, y left, yaw CCW.
OpenScene boxes are stored in each frame's LiDAR frame; `lidar2ego` / `ego2global` / `lidar2global` come from the logs.
"""

import numpy as np
import torch

CAMS = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0"]
# detection / slot vocabulary (gt_names in OpenScene annotations)
CLASS_OF = {"vehicle": 0, "pedestrian": 1, "bicycle": 2,
            "traffic_cone": 3, "barrier": 3, "czone_sign": 3, "generic_object": 3}
N_CLS = 4  # vehicle, pedestrian, bicycle, static obstacle
# NAVSIM ego vehicle (Chrysler Pacifica, navsim.common.dataclasses / nuPlan get_pacifica_parameters)
EGO_LENGTH, EGO_WIDTH, EGO_REAR_TO_CENTER = 5.176, 2.297, 1.461


def yaw_of(T):
    return np.arctan2(T[1, 0], T[0, 0])


def to_frame(T, boxes, vel=None):
    """Apply a 4x4 transform to boxes (N,7: x,y,z,l,w,h,yaw) and planar velocities (N,2)."""
    out = boxes.copy()
    out[:, :3] = boxes[:, :3] @ T[:3, :3].T + T[:3, 3]
    out[:, 6] = boxes[:, 6] + yaw_of(T)
    if vel is None:
        return out
    return out, vel @ T[:2, :2].T


def scale_intrinsics(K, orig_wh, new_wh):
    K = np.array(K, np.float64).copy()
    K[0, 0] *= new_wh[0] / orig_wh[0]
    K[0, 2] *= new_wh[0] / orig_wh[0]
    K[1, 1] *= new_wh[1] / orig_wh[1]
    K[1, 2] *= new_wh[1] / orig_wh[1]
    return K


def box_corners_t(box):
    """(..., 7) x,y,z,l,w,h,yaw -> (..., 8, 3) corners (torch)."""
    x, y, z, l, w, h, yaw = box.unbind(-1)
    sx = box.new_tensor([1, 1, 1, 1, -1, -1, -1, -1]) * 0.5
    sy = box.new_tensor([1, -1, -1, 1, 1, -1, -1, 1]) * 0.5
    sz = box.new_tensor([1, 1, -1, -1, 1, 1, -1, -1]) * 0.5
    dx, dy, dz = l[..., None] * sx, w[..., None] * sy, h[..., None] * sz
    c, s = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    return torch.stack([x[..., None] + c * dx - s * dy, y[..., None] + s * dx + c * dy, z[..., None] + dz], -1)


def project_boxes_t(box, cam2ego, K, img_wh, min_side=4.0):
    """Project ego-frame 3D boxes into every camera.

    box (B,N,7); cam2ego (B,C,4,4); K (B,C,3,3) already scaled to img_wh.
    Returns rois (B,N,C,4) clipped x1,y1,x2,y2 in pixels, area (B,N,C) (0 where not visible).
    """
    corners = box_corners_t(box)  # B,N,8,3
    ego2cam = torch.linalg.inv(cam2ego)  # B,C,4,4
    R, t = ego2cam[..., :3, :3], ego2cam[..., :3, 3]
    pc = torch.einsum("bcij,bnkj->bncki", R, corners) + t[:, None, :, None, :]  # B,N,C,8,3
    centre = torch.einsum("bcij,bnj->bnci", R, box[..., :3]) + t[:, None]
    z = pc[..., 2]
    front = z > 0.1
    zc = torch.where(front, z, torch.ones_like(z))
    Kb = K[:, None, :, None]  # B,1,C,1,3,3
    u = (Kb[..., 0, 0] * pc[..., 0] + Kb[..., 0, 1] * pc[..., 1]) / zc + Kb[..., 0, 2]
    v = Kb[..., 1, 1] * pc[..., 1] / zc + Kb[..., 1, 2]
    big = 1e6
    x1 = torch.where(front, u, torch.full_like(u, big)).amin(-1).clamp(0, img_wh[0])
    y1 = torch.where(front, v, torch.full_like(v, big)).amin(-1).clamp(0, img_wh[1])
    x2 = torch.where(front, u, torch.full_like(u, -big)).amax(-1).clamp(0, img_wh[0])
    y2 = torch.where(front, v, torch.full_like(v, -big)).amax(-1).clamp(0, img_wh[1])
    ok = (centre[..., 2] > 1.0) & front.any(-1) & (x2 - x1 >= min_side) & (y2 - y1 >= min_side)
    area = torch.where(ok, (x2 - x1) * (y2 - y1), torch.zeros_like(x1))
    return torch.stack([x1, y1, x2, y2], -1), area


def rect_corners_2d(xy, yaw, l, w):
    """(...,2), (...), (...), (...) -> (...,4,2) planar rectangle corners."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx = torch.stack([l, l, -l, -l], -1) * 0.5
    dy = torch.stack([w, -w, -w, w], -1) * 0.5
    x = xy[..., 0:1] + c[..., None] * dx - s[..., None] * dy
    y = xy[..., 1:2] + s[..., None] * dx + c[..., None] * dy
    return torch.stack([x, y], -1)


def rects_overlap(a, b):
    """Separating-axis test for convex quadrilaterals a, b (...,4,2) -> (...) bool."""
    def axes(p):
        e = p - p.roll(1, dims=-2)
        return torch.stack([-e[..., 1], e[..., 0]], -1)  # edge normals (...,4,2)
    ax = torch.cat([axes(a), axes(b)], -2)  # (...,8,2)
    pa = torch.einsum("...kd,...pd->...kp", ax, a)
    pb = torch.einsum("...kd,...pd->...kp", ax, b)
    sep = (pa.amax(-1) < pb.amin(-1)) | (pb.amax(-1) < pa.amin(-1))
    return ~sep.any(-1)


def ego_rects(traj):
    """Ego footprint along trajectories (...,T,3: rear-axle x,y,yaw) -> (...,T,4,2)."""
    yaw = traj[..., 2]
    centre = traj[..., :2] + EGO_REAR_TO_CENTER * torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    return rect_corners_2d(centre, yaw, torch.full_like(yaw, EGO_LENGTH), torch.full_like(yaw, EGO_WIDTH))
