from matplotlib.pyplot import grid
import torch
from torch import nn
from torch.nn import functional as F
import math
from scipy.spatial.transform import Rotation

def tum_to_pose_matrix(pose):
    # pose: [tx, ty, tz, qw, qx, qy, qz]
    assert pose.shape == (7,)
    pose_xyzw = pose[[3, 4, 5, 6]]
    r = Rotation.from_quat(pose_xyzw)
    return np.vstack([np.hstack([r.as_matrix(), pose[:3].reshape(-1, 1)]), [0, 0, 0, 1]])

def depth_regularization_si_weighted(depth_pred, depth_init, pixel_wise_weight=None, pixel_wise_weight_scale=1, pixel_wise_weight_bias=1, eps=1e-6, pixel_weight_normalize=False):
    # scale compute:
    depth_pred = torch.clamp(depth_pred, min=eps)
    depth_init = torch.clamp(depth_init, min=eps)
    log_d_pred = torch.log(depth_pred)
    log_d_init = torch.log(depth_init)
    B, _, H, W = depth_pred.shape
    scale = torch.sum(log_d_init - log_d_pred,
                      dim=[1, 2, 3], keepdim=True)/(H*W)
    if pixel_wise_weight is not None:
        if pixel_weight_normalize:
            norm = torch.max(pixel_wise_weight.detach().view(
                B, -1), dim=1, keepdim=False)[0]
            pixel_wise_weight = pixel_wise_weight / \
                (norm[:, None, None, None]+eps)
        pixel_wise_weight = pixel_wise_weight * \
            pixel_wise_weight_scale + pixel_wise_weight_bias
    else:
        pixel_wise_weight = 1
    si_loss = torch.sum(pixel_wise_weight*(log_d_pred -
                        log_d_init + scale)**2, dim=[1, 2, 3])/(H*W)
    return si_loss.mean()

class WarpImage(torch.nn.Module):
    def __init__(self):
        super(WarpImage, self).__init__()
        self.base_coord = None

    def init_grid(self, shape, device):
        H, W = shape
        hh, ww = torch.meshgrid(torch.arange(
            H).float(), torch.arange(W).float())
        coord = torch.zeros([1, H, W, 2])
        coord[0, ..., 0] = ww
        coord[0, ..., 1] = hh
        self.base_coord = coord.to(device)
        self.W = W
        self.H = H

    def warp_image(self, base_coord, img_1, flow_2_1):
        B, C, H, W = flow_2_1.shape
        sample_grids = base_coord + flow_2_1.permute([0, 2, 3, 1])
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        warped_image_2_from_1 = F.grid_sample(
            img_1, sample_grids, align_corners=True)
        return warped_image_2_from_1

    def forward(self, img_1, flow_2_1):
        B, _, H, W = flow_2_1.shape
        if self.base_coord is None:
            self.init_grid([H, W], device=flow_2_1.device)
        base_coord = self.base_coord.expand([B, -1, -1, -1])
        return self.warp_image(base_coord, img_1, flow_2_1)


class CameraIntrinsics(nn.Module):
    def __init__(self, init_focal_length=0.45, pixel_size=None):
        super().__init__()
        self.focal_length = nn.Parameter(torch.tensor(init_focal_length))
        self.pixel_size_buffer = pixel_size

    def register_shape(self, orig_shape, opt_shape) -> None:
        self.orig_shape = orig_shape
        self.opt_shape = opt_shape
        H_orig, W_orig = orig_shape
        H_opt, W_opt = opt_shape
        if self.pixel_size_buffer is None:
            # initialize as 35mm film
            pixel_size = 0.433 / (H_orig ** 2 + W_orig ** 2) ** 0.5
        else:
            pixel_size = self.pixel_size_buffer
        self.register_buffer("pixel_size", torch.tensor(pixel_size))
        intrinsics_mat_buffer = torch.zeros(3, 3)
        intrinsics_mat_buffer[0, -1] = (W_opt - 1) / 2
        intrinsics_mat_buffer[1, -1] = (H_opt - 1) / 2
        intrinsics_mat_buffer[2, -1] = 1
        self.register_buffer("intrinsics_mat", intrinsics_mat_buffer)
        self.register_buffer("scale_H", torch.tensor(
            H_opt / (H_orig * pixel_size)))
        self.register_buffer("scale_W", torch.tensor(
            W_opt / (W_orig * pixel_size)))

    def get_K_and_inv(self, with_batch_dim=True) -> torch.Tensor:
        intrinsics_mat = self.intrinsics_mat.clone()
        intrinsics_mat[0, 0] = self.focal_length * self.scale_W
        intrinsics_mat[1, 1] = self.focal_length * self.scale_H
        inv_intrinsics_mat = torch.linalg.inv(intrinsics_mat)
        if with_batch_dim:
            return intrinsics_mat[None, ...], inv_intrinsics_mat[None, ...]
        else:
            return intrinsics_mat, inv_intrinsics_mat


@torch.jit.script
def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    # if dim != 3:
    #    raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


@torch.jit.script
def get_relative_transform(src_R, src_t, tgt_R, tgt_t):
    tgt_R_inv = tgt_R.permute([0, 2, 1])
    relative_R = torch.matmul(tgt_R_inv, src_R)
    relative_t = torch.matmul(tgt_R_inv, src_t - tgt_t)
    return relative_R, relative_t


def reproject_depth(src_R, src_t, src_disp, tgt_R, tgt_t, tgt_disp, K_src, K_inv_src, K_trg, K_inv_trg, coord, eps=1e-6):
    """
    Convert the depth map's value to another camera pose.
    input:
        src_R: rotation matrix of source camera
        src_t: translation vector of source camera
        tgt_R: rotation matrix of target camera
        tgt_t: translation vector of target camera
        K: intrinsics matrix of the camera
        src_disp: disparity map of source camera
        tgt_disp: disparity map of target camera
        coord: coordinate grids
        K_inv: inverse intrinsics matrix of the camera
    output:
        tgt_depth_from_src: source depth map reprojected to target camera, values are ready for warping.
        src_depth_from_tgt: target depth map reprojected to source camera, values are ready for warping.
    """
    B, _, H, W = src_disp.shape

    src_depth = 1/(src_disp + eps)
    tgt_depth = 1/(tgt_disp + eps)
    # project 1 to 2

    src_depth_flat = src_depth.view([B, 1, H*W])
    src_xyz = src_depth_flat * src_R.matmul(K_inv_src.matmul(coord)) + src_t
    src_xyz_at_tgt_cam = K_trg.matmul(
        tgt_R.transpose(1, 2).matmul(src_xyz - tgt_t))
    tgt_depth_from_src = src_xyz_at_tgt_cam[:, 2, :].view([B, 1, H, W])
    # project 2 to 1
    tgt_depth_flat = tgt_depth.view([B, 1, H*W])
    tgt_xyz = tgt_depth_flat * tgt_R.matmul(K_inv_trg.matmul(coord)) + tgt_t
    tgt_xyz_at_src_cam = K_src.matmul(
        src_R.transpose(1, 2).matmul(tgt_xyz - src_t))
    src_depth_from_tgt = tgt_xyz_at_src_cam[:, 2, :].view([B, 1, H, W])
    return tgt_depth_from_src, src_depth_from_tgt


# @torch.jit.script
def warp_by_disp(src_R, src_t, tgt_R, tgt_t, K, src_disp, coord, inv_K, debug_mode=False, use_depth=False):

    if debug_mode:
        B, C, H, W = src_disp.shape
        relative_R, relative_t = get_relative_transform(
            src_R, src_t, tgt_R, tgt_t)

        print(relative_t.shape)
        H_mat = K.matmul(relative_R.matmul(inv_K))  # Nx3x3
        flat_disp = src_disp.view([B, 1, H * W])  # Nx1xNpoints
        relative_t_flat = relative_t.expand([-1, -1, H*W])
        rot_coord = torch.matmul(H_mat, coord)
        tr_coord = flat_disp * \
            torch.matmul(K, relative_t_flat)
        tgt_coord = rot_coord + tr_coord
        normalization_factor = (tgt_coord[:, 2:, :] + 1e-6)
        rot_coord_normalized = rot_coord / normalization_factor
        tr_coord_normalized = tr_coord / normalization_factor
        tgt_coord_normalized = rot_coord_normalized + tr_coord_normalized
        debug_info = {}
        debug_info['tr_coord_normalized'] = tr_coord_normalized
        debug_info['rot_coord_normalized'] = rot_coord_normalized
        debug_info['tgt_coord_normalized'] = tgt_coord_normalized
        debug_info['tr_coord'] = tr_coord
        debug_info['rot_coord'] = rot_coord
        debug_info['normalization_factor'] = normalization_factor
        debug_info['relative_t_flat'] = relative_t_flat
        return (tgt_coord_normalized - coord).view([B, 3, H, W]), debug_info
    else:
        B, C, H, W = src_disp.shape
        relative_R, relative_t = get_relative_transform(
            src_R, src_t, tgt_R, tgt_t)
        H_mat = K.matmul(relative_R.matmul(inv_K))  # Nx3x3
        flat_disp = src_disp.view([B, 1, H * W])  # Nx1xNpoints
        if use_depth:
            tgt_coord = flat_disp * torch.matmul(H_mat, coord) + \
                torch.matmul(K, relative_t)
        else:
            tgt_coord = torch.matmul(H_mat, coord) + flat_disp * \
                torch.matmul(K, relative_t)
        tgt_coord = tgt_coord / (tgt_coord[:, -1:, :] + 1e-6)
        return (tgt_coord - coord).view([B, 3, H, W]), tgt_coord


def unproject_depth(depth, K_inv, R, t, coord):
    # this need verification
    B, _, H, W = depth.shape
    disp_flat = depth.view([B, 1, H * W])
    xyz = disp_flat * R.matmul(K_inv.matmul(coord)) + t
    return xyz.reshape([B, 3, H, W])


@torch.jit.script
def _so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001):
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    # if dim != 3:
    #    raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class CameraPoseDeltaCollection(torch.nn.Module):
    def __init__(self, number_of_points=10) -> None:
        super().__init__()
        zero_rotation = torch.ones([1, 3]) * 1e-3
        zero_translation = torch.zeros([1, 3, 1]) + 1e-4
        for n in range(number_of_points):
            self.register_parameter(
                f"delta_rotation_{n}", nn.Parameter(zero_rotation))
            self.register_parameter(
                f"delta_translation_{n}", nn.Parameter(zero_translation)
            )
        self.register_buffer("zero_rotation", torch.eye(3)[None, ...])
        self.register_buffer("zero_translation", torch.zeros([1, 3, 1]))
        self.traced_so3_exp_map = None
        self.number_of_points = number_of_points

    def get_rotation_and_translation_params(self):
        rotation_params = []
        translation_params = []
        for n in range(self.number_of_points):
            rotation_params.append(getattr(self, f"delta_rotation_{n}"))
            translation_params.append(getattr(self, f"delta_translation_{n}"))
        return rotation_params, translation_params

    def set_rotation_and_translation(self, index, rotaion_so3, translation):
        delta_rotation = getattr(self, f"delta_rotation_{index}")
        delta_translation = getattr(self, f"delta_translation_{index}")
        delta_rotation.data = rotaion_so3.detach().clone()
        delta_translation.data = translation.detach().clone()

    def set_first_frame_pose(self, R, t):
        self.zero_rotation.data = R.detach().clone().reshape([1, 3, 3])
        self.zero_translation.data = t.detach().clone().reshape([1, 3, 1])

    def get_raw_value(self, index):
        so3 = getattr(self, f"delta_rotation_{index}")
        translation = getattr(self, f"delta_translation_{index}")
        return so3, translation

    def forward(self, list_of_index):
        se_3 = []
        t_out = []
        for idx in list_of_index:
            delta_rotation, delta_translation = self.get_raw_value(idx)
            se_3.append(delta_rotation)
            t_out.append(delta_translation)
        se_3 = torch.cat(se_3, dim=0)
        t_out = torch.cat(t_out, dim=0)
        if self.traced_so3_exp_map is None:
            self.traced_so3_exp_map = torch.jit.trace(
                _so3_exp_map, (se_3,))
        R_out = _so3_exp_map(se_3)[0]
        return R_out, t_out

    def forward_index(self, index):
        # if index == 0:
        #     return self.zero_rotation, self.zero_translation
        # else:
        delta_rotation, delta_translation = self.get_raw_value(index)
        if self.traced_so3_exp_map is None:
            self.traced_so3_exp_map = torch.jit.trace(
                _so3_exp_map, (delta_rotation,))
        R = _so3_exp_map(delta_rotation)[0]
        return R, delta_translation


class DepthScaleShiftCollection(torch.nn.Module):
    def __init__(self, n_points=10, use_inverse=False, grid_size=1):
        super().__init__()
        self.grid_size = grid_size
        for n in range(n_points):
            self.register_parameter(
                f"shift_{n}", nn.Parameter(torch.FloatTensor([0.0]))
            )
            self.register_parameter(
                f"scale_{n}", nn.Parameter(
                    torch.ones([1, 1, grid_size, grid_size]))
            )

        self.use_inverse = use_inverse
        self.output_shape = None

    def set_outputshape(self, output_shape):
        self.output_shape = output_shape

    def forward(self, index):
        shift = getattr(self, f"shift_{index}")
        scale = getattr(self, f"scale_{index}")
        if self.use_inverse:
            scale = torch.exp(scale)  # 1 / (scale ** 4)
        if self.grid_size != 1:
            scale = F.interpolate(scale, self.output_shape,
                                  mode='bilinear', align_corners=True)
        return scale, shift

    def set_scale(self, index, scale):
        scale_param = getattr(self, f"scale_{index}")
        if self.use_inverse:
            scale = math.log(scale)  # (1 / scale) ** 0.25
        scale_param.data.fill_(scale)

    def get_scale_data(self, index):
        scale = getattr(self, f"scale_{index}").data
        if self.use_inverse:
            scale = torch.exp(scale)  # 1 / (scale ** 4)
        if self.grid_size != 1:
            scale = F.interpolate(scale, self.output_shape,
                                  mode='bilinear', align_corners=True)
        return scale


def check_R_shape(R):
    r0, r1, r2 = R.shape
    assert r1 == 3 and r2 == 3


def check_t_shape(t):
    t0, t1, t2 = t.shape
    assert t1 == 3 and t2 == 1


class DepthBasedWarping(nn.Module):
    # tested
    def __init__(self) -> None:
        super().__init__()

    def generate_grid(self, H, W, device):
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
        )
        self.coord = torch.ones(
            [1, 3, H, W], device=device, dtype=torch.float32)
        self.coord[0, 0, ...] = xx
        self.coord[0, 1, ...] = yy
        self.coord = self.coord.reshape([1, 3, H * W])
        self.jitted_warp_by_disp = None

    def reproject_depth(self, src_R, src_t, src_disp, tgt_R, tgt_t, tgt_disp, K_src, K_inv_src, K_trg, K_inv_trg, eps=1e-6, check_shape=False):
        if check_shape:
            check_R_shape(src_R)
            check_R_shape(tgt_R)
            check_t_shape(src_t)
            check_t_shape(tgt_t)
            check_t_shape(src_disp)
            check_t_shape(tgt_disp)
        device = src_disp.device
        B, _, H, W = src_disp.shape
        if not hasattr(self, "coord"):
            self.generate_grid(src_disp.shape[2], src_disp.shape[3], device)
        else:
            if self.coord.shape[-1] != H * W:
                del self.coord
                self.generate_grid(H, W, device)
        return reproject_depth(src_R, src_t, src_disp, tgt_R, tgt_t, tgt_disp, K_src, K_inv_src, K_trg, K_inv_trg, self.coord, eps=eps)

    def unproject_depth(self, disp, R, t, K_inv, eps=1e-6, check_shape=False):
        if check_shape:
            check_R_shape(R)
            check_R_shape(t)

        _, _, H, W = disp.shape
        B = R.shape[0]
        device = disp.device
        if not hasattr(self, "coord"):
            self.generate_grid(H, W, device=device)
        else:
            if self.coord.shape[-1] != H * W:
                del self.coord
                self.generate_grid(H, W, device=device)
        # if self.jitted_warp_by_disp is None:
        # self.jitted_warp_by_disp = torch.jit.trace(
        #     warp_by_disp, (src_R.detach(), src_t.detach(), tgt_R.detach(), tgt_t.detach(), K, src_disp.detach(), self.coord, inv_K))
        return unproject_depth(1 / (disp + eps), K_inv, R, t, self.coord)

    def forward(
        self,
        src_R,
        src_t,
        tgt_R,
        tgt_t,
        src_disp,
        K,
        inv_K,
        eps=1e-6,
        use_depth=False,
        check_shape=False,
        debug_mode=False,
    ):
        """warp the current depth frame and generate flow field.

        Args:
            src_R (FloatTensor): 1x3x3
            src_t (FloatTensor): 1x3x1
            tgt_R (FloatTensor): Nx3x3
            tgt_t (FloatTensor): Nx3x1
            src_disp (FloatTensor): Nx1XHxW
            src_K (FloatTensor): 1x3x3
        """
        if check_shape:
            check_R_shape(src_R)
            check_R_shape(tgt_R)
            check_t_shape(src_t)
            check_t_shape(tgt_t)

        _, _, H, W = src_disp.shape
        B = tgt_R.shape[0]
        device = src_disp.device
        if not hasattr(self, "coord"):
            self.generate_grid(H, W, device=device)
        else:
            if self.coord.shape[-1] != H * W:
                del self.coord
                self.generate_grid(H, W, device=device)
        # if self.jitted_warp_by_disp is None:
        # self.jitted_warp_by_disp = torch.jit.trace(
        #     warp_by_disp, (src_R.detach(), src_t.detach(), tgt_R.detach(), tgt_t.detach(), K, src_disp.detach(), self.coord, inv_K))

        return warp_by_disp(src_R, src_t, tgt_R, tgt_t, K, src_disp, self.coord, inv_K, debug_mode, use_depth)


class DepthToXYZ(nn.Module):
    # tested
    def __init__(self) -> None:
        super().__init__()

    def generate_grid(self, H, W, device):
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
        )
        self.coord = torch.ones(
            [1, 3, H, W], device=device, dtype=torch.float32)
        self.coord[0, 0, ...] = xx
        self.coord[0, 1, ...] = yy
        self.coord = self.coord.reshape([1, 3, H * W])

    def forward(self, disp, K_inv, R, t, eps=1e-6, check_shape=False):
        """warp the current depth frame and generate flow field.

        Args:
            src_R (FloatTensor): 1x3x3
            src_t (FloatTensor): 1x3x1
            tgt_R (FloatTensor): Nx3x3
            tgt_t (FloatTensor): Nx3x1
            src_disp (FloatTensor): Nx1XHxW
            src_K (FloatTensor): 1x3x3
        """
        if check_shape:
            check_R_shape(R)
            check_R_shape(t)

        _, _, H, W = disp.shape
        B = R.shape[0]
        device = disp.device
        if not hasattr(self, "coord"):
            self.generate_grid(H, W, device=device)
        else:
            if self.coord.shape[-1] != H * W:
                del self.coord
                self.generate_grid(H, W, device=device)
        # if self.jitted_warp_by_disp is None:
        # self.jitted_warp_by_disp = torch.jit.trace(
        #     warp_by_disp, (src_R.detach(), src_t.detach(), tgt_R.detach(), tgt_t.detach(), K, src_disp.detach(), self.coord, inv_K))

        return unproject_depth(1 / (disp + eps), K_inv, R, t, self.coord)

class OccMask(torch.nn.Module):
    def __init__(self, th=3):
        super(OccMask, self).__init__()
        self.th = th
        self.base_coord = None

    def init_grid(self, shape, device):
        H, W = shape
        hh, ww = torch.meshgrid(torch.arange(
            H).float(), torch.arange(W).float())
        coord = torch.zeros([1, H, W, 2])
        coord[0, ..., 0] = ww
        coord[0, ..., 1] = hh
        self.base_coord = coord.to(device)
        self.W = W
        self.H = H

    @torch.no_grad()
    def get_oob_mask(self, base_coord, flow_1_2):
        target_range = base_coord + flow_1_2.permute([0, 2, 3, 1])
        oob_mask = (target_range[..., 0] < 0) | (target_range[..., 0] > self.W-1) | (
            target_range[..., 1] < 0) | (target_range[..., 1] > self.H-1)
        return ~oob_mask[:, None, ...]

    @torch.no_grad()
    def get_flow_inconsistency_tensor(self, base_coord, flow_1_2, flow_2_1):
        B, C, H, W = flow_1_2.shape
        sample_grids = base_coord + flow_1_2.permute([0, 2, 3, 1])
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        sampled_flow = F.grid_sample(
            flow_2_1, sample_grids, align_corners=True)
        return torch.abs((sampled_flow+flow_1_2).sum(1, keepdim=True))

    def forward(self, flow_1_2, flow_2_1):
        B, _, H, W = flow_1_2.shape
        if self.base_coord is None:
            self.init_grid([H, W], device=flow_1_2.device)
        base_coord = self.base_coord.expand([B, -1, -1, -1])
        oob_mask = self.get_oob_mask(base_coord, flow_1_2)
        flow_inconsistency_tensor = self.get_flow_inconsistency_tensor(
            base_coord, flow_1_2, flow_2_1)
        valid_flow_mask = flow_inconsistency_tensor < self.th
        return valid_flow_mask*oob_mask