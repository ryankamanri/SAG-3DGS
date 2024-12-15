import numpy as np
import cv2
import open3d.visualization
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from dataclasses import dataclass
from .mvsnet import MVSNet
from .cas_mvsnet import CascadeMVSNet

@dataclass
class MVSNetCfg:
    model: str
    ckpt_path: str


def adapt_mvsnet_state_dict(state_dict: dict):
    """
    this function will remove every prefix string of key `module.` in `state_dict.model`
    which was appended by executing `nn.DataParallel(model)`
    """
    model: dict = state_dict["model"]
    origin_keys = [key for key in model]
        
    for key in origin_keys:
        update_key = key[7:]
        model[update_key] = model[key]
        model.pop(key)
        
    return state_dict

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    batch, height, width = depth_ref.shape
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = torch.meshgrid(torch.arange(0, width, device="cuda"), torch.arange(0, height, device="cuda"), indexing='xy')
    x_ref, y_ref = x_ref.repeat(batch, 1, 1).reshape(batch, -1), y_ref.repeat(batch, 1, 1).reshape(batch, -1)
    # reference 3D space
    ones = torch.ones_like(x_ref, device="cuda")
    ref_homogeneous = torch.stack((x_ref, y_ref, ones), dim=1)
    xyz_ref = torch.matmul(torch.linalg.inv(intrinsics_ref), ref_homogeneous * depth_ref.reshape(batch, 1, -1).repeat(1, 3, 1)) # (B, C, H*W)
    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.linalg.inv(extrinsics_ref)),
                        torch.cat((xyz_ref, ones.unsqueeze(1)), dim=1))[:, :3, :] # (B, 3, H*W)
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2, :] / K_xyz_src[:, 2:3, :]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0, :].reshape([batch, height, width])
    y_src = xy_src[:, 1, :].reshape([batch, height, width])
    # # Prepare the source view grid for remap
    grid = torch.stack((x_src, y_src), dim=-1)  # [B, height, width, 2]
    grid = 2.0 * grid / torch.tensor([height - 1, width - 1], dtype=torch.float32, device="cuda") - 1.0  # Normalizing the grid to [-1, 1]

    # # Sample the source depth using bilinear interpolation
    sampled_depth_src = F.grid_sample(depth_src.unsqueeze(1), grid, mode='bilinear', padding_mode='zeros').squeeze(1)

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.linalg.inv(intrinsics_src),
                        torch.cat((xy_src, ones.unsqueeze(1)), dim=1) * sampled_depth_src.reshape(batch, 1, -1).repeat(1, 3, 1))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.linalg.inv(extrinsics_src)),
                                torch.cat((xyz_src, ones.unsqueeze(1)), dim=1))[:, :3, :]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2, :].reshape([batch, height, width])
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2, :] / K_xyz_reprojected[:, 2:3, :]
    x_reprojected = xy_reprojected[:, 0, :].reshape([batch, height, width])
    y_reprojected = xy_reprojected[:, 1, :].reshape([batch, height, width])

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[2], depth_ref.shape[1]
    x_ref, y_ref = torch.meshgrid(torch.arange(0, width, device="cuda"), torch.arange(0, height, device="cuda"), indexing='xy')
    
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                    depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    # mask = torch.logical_and(dist < 1, relative_depth_diff < 0.01)
    mask = torch.logical_and(dist < 5, relative_depth_diff < 0.05)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def colored_icp_registration(
    source: o3d.geometry.PointCloud, 
    target: o3d.geometry.PointCloud, 
    voxel_radius = [0.04, 0.02, 0.01], 
    max_iter = [50, 30, 14]
    ):
    current_transformation = np.identity(4)
    
    for scale in range(len(voxel_radius)):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        # print([iter, radius, scale])
    
        # print("3-1. 下采样的点云的体素大小： %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
    
        # print("3-2. 法向量估计.")
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    
        # print("3-3. 应用彩色点云配准")
        try:        
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=1e-6,
                                                                max_iteration=iter))
        except RuntimeError as e:
            pass
            
        current_transformation = result_icp.transformation
        # print(result_icp)

    return result_icp


def global_point_cloud_registration(vertices: list[list[torch.Tensor]], vertices_color: list[list[torch.Tensor]], b, v):
    """
    # DEPRECATED
    from J. Park, Q.-Y. Zhou, V. Koltun, Colored Point Cloud Registration Revisited, ICCV 2017
    
    input:
        `vertices`: `[[(npoints, 4) * V] * B]`
        `vertices_color`: `[[(npoints, 3) * V] * B]`
        `b`: batch size
        `v`: num of view >=2
        
    output: `[(npoints_sum, 4) * B], [(npoints_sum, 3) * B]`
    """
    
    for batch_idx in range(b):
        # build open3d point cloud class list(per view).
        pcd_list = []
        for view_idx in range(v):
            pcd = o3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(vertices[batch_idx][view_idx][:, :3].detach().cpu())
            pcd.colors = open3d.utility.Vector3dVector(vertices_color[batch_idx][view_idx].detach().cpu())
            pcd_list.append(pcd)
            
        for source_idx in range(v-1):
            for target_idx in range(source_idx+1, v):
                if source_idx == target_idx: continue
                result_icp = colored_icp_registration(pcd_list[source_idx], pcd_list[target_idx])
    pass


def generate_point_cloud_from_depth_maps(imgs: torch.Tensor, extrinsics, intrinsics, depths_est, photometric_confidences, use_point_registration=False):
    b, v, c, h, w = imgs.shape
    # the final point cloud list (per batch)
    vertices = [[] for _ in range(b)]
    vertices_color = [[] for _ in range(b)]
    # for every reference image and source image, compute the photometric mask and geometric mask
    # and generate point cloud
    for ref_idx in range(v):
        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []
        
        ref_img, ref_depth_est, ref_intrinsics, ref_extrinsics = imgs[:, ref_idx, :, :, :], depths_est[ref_idx], intrinsics[:, ref_idx, :, :], extrinsics[:, ref_idx, :, :]
        geo_mask_sum = torch.zeros_like(ref_depth_est, dtype=int)
        
        for src_idx in range(v):
            if src_idx == ref_idx: continue
            
            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(
                ref_depth_est, ref_intrinsics, ref_extrinsics,
                depths_est[src_idx], intrinsics[:, src_idx, :, :], extrinsics[:, src_idx, :, :])
            
            geo_mask_sum += geo_mask
            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)
            pass
        
        depth_est_averaged = (sum(all_srcview_depth_ests) + depths_est[ref_idx]) / (geo_mask_sum + 1)
        # at least half of source views matched
        geo_mask = geo_mask_sum >= v // 2
        photo_mask = photometric_confidences[ref_idx] > 0.8
        final_mask = torch.logical_and(photo_mask, geo_mask)
        
        if False:
            import cv2
            far = 10
            def interp_to_show(img):
                return F.interpolate(img.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0).squeeze(0)
            # convert image color channel: rgb -> bgr
            ref_img_bgr = torch.stack((ref_img[:, 2], ref_img[:, 1], ref_img[:, 0]), dim=1)
            cv2.imshow('ref_img', np.array(ref_img_bgr[0].transpose(0, 1).transpose(1, 2).detach().cpu()))
            cv2.imshow('ref_depth', np.array(interp_to_show(ref_depth_est[0]).detach().cpu()) / far)
            cv2.imshow('ref_depth * photo_mask', np.array(interp_to_show(ref_depth_est[0] * photo_mask[0]).detach().cpu()) / far)
            cv2.imshow('ref_depth * geo_mask', np.array(interp_to_show(ref_depth_est[0] * geo_mask[0]).detach().cpu()) / far)
            cv2.imshow('ref_depth * mask', np.array(interp_to_show(ref_depth_est[0] * final_mask[0]).detach().cpu()) / far)
            # cv2.waitKey(0)
        
        # project valid depth to 3d points
        # Note that we filter the valid point at last to facilitate batch parallel processing
        height, width = depth_est_averaged.shape[1:3]
        x, y = torch.meshgrid(torch.arange(0, width, device="cuda"), torch.arange(0, height, device="cuda"), indexing='xy')
        x, y = x.unsqueeze(0).repeat(b, 1, 1), y.unsqueeze(0).repeat(b, 1, 1)
        # print("valid_points", valid_points.sum())
        # x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        uvd_ref = (torch.stack((x, y, torch.ones_like(x)), dim=0) * depth_est_averaged).view(b, 3, -1)
        xyz_ref = torch.matmul(torch.linalg.inv(ref_intrinsics), uvd_ref)
        xyz_world = torch.matmul(torch.linalg.inv(ref_extrinsics),
                            torch.cat((xyz_ref, torch.ones_like(x.view(b, 1, -1))), dim=1))[:3] # (B, 4, H*W)
        colors = ref_img.view(b, c, -1) # (B, C=3, H*W)
        # colors = torch.rand(c).view(1, c, 1).repeat(b, 1, h*w).cuda() # show point from multi-view
        
        valid_points = final_mask.reshape(b, -1) # (B, H*W)
        for b_idx in range(b):
            vertices[b_idx].append(xyz_world.transpose(1, 2)[b_idx][valid_points[b_idx]])
            vertices_color[b_idx].append(colors.transpose(1, 2)[b_idx][valid_points[b_idx]])
        
    
    # concat all vertices for every scene/batch
    # every item in vertices (n_points, 4)
    for b_idx in range(b):
        vertices[b_idx] = torch.cat(vertices[b_idx], dim=0)
        vertices_color[b_idx] = torch.cat(vertices_color[b_idx], dim=0)
        
    if False:
        import open3d
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(vertices[0][:, :3].detach().cpu())
        pcd.colors = open3d.utility.Vector3dVector(vertices_color[0].detach().cpu())
        open3d.visualization.draw_geometries([pcd])
        
    return vertices, vertices_color # [(n_points, 4) * B], [(n_points, 3) * B]



class PCDGenerator(nn.Module):
    mvsnet: MVSNet
    
    def __init__(self, mvsnet_cfg: MVSNetCfg) -> None:
        super().__init__()
        self.mvsnet_cfg = mvsnet_cfg
        self.use_mvsnet = mvsnet_cfg.model == "mvsnet"
        self.use_cas_mvsnet = mvsnet_cfg.model == "cas_mvsnet"
        print(f"The Point Cloud Generator use model: {mvsnet_cfg.model}, loading checkpoint from {mvsnet_cfg.ckpt_path}...")
        # initialize pretrained mvsnet
        if self.use_mvsnet:
            self.mvsnet = MVSNet(refine=False)
            state_dict = torch.load(mvsnet_cfg.ckpt_path)
            self.mvsnet.load_state_dict(adapt_mvsnet_state_dict(state_dict)["model"])
            self.mvsnet.eval()
        elif self.use_cas_mvsnet:
            self.mvsnet = CascadeMVSNet(refine=False)
            state_dict = torch.load(mvsnet_cfg.ckpt_path)
            self.mvsnet.load_state_dict(state_dict["model"])
            self.mvsnet.eval()
            
    def preprocess(self, batch, ndepths = 192):
        imgs : torch.Tensor = batch["context"]["origin_image"] # (B, V, C, H, W)
        c2w_extrinsics : torch.Tensor = batch["context"]["extrinsics"] # (B, V, 4, 4)
        normalized_intrinsics : torch.Tensor = batch["context"]["origin_intrinsics"] # (B, V, 3, 3)
        nears, fars = batch["context"]["near"], batch["context"]["far"] # (B, V)
        b, v, c, h, w = imgs.shape
        # crop image to adapt to mvsnet (h and w can be devided by 16)
        if h % 16 != 0:
            imgs = imgs[..., :-(h % 16), :]
        if w % 16 != 0:
            imgs = imgs[..., :-(w % 16)]
        # convert extrinsics c2w -> w2c
        extrinsics = c2w_extrinsics.inverse()
        # make the intrinsic mat adapt to feature map (w / 4, w / 4)
        intrinsics = normalized_intrinsics.clone()
        intrinsics[..., 0, :] *= w / 4
        intrinsics[..., 1, :] *= h / 4
        
        if self.use_mvsnet:
            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.clone().view(b * v, 4, 4)
            proj_mat[:, :3, :4] = torch.bmm(intrinsics.view(b * v, 3, 3), proj_mat[:, :3, :4])
            proj_mat = proj_mat.view(b, v, 4, 4)
        
        elif self.use_cas_mvsnet:
            # multi-stage proj_mats
            # proj_matrices (B, V, 2(intr & extr), 4, 4)
            proj_matrices = torch.zeros(b, v, 2, 4, 4, device="cuda")
            proj_matrices[..., 0, :, :] = extrinsics
            proj_matrices[..., 1, :3, :3] = intrinsics
            
            stage2_pjmats = proj_matrices.clone()
            stage2_pjmats[..., 1, :2, :] = proj_matrices[..., 1, :2, :] * 2
            stage3_pjmats = proj_matrices.clone()
            stage3_pjmats[..., 1, :2, :] = proj_matrices[..., 1, :2, :] * 4

            proj_mat = {
                "stage1": proj_matrices,
                "stage2": stage2_pjmats,
                "stage3": stage3_pjmats
            }
            
            # the intrinsics adapts depth map (w, h), not (w / 4, h / 4)
            intrinsics[..., :2, :] *= 4
        
        # prepare depth bound (inverse depth) [v*b, d]
        min_depth = (1.0 / fars).view(b*v, 1)
        max_depth = (1.0 / nears).view(b*v, 1)
        depth_candi_curr = (
            min_depth
            + torch.linspace(0.0, 1.0, ndepths).unsqueeze(0).to(min_depth.device)
            * (max_depth - min_depth)
        )
        depth_values = 1 / depth_candi_curr # (B*V, ndepths)
        depth_values = depth_values.reshape(b, v, ndepths) # (B, V, ndepths)
        return imgs, extrinsics, intrinsics, proj_mat, depth_values
        
    @torch.no_grad()
    def forward(self, batch, ndepths = 192):
        imgs, extrinsics, intrinsics, proj_mat, depth_values = self.preprocess(batch, ndepths)
        b, v, c, h, w = imgs.shape
        
        depths_est = [] # depth map list
        photometric_confidences = []
        # for every reference image, the mvsnet will generate a depth map and a photometric confidence map
        for vi in range(v):
            outputs = self.mvsnet(imgs, proj_mat, depth_values[:, vi, :]) # depth and photometric_confidence
            depths_est.append(outputs["depth"])
            photometric_confidences.append(outputs["photometric_confidence"])
            # switch to next image
            imgs = imgs.roll(dims=1, shifts=-1)
            if self.use_mvsnet:
                proj_mat = proj_mat.roll(dims=1, shifts=-1)
            elif self.use_cas_mvsnet:
                for stage in proj_mat:
                    proj_mat[stage] = proj_mat[stage].roll(dims=1, shifts=-1)
                    
        pcd = generate_point_cloud_from_depth_maps(imgs, extrinsics, intrinsics, depths_est, photometric_confidences)
        return pcd
    pass
    