import os
import argparse
import torch
import numpy as np
from plyfile import PlyElement, PlyData

def convert_pt_to_ply(checkpoint_path, output_ply_path):
    print(f"正在加载权重文件: {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']

    xyz = state_dict['positions'].numpy()  # (N, 3)
    n_points = xyz.shape[0]
    
    logit_colors = state_dict['colors']
    rgb = torch.sigmoid(logit_colors)

    f_dc = ((rgb - 0.5) / 0.28209479177387814).numpy()  # (N, 3)
    
    opacities = state_dict['opacities'].numpy()  # (N, 1)
    
    scales = state_dict['scales'].numpy()  # (N, 3)
    
    rotations = state_dict['rotations'].numpy()  # (N, 4)
    
    normals = np.zeros_like(xyz)

    attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz', 
             'f_dc_0', 'f_dc_1', 'f_dc_2', 
             'opacity', 
             'scale_0', 'scale_1', 'scale_2', 
             'rot_0', 'rot_1', 'rot_2', 'rot_3']
    
    dtype = [(name, 'f4') for name in attrs]
    elements = np.empty(n_points, dtype=dtype)
    
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    elements['opacity'] = opacities[:, 0]
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = rotations[:, 0]  # w
    elements['rot_1'] = rotations[:, 1]  # x
    elements['rot_2'] = rotations[:, 2]  # y
    elements['rot_3'] = rotations[:, 3]  # z

    print(f"正在构建 PLY 结构并写入文件...")
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_ply_path)
    print(f"转换成功！标准的 3DGS 文件已保存至: {output_ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom .pt checkpoint to standard .ply format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output .ply file")
    args = parser.parse_args()
    
    convert_pt_to_ply(args.checkpoint, args.output)