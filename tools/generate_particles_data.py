import trimesh
import numpy as np
from scipy.spatial import KDTree
import os

def generate_unified_ibm_data(input_path, dx):
    # 1. 加载与预处理
    mesh = trimesh.load(input_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    
    # 2. 生成拉格朗日点
    target_count = int(np.ceil(mesh.area / (dx**2)))
    points_raw, face_indices_raw = trimesh.sample.sample_surface(mesh, target_count * 15)
    
    # 泊松盘筛选
    tree_raw = KDTree(points_raw)
    keep_idx = []
    ignored = np.zeros(len(points_raw), dtype=bool)
    for i in range(len(points_raw)):
        if ignored[i]: continue
        keep_idx.append(i)
        neighbors = tree_raw.query_ball_point(points_raw[i], dx * 0.9)
        ignored[neighbors] = True
    
    l_points = points_raw[keep_idx]
    l_normals = mesh.face_normals[face_indices_raw[keep_idx]]
    num_l = len(l_points)

    # 3. 计算每个点精确的 s_k (微元贡献法)
    num_micros = 1000000 
    micro_points, _ = trimesh.sample.sample_surface(mesh, num_micros)
    dA = mesh.area / num_micros 
    
    l_tree = KDTree(l_points)
    _, closest_indices = l_tree.query(micro_points)
    
    sk_array = np.zeros(num_l)
    for idx in closest_indices:
        sk_array[idx] += dA

    # 4. --- 核心步骤：封装数据 ---
    # 我们创建一个 7 列的矩阵：[x, y, z, nx, ny, nz, sk]
    # 这样每一行就是一个完整的质点信息，永远不会搞混
    unified_data = np.zeros((num_l, 7))
    unified_data[:, 0:3] = l_points  # 坐标
    unified_data[:, 3:6] = l_normals # 法线
    unified_data[:, 6]   = sk_array  # 面积 s_k

    # 5. 导出文件
    base_name = os.path.splitext(input_path)[0]
    
    # 导出 A: CSV 文件 (方便 C++ 使用 fscanf 或 stringstream 读取)
    csv_path = f"{base_name}_unified_data.csv"
    header = "x,y,z,nx,ny,nz,sk"
    np.savetxt(csv_path, unified_data, delimiter=',', header=header, comments='', fmt='%.10e')

    # 导出 B: 二进制文件 (如果点数极多，二进制读取速度最快，CUDA 最爱)
    bin_path = f"{base_name}_unified_data.bin"
    unified_data.astype(np.float32).tofile(bin_path)

    # 导出 C: GLB 文件 (仅供可视化)
    pc = trimesh.PointCloud(vertices=l_points)
    pc.vertices.normals = l_normals
    pc.export(f"{base_name}_visualization.glb")

    print(f"\n--- 数据封装完成 ---")
    print(f"1. 统一数据文件: {csv_path} (每行: x,y,z,nx,ny,nz,sk)")
    print(f"2. 二进制数据文件: {bin_path} (Float32 连续存储)")
    print(f"3. 粒子总数: {num_l}")
    print(f"验证面积总和: {np.sum(sk_array):.4f} (原始面积: {mesh.area:.4f})")

if __name__ == "__main__":
    generate_unified_ibm_data("box.glb", 0.05)
