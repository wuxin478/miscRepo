import trimesh
import numpy as np
from scipy.spatial import KDTree
import os
import time

def generate_professional_ibm_glb(input_path, dx):
    start_time = time.time()
    
    # 1. 加载模型
    mesh = trimesh.load(input_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    
    total_area = mesh.area
    centroid = mesh.centroid
    bbox = mesh.bounds

    # 2. 生成粒子 (N)
    target_count = int(np.ceil(total_area / (dx**2)))
    points_raw, face_indices_raw = trimesh.sample.sample_surface(mesh, target_count * 15)
    
    # 泊松盘筛选
    tree_raw = KDTree(points_raw)
    keep_idx = []
    ignored = np.zeros(len(points_raw), dtype=bool)
    for i in range(len(points_raw)):
        if ignored[i]: continue
        keep_idx.append(i)
        ignored[tree_raw.query_ball_point(points_raw[i], dx * 0.9)] = True
    
    l_points = points_raw[keep_idx].astype(np.float32)
    l_normals = mesh.face_normals[face_indices_raw[keep_idx]].astype(np.float32)
    num_l = len(l_points)

    # 3. 计算精确 s_k (100万微元积分)
    num_micros = 1000000 
    micro_points, _ = trimesh.sample.sample_surface(mesh, num_micros)
    dA = total_area / num_micros 
    l_tree = KDTree(l_points)
    _, closest_indices = l_tree.query(micro_points)
    
    sk_array = np.zeros(num_l, dtype=np.float32)
    for idx in closest_indices:
        sk_array[idx] += dA

    # 物理守恒验证
    sum_sk = np.sum(sk_array)
    error_pct = (sum_sk - total_area) / total_area * 100

    # 4. 构造 2N 顶点数据
    hacked_vertices = np.zeros((2 * num_l, 3), dtype=np.float32)
    hacked_vertices[0:num_l, :] = l_points       # 前N个点: 位置
    hacked_vertices[num_l:, 0] = sk_array         # 后N个点: X轴存s_k
    hacked_vertices[num_l:, 1:3] = 0.0            # 后N个点: YZ轴清零

    hacked_normals = np.zeros((2 * num_l, 3), dtype=np.float32)
    hacked_normals[0:num_l, :] = l_normals        # 仅前N个点有法线数据

    # 5. 导出
    output_path = f"{os.path.splitext(input_path)[0]}_pointcloud.glb"
    pc = trimesh.PointCloud(vertices=hacked_vertices)
    pc.vertices.normals = hacked_normals
    pc.export(output_path)
    
    # 6. --- 输出详尽报告 ---
    print("\n" + "="*60)
    print(f"       IBM-LBM 粒子模型生成报告 (GLB Hacked Format)")
    print("="*60)
    
    print(f"【1. 模型原始信息】")
    print(f"   - 文件名称: {input_path}")
    print(f"   - 原始面积: {total_area:.6f} m²")
    print(f"   - 几何中心: {centroid}")
    print(f"   - 包围盒 (Min): {bbox[0]}")
    print(f"   - 包围盒 (Max): {bbox[1]}")
    
    print(f"\n【2. 采样物理统计 (dx={dx})】")
    print(f"   - 粒子总数 (N): {num_l}")
    print(f"   - 面积总和 (Sum sk): {sum_sk:.8f} (误差: {error_pct:.4f}%)")
    print(f"   - s_k 分布: [Min: {np.min(sk_array):.4e} | Max: {np.max(sk_array):.4e}]")
    print(f"   - 平均 s_k: {np.mean(sk_array):.4e}")
    
    print(f"\n【3. 内存布局与读取指南】")
    print(f"   - 顶点数组总长度: {2 * num_l}")
    print(f"   - 单个顶点大小: 12 Bytes (3 * float32)")
    print(f"   - 位置数据区: 索引 [0] 至 [{num_l - 1}]")
    print(f"   - 物理面积区: 索引 [{num_l}] 至 [{2 * num_l - 1}]")
    print(f"   - 显存占用估计: {(2 * num_l * 12) / 1024:.2f} KB")
    print(f"\n   [核心读取算法]:")
    print(f"   For thread index 'i' (0 <= i < {num_l}):")
    print(f"      pos_i = buffer[i].xyz;")
    print(f"      sk_i  = buffer[i + {num_l}].x;")
    
    print(f"\n【4. 文件导出状态】")
    print(f"   - 保存路径: {os.path.abspath(output_path)}")
    print(f"   - 运行耗时: {time.time() - start_time:.2f} 秒")
    print("="*60 + "\n")

if __name__ == "__main__":
    # 配置
    INPUT = "models/fan.glb"
    DX = 0.005
    generate_professional_ibm_glb(INPUT, DX)
