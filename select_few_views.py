import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F 
import glob
from PIL import Image
from scipy.spatial.transform import Rotation as R_scipy

# --- VGGT/DINO Imports ---
try:
    from vggt.dependency.vggsfm_utils import generate_rank_by_dino
except ImportError as e:
    print(f"Error importing VGGT modules: {e}")
    print("Please ensure the 'vggt' directory (the Python package) is findable.")
    # ... (error message)
    exit()

# --- Helper Functions (保持不变) ---
def parse_images_txt(images_file_path):
    images = {}
    with open(images_file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#'): i += 1; continue
            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8]); name = parts[9]
            i += 1
            points2d_line = lines[i].strip().split()
            points2d_ids = []
            if len(points2d_line) > 0: 
                for j_idx in range(0, len(points2d_line), 3):
                    point3d_id = int(points2d_line[j_idx+2])
                    if point3d_id != -1: points2d_ids.append(point3d_id)
            images[image_id] = {
                "qw": qw, "qx": qx, "qy": qy, "qz": qz, "tx": tx, "ty": ty, "tz": tz,
                "camera_id": camera_id, "name": name, "points2d_ids": points2d_ids
            }
            i += 1
    return images

def parse_points3D_txt(points3D_file_path):
    points3D = {}
    with open(points3D_file_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            point3d_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
            points3D[point3d_id] = { "xyz": xyz, "rgb": rgb }
    return points3D

def load_colmap_data(colmap_dir):
    images_file = os.path.join(colmap_dir, "images.txt")
    points3D_file = os.path.join(colmap_dir, "points3D.txt")
    if not os.path.exists(images_file) or not os.path.exists(points3D_file):
        raise FileNotFoundError(f"COLMAP files (images.txt, points3D.txt) not found in {colmap_dir}")
    print(f"Parsing {images_file}...")
    images_info = parse_images_txt(images_file)
    print(f"Parsing {points3D_file}...")
    points3D_info_global = parse_points3D_txt(points3D_file)
    all_valid_point3d_ids = set(points3D_info_global.keys())
    for img_id in images_info:
        images_info[img_id]["points2d_ids"] = [
            pid for pid in images_info[img_id]["points2d_ids"] if pid in all_valid_point3d_ids
        ]
    return images_info, points3D_info_global

def get_point_id_sets_per_frame(images_info):
    point_id_sets = {}
    for img_id, data in images_info.items(): point_id_sets[img_id] = set(data["points2d_ids"])
    return point_id_sets

def get_camera_extrinsics_and_names(images_info):
    extrinsics = {}
    for img_id, data in images_info.items():
        q = np.array([data["qw"], data["qx"], data["qy"], data["qz"]])
        t = np.array([data["tx"], data["ty"], data["tz"]])
        R_cw = R_scipy.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        extrinsics[img_id] = {"R_cw": R_cw, "C_w": t, "name": data["name"]}
    return extrinsics

def calculate_point_cloud_overlap(point_ids1_set, point_ids2_set, method="jaccard"):
    if not isinstance(point_ids1_set, set): point_ids1_set = set(point_ids1_set)
    if not isinstance(point_ids2_set, set): point_ids2_set = set(point_ids2_set)
    intersection_size = len(point_ids1_set.intersection(point_ids2_set))
    if not point_ids1_set and not point_ids2_set: return 0.0
    if method == "jaccard":
        union_size = len(point_ids1_set.union(point_ids2_set))
        return intersection_size / union_size if union_size > 0 else 0.0
    elif method == "min_max": 
        min_size = min(len(point_ids1_set), len(point_ids2_set))
        if min_size == 0 and intersection_size == 0: return 0.0
        return intersection_size / min_size if min_size > 0 else 0.0
    else: 
        union_size = len(point_ids1_set.union(point_ids2_set))
        return intersection_size / union_size if union_size > 0 else 0.0

def load_and_prep_images_for_dino(image_paths_ordered, target_size=336, device="cpu"):
    from torchvision import transforms 
    images_for_dino = []
    pil_to_tensor = transforms.ToTensor()
    for img_path in image_paths_ordered:
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = pil_to_tensor(img)
            images_for_dino.append(img_tensor)
        except Exception as e: print(f"Error loading image {img_path} for DINO: {e}"); raise e
    if not images_for_dino: raise ValueError("No images could be loaded for DINO.")
    processed_images = []
    for img_tensor in images_for_dino:
        img_tensor_resized = F.interpolate(img_tensor.unsqueeze(0), 
                                           size=(target_size, target_size), 
                                           mode="bilinear", 
                                           align_corners=False).squeeze(0)
        processed_images.append(img_tensor_resized)
    return torch.stack(processed_images).to(device)

def angular_distance_between_rotations(R1_cw, R2_cw):
    R_diff = np.dot(R1_cw, R2_cw.T)
    trace = np.trace(R_diff)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return angle

def get_point_cloud_bbox(points3D_info_global):
    if not points3D_info_global: return np.array([0,0,0]), np.array([0,0,0])
    all_points_xyz_list = [data["xyz"] for data in points3D_info_global.values() if "xyz" in data]
    if not all_points_xyz_list: return np.array([0,0,0]), np.array([0,0,0])
    all_points = np.array(all_points_xyz_list)
    if all_points.shape[0] == 0: return np.array([0,0,0]), np.array([0,0,0])
    return np.min(all_points, axis=0), np.max(all_points, axis=0)

def get_voxel_indices(points_xyz, bbox_min, voxel_size):
    if not isinstance(points_xyz, np.ndarray) or points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        if hasattr(points_xyz, 'shape') and points_xyz.shape[0] == 0: return set()
        return set() 
    if points_xyz.shape[0] == 0: return set()
    if voxel_size <= 1e-6: voxel_size = 1.0 
    indices = np.floor((points_xyz - bbox_min) / voxel_size).astype(int)
    return set(map(tuple, indices))

def calculate_newly_occupied_voxels(candidate_point_ids_set, all_points_xyz_map, 
                                     occupied_voxels_set, bbox_min, voxel_size):
    candidate_points_xyz_list = [all_points_xyz_map[pid] for pid in candidate_point_ids_set if pid in all_points_xyz_map]
    if not candidate_points_xyz_list: return 0, set()
    candidate_points_xyz_np = np.array(candidate_points_xyz_list)
    if candidate_points_xyz_np.ndim == 1 and candidate_points_xyz_np.shape[0] == 3: 
        candidate_points_xyz_np = candidate_points_xyz_np.reshape(1,3)
    elif candidate_points_xyz_np.ndim == 1 and candidate_points_xyz_np.shape[0] == 0:
         return 0, set()
    candidate_voxels = get_voxel_indices(candidate_points_xyz_np, bbox_min, voxel_size)
    newly_occupied = candidate_voxels - occupied_voxels_set
    return len(newly_occupied), candidate_voxels
# --- End of Helper Functions ---

# --- Strategy Implementations ---
def run_strategy_A(start_view_id, all_frame_ids_sorted, point_id_sets_per_frame, camera_params_per_frame, view_graph, num_target_views, weights_score):
    print(f"Running Strategy A with weights: {weights_score}")
    selected_views_ids = [start_view_id]
    combined_selected_points_ids = point_id_sets_per_frame.get(start_view_id, set()).copy()
    print(f"Strategy A - Selected view 1/{num_target_views}: IMAGE_ID {start_view_id} (Name: {camera_params_per_frame[start_view_id]['name']}, Points: {len(combined_selected_points_ids)})")

    while len(selected_views_ids) < num_target_views:
        best_candidate_id = -1; max_score = -float('inf')
        candidate_ids = [fid for fid in all_frame_ids_sorted if fid not in selected_views_ids]
        if not candidate_ids: break
        for cand_id in candidate_ids:
            cand_points_set = point_id_sets_per_frame.get(cand_id, set())
            cand_cam_params = camera_params_per_frame.get(cand_id);
            if not cand_cam_params: continue
            cand_cam_C = cand_cam_params["C_w"]; cand_cam_R_cw = cand_cam_params["R_cw"]
            is_connected = any(cand_id in view_graph.get(sel_id, []) for sel_id in selected_views_ids)
            if not is_connected: continue
            
            avg_overlap = sum(calculate_point_cloud_overlap(cand_points_set, point_id_sets_per_frame.get(sel_id, set())) for sel_id in selected_views_ids) / len(selected_views_ids) if selected_views_ids else 0
            novel_pts = len(cand_points_set - combined_selected_points_ids)
            novelty = novel_pts / max(1, len(cand_points_set))
            
            avg_dist = sum(np.linalg.norm(cand_cam_C - camera_params_per_frame[sel_id]["C_w"]) for sel_id in selected_views_ids) / len(selected_views_ids) if selected_views_ids else 0
            avg_angle = sum(angular_distance_between_rotations(cand_cam_R_cw, camera_params_per_frame[sel_id]["R_cw"]) for sel_id in selected_views_ids) / len(selected_views_ids) if selected_views_ids else 0
            div_dist = avg_dist / 5.0; div_angle = avg_angle / (np.pi / 2.0)
            
            score = (weights_score.get("novelty",0)*novelty + weights_score.get("overlap",0)*avg_overlap + 
                     weights_score.get("diversity_dist",0)*div_dist + weights_score.get("diversity_angle",0)*div_angle)
            if score > max_score: max_score = score; best_candidate_id = cand_id
        
        if best_candidate_id != -1:
            selected_views_ids.append(best_candidate_id)
            new_pts_set = point_id_sets_per_frame.get(best_candidate_id, set())
            novel_added = len(new_pts_set - combined_selected_points_ids)
            combined_selected_points_ids.update(new_pts_set)
            print(f"Strategy A - Selected view {len(selected_views_ids)}/{num_target_views}: IMAGE_ID {best_candidate_id} (Name: {camera_params_per_frame[best_candidate_id]['name']}, Score: {max_score:.4f}, Novel: {novel_added}, Total: {len(new_pts_set)})")
        else: print("Strategy A - No suitable next view. Stopping."); break
    return selected_views_ids, combined_selected_points_ids

def run_strategy_B( 
    start_view_id, all_frame_ids_sorted, point_id_sets_per_frame,
    camera_params_per_frame, view_graph, num_target_views, weights_score,
    points3D_info_global 
):
    print(f"Running Strategy B (ActiveGAMER Inspired) with weights: {weights_score}")
    selected_views_ids = [start_view_id]
    combined_selected_points_ids = point_id_sets_per_frame.get(start_view_id, set()).copy()
    all_global_point_ids = set(points3D_info_global.keys())

    print(f"Strategy B - Selected view 1/{num_target_views}: IMAGE_ID {start_view_id} "
          f"(Name: {camera_params_per_frame[start_view_id]['name']}, "
          f"Points: {len(point_id_sets_per_frame.get(start_view_id, set()))})")

    while len(selected_views_ids) < num_target_views:
        best_candidate_id = -1; max_score = -float('inf')
        potential_candidates = set(n_id for sel_id in selected_views_ids for n_id in view_graph.get(sel_id, []) if n_id not in selected_views_ids)
        candidate_ids = list(potential_candidates)
        if not candidate_ids: print("Strategy B - No more connected candidate views left."); break

        global_points_yet_to_be_seen_by_selected_set = all_global_point_ids - combined_selected_points_ids
        if not global_points_yet_to_be_seen_by_selected_set:
            print("Strategy B - All global points are already covered by selected views. Stopping."); break
        
        scores_log_for_iter = {}
        for cand_id in candidate_ids:
            cand_points_set = point_id_sets_per_frame.get(cand_id, set())
            if not cand_points_set: scores_log_for_iter[cand_id] = {"final_score": -float('inf'), "reason": "no points"}; continue
            cand_cam_params = camera_params_per_frame.get(cand_id); 
            if not cand_cam_params: continue
            cand_cam_C = cand_cam_params["C_w"]; cand_cam_R_cw = cand_cam_params["R_cw"]

            newly_covered_global_points_count = len(cand_points_set.intersection(global_points_yet_to_be_seen_by_selected_set))
            info_gain_proxy_score = float(newly_covered_global_points_count) 

            avg_dist = sum(np.linalg.norm(cand_cam_C - camera_params_per_frame[sel_id]["C_w"]) for sel_id in selected_views_ids) / len(selected_views_ids) if selected_views_ids else 0
            avg_angle = sum(angular_distance_between_rotations(cand_cam_R_cw, camera_params_per_frame[sel_id]["R_cw"]) for sel_id in selected_views_ids) / len(selected_views_ids) if selected_views_ids else 0
            div_dist = avg_dist / 5.0; div_angle = avg_angle / (np.pi / 2.0)
            
            max_overlap_with_selected = 0
            for sel_id in selected_views_ids:
                 if cand_id in view_graph.get(sel_id, []): 
                    overlap_val = calculate_point_cloud_overlap(cand_points_set, point_id_sets_per_frame.get(sel_id, set()))
                    if overlap_val > max_overlap_with_selected: max_overlap_with_selected = overlap_val
            coherence = max_overlap_with_selected

            score = (weights_score.get("info_gain_proxy", 0.6) * info_gain_proxy_score +
                     weights_score.get("diversity_dist", 0.15) * div_dist +
                     weights_score.get("diversity_angle", 0.15) * div_angle +
                     weights_score.get("coherence", 0.1) * coherence)
            scores_log_for_iter[cand_id] = {"newly_covered_global_pts": newly_covered_global_points_count, "final_score": score}
            if score > max_score: max_score = score; best_candidate_id = cand_id
        
        if best_candidate_id != -1:
            selected_views_ids.append(best_candidate_id)
            new_pts_set = point_id_sets_per_frame.get(best_candidate_id, set())
            combined_selected_points_ids.update(new_pts_set)
            print(f"Strategy B - Selected view {len(selected_views_ids)}/{num_target_views}: IMAGE_ID {best_candidate_id} (Name: {camera_params_per_frame[best_candidate_id]['name']}, Score: {max_score:.4f}, CoversNewGlobal: {scores_log_for_iter[best_candidate_id]['newly_covered_global_pts']}, Total: {len(new_pts_set)})")
        else: print("Strategy B - No suitable next view. Stopping."); break
    return selected_views_ids, combined_selected_points_ids

def run_strategy_C(start_view_id, all_frame_ids_sorted, point_id_sets_per_frame, camera_params_per_frame, view_graph, num_target_views, weights_score):
    print(f"Running Strategy C with weights: {weights_score}")
    selected_views_ids = [start_view_id]
    combined_selected_points_ids = point_id_sets_per_frame.get(start_view_id, set()).copy()
    print(f"Strategy C - Selected view 1/{num_target_views}: IMAGE_ID {start_view_id} (Name: {camera_params_per_frame[start_view_id]['name']}, Points: {len(combined_selected_points_ids)})")
    
    while len(selected_views_ids) < num_target_views:
        best_candidate_id = -1; max_score = -float('inf')
        candidate_ids = [fid for fid in all_frame_ids_sorted if fid not in selected_views_ids]
        if not candidate_ids: break
        last_sel_id = selected_views_ids[-1]
        last_sel_cam_C = camera_params_per_frame[last_sel_id]["C_w"]
        
        scores_log_for_iter = {}
        for cand_id in candidate_ids:
            cand_points_set = point_id_sets_per_frame.get(cand_id, set())
            cand_cam_params = camera_params_per_frame.get(cand_id); 
            if not cand_cam_params: continue
            cand_cam_C = cand_cam_params["C_w"]
            
            if cand_id not in view_graph.get(last_sel_id, []): 
                scores_log_for_iter[cand_id] = {"final_score": -float('inf'), "reason": "not_connected_to_last"}; continue

            overlap_last = calculate_point_cloud_overlap(cand_points_set, point_id_sets_per_frame.get(last_sel_id,set()))
            baseline_last_raw = np.linalg.norm(cand_cam_C - last_sel_cam_C)
            baseline_last = baseline_last_raw / 5.0
            novel_pts = len(cand_points_set - combined_selected_points_ids)
            novelty = novel_pts / max(1, len(cand_points_set))
            
            score = (weights_score.get("novelty",0)*novelty + weights_score.get("overlap_last",0)*overlap_last + weights_score.get("baseline_last",0)*baseline_last)
            scores_log_for_iter[cand_id] = {"final_score": score, "novelty_pts": novel_pts, "baseline_raw": baseline_last_raw}
            if score > max_score: max_score = score; best_candidate_id = cand_id

        if best_candidate_id != -1:
            selected_views_ids.append(best_candidate_id)
            new_pts_set = point_id_sets_per_frame.get(best_candidate_id, set())
            novel_added = len(new_pts_set - combined_selected_points_ids)
            combined_selected_points_ids.update(new_pts_set)
            print(f"Strategy C - Selected view {len(selected_views_ids)}/{num_target_views}: IMAGE_ID {best_candidate_id} (Name: {camera_params_per_frame[best_candidate_id]['name']}, Score: {max_score:.4f}, Novel: {novel_added}, Baseline: {scores_log_for_iter[best_candidate_id]['baseline_raw']:.2f}, Total: {len(new_pts_set)})")
        else: print("Strategy C - No suitable next view. Stopping."); break
    return selected_views_ids, combined_selected_points_ids

def run_strategy_D(start_view_id, all_frame_ids_sorted, point_id_sets_per_frame, camera_params_per_frame, view_graph, num_target_views, weights_score, points3D_info_global, voxel_size_ratio_arg):
    print(f"Running Strategy D (Density & Occupancy) with weights: {weights_score}")
    selected_views_ids = [start_view_id]
    combined_selected_points_ids = point_id_sets_per_frame.get(start_view_id, set()).copy()
    all_points_xyz_map = {pid: data["xyz"] for pid, data in points3D_info_global.items()}
    bbox_min, bbox_max = get_point_cloud_bbox(points3D_info_global)
    scene_diagonal = np.linalg.norm(bbox_max - bbox_min)
    voxel_size = scene_diagonal * voxel_size_ratio_arg if scene_diagonal > 1e-6 else 1.0
    print(f"Scene BBox Min: {bbox_min}, Max: {bbox_max}, Diagonal: {scene_diagonal:.2f}, Voxel Size: {voxel_size:.2f}")
    initial_pts_xyz_list = [all_points_xyz_map[pid] for pid in combined_selected_points_ids if pid in all_points_xyz_map]
    initial_pts_xyz = np.array(initial_pts_xyz_list) if initial_pts_xyz_list else np.empty((0,3))
    current_occupied_voxels = get_voxel_indices(initial_pts_xyz, bbox_min, voxel_size)
    print(f"Strategy D - Selected view 1/{num_target_views}: IMAGE_ID {start_view_id} (Name: {camera_params_per_frame[start_view_id]['name']}, Points: {len(combined_selected_points_ids)}, OccupiedVoxels: {len(current_occupied_voxels)})")

    while len(selected_views_ids) < num_target_views:
        best_candidate_id = -1; max_score = -float('inf')
        candidate_ids = [fid for fid in all_frame_ids_sorted if fid not in selected_views_ids]
        if not candidate_ids: break
        
        scores_log_for_iter = {}
        for cand_id in candidate_ids:
            cand_points_set = point_id_sets_per_frame.get(cand_id, set())
            if not cand_points_set: scores_log_for_iter[cand_id] = {"final_score": -float('inf'), "reason": "no points"}; continue
            cand_cam_params = camera_params_per_frame.get(cand_id); 
            if not cand_cam_params: continue
            
            is_connected = any(cand_id in view_graph.get(sel_id, []) for sel_id in selected_views_ids)
            if not is_connected: scores_log_for_iter[cand_id] = {"final_score": -float('inf'), "reason": "not_connected"}; continue

            novel_pts = len(cand_points_set - combined_selected_points_ids)
            density = novel_pts / max(1, len(cand_points_set))
            
            num_new_voxels, _ = calculate_newly_occupied_voxels(cand_points_set, all_points_xyz_map, current_occupied_voxels, bbox_min, voxel_size)
            _, cand_total_voxels_if_alone = calculate_newly_occupied_voxels(cand_points_set, all_points_xyz_map, set(), bbox_min, voxel_size)
            occupancy = num_new_voxels / max(1, len(cand_total_voxels_if_alone))
            
            cand_cam_C = cand_cam_params["C_w"]; cand_cam_R_cw = cand_cam_params["R_cw"]
            avg_dist = sum(np.linalg.norm(cand_cam_C - camera_params_per_frame[sel_id]["C_w"]) for sel_id in selected_views_ids) / len(selected_views_ids) if selected_views_ids else 0
            avg_angle = sum(angular_distance_between_rotations(cand_cam_R_cw, camera_params_per_frame[sel_id]["R_cw"]) for sel_id in selected_views_ids) / len(selected_views_ids) if selected_views_ids else 0
            div_dist = avg_dist / 5.0; div_angle = avg_angle / (np.pi / 2.0)

            score = (weights_score.get("density",0)*density + weights_score.get("occupancy",0)*occupancy +
                     weights_score.get("diversity_dist",0)*div_dist + weights_score.get("diversity_angle",0)*div_angle)
            scores_log_for_iter[cand_id] = {"final_score": score, "novel_pts": novel_pts, "new_voxels": num_new_voxels}
            if score > max_score: max_score = score; best_candidate_id = cand_id
        
        if best_candidate_id != -1:
            selected_views_ids.append(best_candidate_id)
            new_pts_set = point_id_sets_per_frame.get(best_candidate_id, set())
            novel_added = len(new_pts_set - combined_selected_points_ids)
            combined_selected_points_ids.update(new_pts_set)
            
            best_cand_pts_xyz_list = [all_points_xyz_map[pid] for pid in new_pts_set if pid in all_points_xyz_map]
            if best_cand_pts_xyz_list:
                 current_occupied_voxels.update(get_voxel_indices(np.array(best_cand_pts_xyz_list), bbox_min, voxel_size))

            print(f"Strategy D - Selected view {len(selected_views_ids)}/{num_target_views}: IMAGE_ID {best_candidate_id} (Name: {camera_params_per_frame[best_candidate_id]['name']}, Score: {max_score:.4f}, Novel: {novel_added}, NewVox: {scores_log_for_iter[best_candidate_id]['new_voxels']}, Total: {len(new_pts_set)})")
        else: print("Strategy D - No suitable next view. Stopping."); break
    return selected_views_ids, combined_selected_points_ids

def run_strategy_random_fully(
    start_view_id, all_frame_ids_sorted, 
    point_id_sets_per_frame, 
    camera_params_per_frame, 
    num_target_views
):
    print(f"Running Strategy Random_Fully (Initial: {start_view_id}, Subsequent: Random)")
    selected_views_ids = [start_view_id]
    combined_selected_points_ids = point_id_sets_per_frame.get(start_view_id, set()).copy()

    start_view_name = camera_params_per_frame.get(start_view_id, {}).get('name', 'Unknown')
    print(f"Strategy Random_Fully - Selected view 1/{num_target_views}: IMAGE_ID {start_view_id} "
          f"(Name: {start_view_name}), "
          f"Points: {len(point_id_sets_per_frame.get(start_view_id, set()))})")
    
    candidate_ids_pool = [fid for fid in all_frame_ids_sorted if fid not in selected_views_ids]
    np.random.shuffle(candidate_ids_pool) 

    while len(selected_views_ids) < num_target_views and candidate_ids_pool:
        next_view_id = candidate_ids_pool.pop(0)
        
        selected_views_ids.append(next_view_id)
        newly_added_points_set = point_id_sets_per_frame.get(next_view_id, set())
        combined_selected_points_ids.update(newly_added_points_set)
        
        next_view_name = camera_params_per_frame.get(next_view_id, {}).get('name', 'Unknown')
        print(f"Strategy Random_Fully - Selected view {len(selected_views_ids)}/{num_target_views}: IMAGE_ID {next_view_id} "
              f"(Name: {next_view_name}), "
              f"Total Points in View: {len(newly_added_points_set)})")
            
    if len(selected_views_ids) < num_target_views and not candidate_ids_pool :
        print(f"Warning: Could only select {len(selected_views_ids)} views as candidate pool exhausted.")

    return selected_views_ids, combined_selected_points_ids
# --- End of Strategy Implementations ---

def select_few_views_main_logic(
    all_image_paths_dir_arg, global_colmap_dir_arg, num_target_views_arg,
    overlap_threshold_a_arg, dino_image_size_arg, 
    initial_frame_strategy_arg, 
    dino_top_k_for_refinement_arg, 
    selection_strategy_arg, 
    weights_strategy_A_arg, 
    weights_strategy_B_arg, 
    weights_strategy_C_arg, 
    weights_strategy_D_arg, 
    voxel_size_ratio_arg, 
    device_arg
):
    # Step 1, 2, 3 ...
    print("--- Step 1: Loading and Preparing Data ---")
    try:
        images_info, points3D_info_global = load_colmap_data(global_colmap_dir_arg)
        point_id_sets_per_frame = get_point_id_sets_per_frame(images_info)
        camera_params_per_frame = get_camera_extrinsics_and_names(images_info)
        if not images_info: print("No image data loaded. Exiting."); return [], None, None, None
        all_frame_ids_sorted = sorted(list(images_info.keys()))
        print(f"Successfully loaded data for {len(all_frame_ids_sorted)} images.")
    except FileNotFoundError as e: print(f"Error: {e}"); return [], None, None, None
    except Exception as e: print(f"An unexpected error during data prep: {e}"); return [], None, None, None
    print("--- Data Preparation Complete ---")

    print("\n--- Step 2: Building View Graph ---")
    view_graph = {fid: [] for fid in all_frame_ids_sorted}
    num_edges = 0
    for i_vg in range(len(all_frame_ids_sorted)):
        for j_vg in range(i_vg + 1, len(all_frame_ids_sorted)):
            id1_vg, id2_vg = all_frame_ids_sorted[i_vg], all_frame_ids_sorted[j_vg]
            set1_vg, set2_vg = point_id_sets_per_frame.get(id1_vg, set()), point_id_sets_per_frame.get(id2_vg, set())
            overlap_vg = calculate_point_cloud_overlap(set1_vg, set2_vg)
            if overlap_vg >= overlap_threshold_a_arg:
                view_graph[id1_vg].append(id2_vg); view_graph[id2_vg].append(id1_vg); num_edges += 1
    print(f"View graph: {len(all_frame_ids_sorted)} nodes, {num_edges} edges. Avg degree: {(sum(len(n) for n in view_graph.values()) / len(view_graph) if view_graph else 0):.2f}")
    print("--- View Graph Construction Complete ---")
    
    print(f"\n--- Step 3: Selecting Initial Frame using Strategy: {initial_frame_strategy_arg} ---")
    start_view_id = -1
    ordered_original_image_paths = []
    image_load_error = False
    for frame_id_orig_path in all_frame_ids_sorted:
        img_name_orig_path = camera_params_per_frame[frame_id_orig_path]["name"]
        full_img_path_orig = os.path.join(all_image_paths_dir_arg, img_name_orig_path)
        if not os.path.exists(full_img_path_orig): print(f"Error: Image file not found: {full_img_path_orig}"); image_load_error = True; break
        ordered_original_image_paths.append(full_img_path_orig)

    if image_load_error and initial_frame_strategy_arg == "dino_top_k":
        print("Cannot proceed with DINO due to missing image files. Falling back to 'first_frame'.")
        initial_frame_strategy_arg = "first_frame"

    if initial_frame_strategy_arg == "dino_top_k":
        if not ordered_original_image_paths: print("No images to process for DINO. Falling back to 'first_frame'."); initial_frame_strategy_arg = "first_frame"
        else:
            try:
                print(f"Loading {len(ordered_original_image_paths)} images for DINO (size: {dino_image_size_arg})...")
                images_tensor_dino = load_and_prep_images_for_dino(ordered_original_image_paths, target_size=dino_image_size_arg, device=device_arg)
                print("Running DINO feature ranking..."); torch.cuda.empty_cache()
                ranked_local_indices = generate_rank_by_dino(images_tensor_dino, query_frame_num=min(len(ordered_original_image_paths), dino_top_k_for_refinement_arg), image_size=dino_image_size_arg, device=device_arg)
                if not ranked_local_indices: print("DINO ranking returned empty. Falling back to 'first_frame'."); initial_frame_strategy_arg = "first_frame"
                else:
                    print(f"Refining initial view from DINO Top-{len(ranked_local_indices)} candidates using new scoring:")
                    best_refined_start_view_id = -1; max_refined_score = -float('inf')
                    max_possible_points = max(len(s) for s in point_id_sets_per_frame.values()) if point_id_sets_per_frame else 1
                    max_possible_degree = max(len(n) for n in view_graph.values()) if view_graph else 1
                    for i_refine, local_idx_refine in enumerate(ranked_local_indices): 
                        cand_id_refine = all_frame_ids_sorted[local_idx_refine] 
                        num_points_refine = len(point_id_sets_per_frame.get(cand_id_refine, set())) 
                        degree_centrality_refine = len(view_graph.get(cand_id_refine, [])) 
                        dino_rank_score_refine = 1.0 / (i_refine + 1) 
                        norm_num_points_refine = num_points_refine / max(1, max_possible_points) 
                        norm_degree_refine = degree_centrality_refine / max(1, max_possible_degree) 
                        w_dino_refine, w_points_refine, w_degree_refine = 0.3, 0.4, 0.3 
                        current_refined_score_val = (w_dino_refine * dino_rank_score_refine + w_points_refine * norm_num_points_refine + w_degree_refine * norm_degree_refine) 
                        print(f"  Cand ID {cand_id_refine} (DINO rank: {i_refine+1}): Pts={num_points_refine}({norm_num_points_refine:.2f}), Deg={degree_centrality_refine}({norm_degree_refine:.2f}), DRS={dino_rank_score_refine:.2f}, TotalScore={current_refined_score_val:.4f}")
                        if current_refined_score_val > max_refined_score: max_refined_score = current_refined_score_val; best_refined_start_view_id = cand_id_refine
                    if best_refined_start_view_id != -1: start_view_id = best_refined_start_view_id
                    else: start_view_id = all_frame_ids_sorted[ranked_local_indices[0]]; print(f"DINO Top-K refinement failed, using DINO Top-1.")
                    print(f"Initial view by DINO Top-K (refined score): ID {start_view_id} (Name: {camera_params_per_frame[start_view_id]['name']})")
            except Exception as e_dino: print(f"Error during DINO: {e_dino}\nFalling back to 'first_frame'."); initial_frame_strategy_arg = "first_frame"
    
    if initial_frame_strategy_arg == "first_frame":
        if not all_frame_ids_sorted: print("Error: No frames for 'first_frame' strategy."); return [], None, None, None
        start_view_id = all_frame_ids_sorted[44]
        print(f"Initial view by 'first_frame': ID {start_view_id} (Name: {camera_params_per_frame[start_view_id]['name']})")
    elif initial_frame_strategy_arg == "random": # This elif will now handle the "random" case.
        if not all_frame_ids_sorted: print("Error: No frames for 'random' strategy."); return [], None, None, None
        start_view_id = np.random.choice(all_frame_ids_sorted)
        print(f"Initial view by 'random': ID {start_view_id} (Name: {camera_params_per_frame[start_view_id]['name']})")
    
    if start_view_id == -1 : 
        if all_frame_ids_sorted: start_view_id = all_frame_ids_sorted[0]; print(f"Ultimate fallback to first frame: ID {start_view_id}")
        else: print("CRITICAL ERROR: No frames available."); return [], None, None, None
    print("--- Initial Frame Selection Complete ---")

    selected_views_ids_final, combined_points_ids_final = [], set()
    if selection_strategy_arg == "strategy_A":
        selected_views_ids_final, combined_points_ids_final = run_strategy_A(start_view_id, all_frame_ids_sorted, point_id_sets_per_frame, camera_params_per_frame, view_graph, num_target_views_arg, weights_strategy_A_arg)
    elif selection_strategy_arg == "strategy_B": 
        selected_views_ids_final, combined_points_ids_final = run_strategy_B(start_view_id, all_frame_ids_sorted, point_id_sets_per_frame, camera_params_per_frame, view_graph, num_target_views_arg, weights_strategy_B_arg, points3D_info_global)
    elif selection_strategy_arg == "strategy_C":
        selected_views_ids_final, combined_points_ids_final = run_strategy_C(start_view_id, all_frame_ids_sorted, point_id_sets_per_frame, camera_params_per_frame, view_graph, num_target_views_arg, weights_strategy_C_arg)
    elif selection_strategy_arg == "strategy_D":
        selected_views_ids_final, combined_points_ids_final = run_strategy_D(start_view_id, all_frame_ids_sorted, point_id_sets_per_frame, camera_params_per_frame, view_graph, num_target_views_arg, weights_strategy_D_arg, points3D_info_global, voxel_size_ratio_arg)
    elif selection_strategy_arg == "strategy_random_fully":
        selected_views_ids_final, combined_points_ids_final = run_strategy_random_fully(start_view_id, all_frame_ids_sorted, point_id_sets_per_frame, camera_params_per_frame, num_target_views_arg)
    else:
        print(f"Error: Unknown subsequent view selection strategy '{selection_strategy_arg}'"); return [], None, None, None
    return selected_views_ids_final, combined_points_ids_final, points3D_info_global, camera_params_per_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a few representative views for SfM.")
    parser.add_argument("--all_image_paths_dir", type=str, required=True)
    parser.add_argument("--global_colmap_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="selected_views.txt")

    parser.add_argument("--initial_frame_strategy", type=str, default="dino_top_k", 
                        choices=["dino_top_k", "first_frame", "random"]) 
    parser.add_argument("--dino_top_k_for_refinement", type=int, default=5)
    
    parser.add_argument("--overlap_threshold_a", type=float, default=0.1)
    parser.add_argument("--dino_image_size", type=int, default=336)
    parser.add_argument("--num_target_views", type=int, default=6)
    
    parser.add_argument("--selection_strategy", type=str, default="strategy_A", 
                        choices=["strategy_A", "strategy_B", "strategy_C", "strategy_D", "strategy_random_fully"])
    
    parser.add_argument("--wA_novelty", type=float, default=0.4); parser.add_argument("--wA_overlap", type=float, default=0.3)
    parser.add_argument("--wA_div_dist", type=float, default=0.15); parser.add_argument("--wA_div_angle", type=float, default=0.15)
    
    parser.add_argument("--wB_info_gain_proxy", type=float, default=0.6); parser.add_argument("--wB_div_dist", type=float, default=0.15)
    parser.add_argument("--wB_div_angle", type=float, default=0.15); parser.add_argument("--wB_coherence", type=float, default=0.1)
    
    parser.add_argument("--wC_novelty", type=float, default=0.3); parser.add_argument("--wC_overlap_last", type=float, default=0.4)
    parser.add_argument("--wC_baseline_last", type=float, default=0.3)

    parser.add_argument("--wD_density", type=float, default=0.4); parser.add_argument("--wD_occupancy", type=float, default=0.4)
    parser.add_argument("--wD_div_dist", type=float, default=0.1); parser.add_argument("--wD_div_angle", type=float, default=0.1)
    parser.add_argument("--voxel_size_ratio", type=float, default=0.05)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_A = {"novelty": args.wA_novelty, "overlap": args.wA_overlap, "diversity_dist": args.wA_div_dist, "diversity_angle": args.wA_div_angle}
    weights_B = {"info_gain_proxy": args.wB_info_gain_proxy, "diversity_dist": args.wB_div_dist, "diversity_angle": args.wB_div_angle, "coherence": args.wB_coherence}
    weights_C = {"novelty": args.wC_novelty, "overlap_last": args.wC_overlap_last, "baseline_last": args.wC_baseline_last}
    weights_D = {"density": args.wD_density, "occupancy": args.wD_occupancy, "diversity_dist": args.wD_div_dist, "diversity_angle": args.wD_div_angle}

    # Note: strategy_random_fully doesn't use weights, so we pass None or an empty dict for its weight argument slot
    selected_views_ids, combined_points_ids, points3D_info, camera_params = select_few_views_main_logic(
        args.all_image_paths_dir, args.global_colmap_dir, args.num_target_views,
        args.overlap_threshold_a, args.dino_image_size, 
        args.initial_frame_strategy, 
        args.dino_top_k_for_refinement, 
        args.selection_strategy,
        weights_A, weights_B, weights_C, weights_D, 
        args.voxel_size_ratio,
        device
    )

    if not selected_views_ids: print("No views were selected. Exiting."); exit()
        
    print("\n--- Final Selected Views (IMAGE_IDs) ---"); print(selected_views_ids)
    selected_image_names = [camera_params[img_id]["name"] for img_id in selected_views_ids if img_id in camera_params]
    print("\n--- Corresponding Image Names ---"); 
    for name in selected_image_names: print(name)

    if points3D_info and combined_points_ids is not None :
        total_global_points = len(points3D_info.keys())
        points_in_selected_views = len(combined_points_ids)
        if total_global_points > 0:
            coverage_percentage = (points_in_selected_views / total_global_points) * 100
            print(f"\n--- Point Cloud Coverage Evaluation ---")
            print(f"# Initial Frame Strategy: {args.initial_frame_strategy}")
            print(f"# Subsequent View Selection Strategy: {args.selection_strategy}")
            print(f"Total 3D points in the global reconstruction: {total_global_points}")
            print(f"Unique 3D points by selected {len(selected_views_ids)} views: {points_in_selected_views}")
            print(f"Coverage Percentage: {coverage_percentage:.2f}%")
        else: print("\n--- Point Cloud Coverage: Total global points is zero. ---")
    else: print("\n--- Point Cloud Coverage: Could not perform evaluation. ---")
    print(f"\n--- View Selection Process Complete ---")

    with open(args.output_file, "w") as f:
        f.write(f"# Initial Frame Strategy: {args.initial_frame_strategy}\n")
        if args.initial_frame_strategy == "dino_top_k":
            f.write(f"# DINO Top-K for refinement: {args.dino_top_k_for_refinement}\n")
        f.write(f"# Subsequent View Selection Strategy: {args.selection_strategy}\n")
        current_weights_str = ""
        if args.selection_strategy == "strategy_A": current_weights_str = f"# Weights_A: {weights_A}"
        elif args.selection_strategy == "strategy_B": current_weights_str = f"# Weights_B: {weights_B}"
        elif args.selection_strategy == "strategy_C": current_weights_str = f"# Weights_C: {weights_C}"
        elif args.selection_strategy == "strategy_D": current_weights_str = f"# Weights_D: {weights_D}, VoxelRatio: {args.voxel_size_ratio}"
        elif args.selection_strategy == "strategy_random_fully": current_weights_str = "# Weights: None (Pure Random Subsequent Selection)"
        f.write(f"{current_weights_str}\n")
        f.write(f"Coverage Percentage: {coverage_percentage:.2f}%\n")
        for name in selected_image_names: f.write(f"{name}\n")
    print(f"\nSelected image names and strategy info saved to {args.output_file}")