import os
import argparse
import numpy as np

def parse_cameras_txt(cameras_file_path):
    """
    Parses the cameras.txt file.
    Returns a dictionary mapping camera_id to camera_info.
    """
    cameras = {}
    with open(cameras_file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(p) for p in parts[4:]])
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras

def parse_images_txt(images_file_path):
    """
    Parses the images.txt file.
    Returns a dictionary mapping image_id to image_info, including observed 3D point IDs.
    """
    images = {}
    with open(images_file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#'):
                i += 1
                continue
            
            parts = line.split()
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]

            i += 1
            points2d_line = lines[i].strip().split()
            points2d = []
            if len(points2d_line) > 0: # Check if there are any 2D points
                for j in range(0, len(points2d_line), 3):
                    x = float(points2d_line[j])
                    y = float(points2d_line[j+1])
                    point3d_id = int(points2d_line[j+2])
                    if point3d_id != -1: # COLMAP uses -1 for points with no 3D correspondence
                        points2d.append({"xy": (x, y), "point3d_id": point3d_id})
            
            images[image_id] = {
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "camera_id": camera_id,
                "name": name,
                "points2d": points2d
            }
            i += 1
    return images

def parse_points3D_txt(points3D_file_path):
    """
    Parses the points3D.txt file.
    Returns a dictionary mapping point3d_id to point_info (XYZ, RGB, error, track).
    """
    points3D = {}
    with open(points3D_file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            point3d_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
            error = float(parts[7])
            track_data = parts[8:]
            track = []
            for k_idx in range(0, len(track_data), 2): # Renamed loop variable
                track.append({"image_id": int(track_data[k_idx]), "point2d_idx": int(track_data[k_idx+1])})
            
            points3D[point3d_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track
            }
    return points3D

def filter_points_for_frame(target_image_id, all_images_info, all_points3D_info):
    """
    Filters 3D points that are visible in the target_image_id.
    """
    if target_image_id not in all_images_info:
        print(f"Error: Image ID {target_image_id} not found in images.txt.")
        return []

    target_image_data = all_images_info[target_image_id]
    visible_point3d_ids = set()
    for p2d in target_image_data["points2d"]:
        visible_point3d_ids.add(p2d["point3d_id"])

    filtered_points = []
    for point_id in visible_point3d_ids:
        if point_id in all_points3D_info:
            point_data = all_points3D_info[point_id]
            filtered_points.append({
                "id": point_id,
                "xyz": point_data["xyz"],
                "rgb": point_data["rgb"]
            })
        else:
            print(f"Warning: Point3D ID {point_id} observed in image {target_image_id} but not found in points3D.txt.")
            
    return filtered_points

def save_filtered_points_to_txt(points_data, output_file_path):
    """
    Saves the filtered points (XYZ RGB) to a .txt file,
    including the number of points in the first line.
    """
    num_points = len(points_data)
    with open(output_file_path, 'w') as f:
        f.write(f"# Number of points: {num_points}\n") # Added this line
        f.write("# X Y Z R G B\n")
        for point in points_data:
            xyz_str = " ".join(map(str, point["xyz"]))
            rgb_str = " ".join(map(str, point["rgb"]))
            f.write(f"{xyz_str} {rgb_str}\n")
    print(f"Saved {num_points} points to {output_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract and save point cloud for all frames from a COLMAP reconstruction.")
    parser.add_argument("--colmap_dir", type=str, required=True,
                        help="Directory containing COLMAP's cameras.txt, images.txt, and points3D.txt from vggt_to_colmap.py output.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output .txt files for each frame's filtered point cloud.")

    args = parser.parse_args()

    cameras_file = os.path.join(args.colmap_dir, "cameras.txt")
    images_file = os.path.join(args.colmap_dir, "images.txt")
    points3D_file = os.path.join(args.colmap_dir, "points3D.txt")

    if not all(os.path.exists(f) for f in [cameras_file, images_file, points3D_file]):
        print(f"Error: Not all required COLMAP files (cameras.txt, images.txt, points3D.txt) found in {args.colmap_dir}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print("Parsing COLMAP files...")
    images_info = parse_images_txt(images_file)
    points3D_info = parse_points3D_txt(points3D_file)

    if not images_info:
        print("No images found in images.txt. Exiting.")
        return

    total_frames = len(images_info)
    print(f"Found {total_frames} frames in images.txt. Processing all of them...")

    for image_id in sorted(images_info.keys()): # Iterate through all image_ids
        print(f"\nProcessing IMAGE_ID: {image_id}...")
        
        image_name = images_info[image_id]["name"]
        # Sanitize image_name for use in filename, e.g., replace slashes or remove extension
        sanitized_image_name = os.path.splitext(os.path.basename(image_name))[0]
        
        output_filename = f"points_for_frame_{image_id}_{sanitized_image_name}.txt"
        output_file_path = os.path.join(args.output_dir, output_filename)

        filtered_points = filter_points_for_frame(image_id, images_info, points3D_info)

        save_filtered_points_to_txt(filtered_points, output_file_path)
        # The save_filtered_points_to_txt function will now handle the case of empty filtered_points internally
        # by writing "# Number of points: 0"

    print(f"\nFinished processing all {total_frames} frames.")

if __name__ == "__main__":
    main()