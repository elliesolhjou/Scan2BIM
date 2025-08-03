#!/usr/bin/env python3
"""
Simple fix - run without aggressive downsampling since the aggressive wall detection 
worked fine with 1.67M points in the original test
"""

import numpy as np
import open3d as o3d
import sys
import os
import time
from typing import List, Tuple

# Import the working segmentation system
from segmentation import CompleteBuildingSegmentation

def load_point_cloud(filename: str):
    """Load point cloud file"""
    
    print("=" * 60)
    print("REAL POINT CLOUD PROCESSING")
    print("=" * 60)
    
    print(f"Loading point cloud from: {filename}")
    
    try:
        point_cloud = o3d.io.read_point_cloud(filename)
        
        if len(point_cloud.points) == 0:
            print(f"❌ Point cloud is empty!")
            return None
        
        points = np.asarray(point_cloud.points)
        
        print(f"✅ Loaded {len(points):,} points")
        print(f"   Bounds: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
              f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
              f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        
        return point_cloud
        
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")
        return None

def process_with_working_segmentation(point_cloud: o3d.geometry.PointCloud):
    """Process with the segmentation system that actually works"""
    
    print(f"\n🔄 Running segmentation pipeline...")
    
    try:
        # Create segmentation system
        segmentation_system = CompleteBuildingSegmentation()
        
        # Use the main process_point_cloud method
        segments, doors = segmentation_system.process_point_cloud(point_cloud)
        
        print(f"✅ Segmentation completed!")
        print(f"   Segments found: {len(segments)}")
        print(f"   Doors found: {len(doors)}")
        
        return segments, doors
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def save_results(segments, doors):
    """Save results"""
    
    print(f"\n💾 Saving results...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save individual segments
    wall_count = 0
    floor_count = 0
    
    for segment in segments:
        # Create point cloud
        segment_cloud = o3d.geometry.PointCloud() 
        segment_cloud.points = o3d.utility.Vector3dVector(segment.points)
        
        # Color and save based on type
        if segment.region_type == 'wall':
            wall_count += 1
            segment_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for walls
            filename = f'results/segment_{wall_count:02d}_wall.ply'
        else:  # floor_ceiling
            floor_count += 1
            segment_cloud.paint_uniform_color([0.6, 0.4, 0.2])  # Brown for floors
            filename = f'results/segment_{floor_count:02d}_floor_ceiling.ply'
        
        o3d.io.write_point_cloud(filename, segment_cloud)
    
    # Create combined view
    if segments:
        all_points = []
        all_colors = []
        
        for segment in segments:
            points = segment.points
            
            if segment.region_type == 'wall':
                color = [0.8, 0.8, 0.8]  # Light gray for walls
            else:
                color = [0.4, 0.2, 0.1]  # Dark brown for floors
            
            point_colors = np.tile(color, (len(points), 1))
            all_points.extend(points)
            all_colors.extend(point_colors)
        
        combined_cloud = o3d.geometry.PointCloud()
        combined_cloud.points = o3d.utility.Vector3dVector(all_points)
        combined_cloud.colors = o3d.utility.Vector3dVector(all_colors)
        
        combined_file = 'results/building_segmented_working.ply'
        o3d.io.write_point_cloud(combined_file, combined_cloud)
        
        print(f"🎨 Combined view: {combined_file}")
    
    print(f"✅ Saved {len(segments)} segments to 'results/'")
    print(f"   Walls: {wall_count}")
    print(f"   Floors/Ceilings: {floor_count}")

def main():
    """Main processing function"""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python simple_fix_loader.py <point_cloud_file.ply>")
        return
    
    filename = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return
    
    # Load point cloud (no aggressive downsampling)
    point_cloud = load_point_cloud(filename)
    if point_cloud is None:
        return
    
    # Light downsampling only if really huge (>5M points)
    if len(point_cloud.points) > 5000000:
        print(f"🔽 Light downsampling for very large point cloud...")
        point_cloud = point_cloud.voxel_down_sample(0.01)  # 1cm voxel
        print(f"   Downsampled to {len(point_cloud.points):,} points")
    
    # Process with working segmentation
    segments, doors = process_with_working_segmentation(point_cloud)
    
    if segments:
        # Save results
        save_results(segments, doors)
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print(f"✅ Successfully processed {filename}")
        
        wall_count = len([s for s in segments if s.region_type == 'wall'])
        floor_count = len([s for s in segments if s.region_type == 'floor_ceiling'])
        
        print(f"📊 Results: {wall_count} walls, {floor_count} floors")
        print(f"📁 Saved to: results/")
        
        if wall_count > 0:
            print(f"\n🎊 SUCCESS! WALLS DETECTED: {wall_count}")
            print(f"👀 View walls: python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('results/segment_01_wall.ply')])\"")
        else:
            print(f"\n😞 Still no walls detected")
    else:
        print(f"\n❌ Processing failed - no segments detected")

if __name__ == "__main__":
    main()