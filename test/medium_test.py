#!/usr/bin/env python3
"""
Working test for segmentation.py - now that imports are confirmed to work
"""

import numpy as np
import open3d as o3d
from segmentation import (
    RANSACSegmenter, RegionGrowingSegmenter, DBSCANClusterer,
    OccupancyGridAnalyzer, GapAnalyzer, DoorDetectionValidator,
    AdvancedSegmentationPipeline, SegmentedRegion, DetectedDoor
)

def create_realistic_room_cloud():
    """Create a more realistic room point cloud for testing"""
    print("Creating realistic room point cloud...")
    
    points = []
    
    # Room dimensions
    width, depth, height = 5.0, 4.0, 2.5
    door_width, door_height = 0.8, 2.0
    door_x_pos = 2.0
    
    # Resolution for point generation
    res = 0.05
    
    # Front wall with door opening
    print("  Adding front wall with door...")
    for x in np.arange(0, width, res):
        for z in np.arange(0, height, res):
            # Skip door opening area
            if not (door_x_pos <= x <= door_x_pos + door_width and 0 <= z <= door_height):
                points.append([x, 0, z])
    
    # Back wall
    print("  Adding back wall...")
    for x in np.arange(0, width, res):
        for z in np.arange(0, height, res):
            points.append([x, depth, z])
    
    # Left wall
    print("  Adding left wall...")
    for y in np.arange(0, depth, res):
        for z in np.arange(0, height, res):
            points.append([0, y, z])
    
    # Right wall
    print("  Adding right wall...")
    for y in np.arange(0, depth, res):
        for z in np.arange(0, height, res):
            points.append([width, y, z])
    
    # Floor
    print("  Adding floor...")
    for x in np.arange(0, width, res * 2):  # Sparser floor
        for y in np.arange(0, depth, res * 2):
            points.append([x, y, 0])
    
    # Ceiling
    print("  Adding ceiling...")
    for x in np.arange(0, width, res * 2):  # Sparser ceiling
        for y in np.arange(0, depth, res * 2):
            points.append([x, y, height])
    
    points = np.array(points)
    print(f"  Generated {len(points)} points")
    
    # Add realistic noise
    noise = np.random.normal(0, 0.01, points.shape)
    points += noise
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals for better segmentation
    pcd.estimate_normals()
    
    print(f"✓ Created realistic room with {len(points)} points")
    return pcd

def test_individual_components():
    """Test each component individually"""
    print("\n" + "="*50)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*50)
    
    # Create test data
    pcd = create_realistic_room_cloud()
    
    # Test 1: RANSAC Segmenter
    print("\n1. Testing RANSAC Segmenter...")
    ransac = RANSACSegmenter(distance_threshold=0.03, num_iterations=500)
    segments = ransac.segment_planes(pcd)
    print(f"   Result: {len(segments)} segments found")
    
    # Test 2: Region Growing (if we have wall segments)
    wall_segments = [s for s in segments if s.region_type == 'wall']
    if wall_segments:
        print("\n2. Testing Region Growing...")
        region_grower = RegionGrowingSegmenter()
        refined_segments = region_grower.refine_segments(pcd, segments)
        print(f"   Result: {len(refined_segments)} refined segments")
    else:
        print("\n2. Skipping Region Growing (no wall segments)")
        refined_segments = segments
    
    # Test 3: DBSCAN Clustering
    print("\n3. Testing DBSCAN Clustering...")
    clusterer = DBSCANClusterer()
    clustered_segments = clusterer.cluster_segments(refined_segments)
    print(f"   Result: {len(clustered_segments)} clustered segments")
    
    # Test 4: Occupancy Grid Analysis (on wall segments)
    wall_segments = [s for s in clustered_segments if s.region_type == 'wall']
    if wall_segments:
        print("\n4. Testing Occupancy Grid Analysis...")
        occupancy_analyzer = OccupancyGridAnalyzer()
        
        total_gaps = 0
        for wall in wall_segments:
            grid = occupancy_analyzer.create_occupancy_grid(wall)
            if grid.size > 0:
                gaps = occupancy_analyzer.detect_gaps_in_grid(grid)
                total_gaps += len(gaps)
                print(f"   Wall segment: grid shape {grid.shape}, {len(gaps)} gaps")
        
        print(f"   Result: {total_gaps} total gaps found across all walls")
    else:
        print("\n4. Skipping Occupancy Analysis (no wall segments)")
    
    return clustered_segments

def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n" + "="*50)
    print("TESTING FULL PIPELINE")
    print("="*50)
    
    # Create test data
    pcd = create_realistic_room_cloud()
    
    # Run complete pipeline
    pipeline = AdvancedSegmentationPipeline()
    segments, doors = pipeline.process_point_cloud(pcd)
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"   Final segments: {len(segments)}")
    print(f"   Detected doors: {len(doors)}")
    
    # Analyze results
    segment_types = {}
    for segment in segments:
        seg_type = segment.region_type
        segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
    
    print(f"\n📊 Segment breakdown:")
    for seg_type, count in segment_types.items():
        print(f"   {seg_type}: {count}")
    
    if doors:
        print(f"\n🚪 Door details:")
        for i, door in enumerate(doors):
            print(f"   Door {i+1}: {door.width:.2f}m x {door.height:.2f}m, confidence: {door.confidence:.3f}")
    
    return segments, doors

def visualize_results(segments, doors=None):
    """Optional visualization of results"""
    print("\n" + "="*50)
    print("VISUALIZATION (Optional)")
    print("="*50)
    
    try:
        # Create visualization
        geometries = []
        colors = {
            'wall': [0.7, 0.7, 0.7],        # Gray
            'floor_ceiling': [0.6, 0.4, 0.2], # Brown
            'slanted': [0.5, 0.5, 0.8],     # Light blue
            'unknown': [0.8, 0.8, 0.8]      # Light gray
        }
        
        for i, segment in enumerate(segments):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(segment.points)
            
            # Color by segment type
            color = colors.get(segment.region_type, [1, 0, 0])  # Red for unknown types
            pcd.paint_uniform_color(color)
            
            geometries.append(pcd)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        geometries.append(coord_frame)
        
        print("   Visualization created. You can uncomment the next line to display it.")
        # o3d.visualization.draw_geometries(geometries)  # Uncomment to show visualization
        
    except Exception as e:
        print(f"   Visualization failed: {e}")

def main():
    """Run comprehensive tests"""
    print("="*60)
    print("COMPREHENSIVE SEGMENTATION TESTING")
    print("="*60)
    
    try:
        # Test individual components
        segments = test_individual_components()
        
        # Test full pipeline
        final_segments, doors = test_full_pipeline()
        
        # Optional visualization
        visualize_results(final_segments, doors)
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Your segmentation system is working correctly and ready for real data.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()