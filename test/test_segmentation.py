#!/usr/bin/env python3
"""
Test script for segmentation.py to verify all components work correctly
"""

import sys
import traceback
import numpy as np
import open3d as o3d

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        from segmentation import (
            RANSACSegmenter, RegionGrowingSegmenter, DBSCANClusterer,
            OccupancyGridAnalyzer, GapAnalyzer, DoorDetectionValidator,
            AdvancedSegmentationPipeline, SegmentedRegion, DetectedDoor
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def create_simple_test_cloud():
    """Create a simple test point cloud"""
    print("Creating simple test point cloud...")
    
    # Create a simple wall
    points = []
    
    # Simple wall (100 points)
    for i in range(10):
        for j in range(10):
            points.append([i * 0.1, 0, j * 0.1])  # Wall at y=0
    
    points = np.array(points)
    
    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    print(f"✓ Created test cloud with {len(points)} points")
    return point_cloud

def test_ransac_segmenter():
    """Test RANSAC segmenter component"""
    print("Testing RANSAC segmenter...")
    try:
        from segmentation import RANSACSegmenter
        
        segmenter = RANSACSegmenter()
        test_cloud = create_simple_test_cloud()
        
        segments = segmenter.segment_planes(test_cloud)
        print(f"✓ RANSAC segmentation completed: {len(segments)} segments found")
        return True
    except Exception as e:
        print(f"✗ RANSAC segmenter error: {e}")
        traceback.print_exc()
        return False

def test_pipeline_components():
    """Test individual pipeline components"""
    print("Testing pipeline components...")
    try:
        from segmentation import (
            RegionGrowingSegmenter, DBSCANClusterer, 
            OccupancyGridAnalyzer, GapAnalyzer, DoorDetectionValidator
        )
        
        # Test instantiation
        region_grower = RegionGrowingSegmenter()
        dbscan_clusterer = DBSCANClusterer()
        occupancy_analyzer = OccupancyGridAnalyzer()
        gap_analyzer = GapAnalyzer()
        door_validator = DoorDetectionValidator()
        
        print("✓ All pipeline components instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Pipeline components error: {e}")
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test the complete pipeline with simple data"""
    print("Testing full pipeline...")
    try:
        from segmentation import AdvancedSegmentationPipeline
        
        pipeline = AdvancedSegmentationPipeline()
        test_cloud = create_simple_test_cloud()
        
        print("Running pipeline...")
        segments, doors = pipeline.process_point_cloud(test_cloud)
        
        print(f"✓ Pipeline completed: {len(segments)} segments, {len(doors)} doors")
        return True
    except Exception as e:
        print(f"✗ Full pipeline error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("SEGMENTATION.PY TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("RANSAC Segmenter Test", test_ransac_segmenter),
        ("Pipeline Components Test", test_pipeline_components),
        ("Full Pipeline Test", test_full_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your segmentation system is ready to use.")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the errors above.")

if __name__ == "__main__":
    main()