#!/usr/bin/env python3
"""
Modified segmentation script that can accept external point cloud data
"""

import sys
import numpy as np
import open3d as o3d

# Execute the original segmentation.py to get all the classes
exec(open('segmentation.py').read().replace('if __name__ == "__main__":', 'if False:'))

def test_with_file(filename):
    """Test segmentation with a specific point cloud file"""
    print(f"Loading point cloud from {filename}...")
    
    try:
        pcd = o3d.io.read_point_cloud(filename)
        
        if len(pcd.points) == 0:
            print("✗ No points loaded from file")
            return
        
        print(f"✓ Loaded {len(pcd.points)} points")
        
        # Run the pipeline
        pipeline = AdvancedSegmentationPipeline()
        segments, doors = pipeline.process_point_cloud(pcd)
        
        print(f"\nResults:")
        print(f"  Segments: {len(segments)}")
        print(f"  Doors: {len(doors)}")
        
        # Show segment breakdown
        segment_types = {}
        for segment in segments:
            seg_type = segment.region_type
            segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
        
        for seg_type, count in segment_types.items():
            print(f"  {seg_type}: {count}")
        
        if doors:
            print("\nDetected doors:")
            for i, door in enumerate(doors):
                print(f"  Door {i+1}: {door.width:.2f}m x {door.height:.2f}m (confidence: {door.confidence:.3f})")
        
        return segments, doors
        
    except Exception as e:
        print(f"✗ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        test_with_file(filename)
    else:
        print("Usage: python modified_segmentation.py <point_cloud_file>")
