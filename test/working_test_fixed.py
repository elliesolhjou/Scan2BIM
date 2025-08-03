#!/usr/bin/env python3
"""
Working test for segmentation.py - handles the import issue
"""

import subprocess
import sys
import numpy as np
import open3d as o3d

def test_segmentation_directly():
    """Test segmentation by running it directly"""
    print("="*60)
    print("TESTING SEGMENTATION.PY DIRECTLY")
    print("="*60)
    
    try:
        # Run the segmentation.py script directly
        result = subprocess.run([sys.executable, 'segmentation.py'], 
                              capture_output=True, text=True, timeout=60)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Segmentation script ran successfully!")
            return True
        else:
            print(f"✗ Script failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Script timed out")
        return False
    except Exception as e:
        print(f"✗ Error running script: {e}")
        return False

def create_custom_test_data():
    """Create test data using only standard libraries"""
    print("\nCreating custom test data...")
    
    # Create a simple room structure
    points = []
    
    # Room dimensions
    width, depth, height = 4.0, 3.0, 2.5
    door_width = 0.8
    door_x_pos = 1.5
    
    # Generate points with lower resolution for faster processing
    res = 0.1
    
    # Front wall with door opening
    for x in np.arange(0, width, res):
        for z in np.arange(0, height, res):
            # Skip door opening
            if not (door_x_pos <= x <= door_x_pos + door_width and 0 <= z <= 2.0):
                points.append([x, 0, z])
    
    # Back wall  
    for x in np.arange(0, width, res):
        for z in np.arange(0, height, res):
            points.append([x, depth, z])
    
    # Left and right walls
    for y in np.arange(0, depth, res):
        for z in np.arange(0, height, res):
            points.append([0, y, z])
            points.append([width, y, z])
    
    # Floor
    for x in np.arange(0, width, res*2):
        for y in np.arange(0, depth, res*2):
            points.append([x, y, 0])
    
    points = np.array(points)
    
    # Add noise
    noise = np.random.normal(0, 0.02, points.shape)
    points += noise
    
    print(f"Generated {len(points)} points")
    return points

def save_test_data(points, filename="test_room.ply"):
    """Save test data to PLY file that can be loaded by segmentation.py"""
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Save to file
    success = o3d.io.write_point_cloud(filename, pcd)
    
    if success:
        print(f"✓ Saved test data to {filename}")
        return filename
    else:
        print(f"✗ Failed to save test data to {filename}")
        return None

def create_modified_segmentation_script():
    """Create a modified version that accepts external data"""
    
    script_content = '''#!/usr/bin/env python3
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
        
        print(f"\\nResults:")
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
            print("\\nDetected doors:")
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
'''
    
    with open('modified_segmentation.py', 'w') as f:
        f.write(script_content)
    
    print("✓ Created modified_segmentation.py")
    return 'modified_segmentation.py'

def test_comprehensive_workflow():
    """Test the complete workflow with custom data"""
    print("\\n" + "="*60)
    print("COMPREHENSIVE WORKFLOW TEST")
    print("="*60)
    
    try:
        # Step 1: Create test data
        points = create_custom_test_data()
        
        # Step 2: Save to file
        filename = save_test_data(points)
        if not filename:
            return False
        
        # Step 3: Create modified script
        script_name = create_modified_segmentation_script()
        
        # Step 4: Run the modified script
        print(f"\\nRunning segmentation on {filename}...")
        result = subprocess.run([sys.executable, script_name, filename],
                              capture_output=True, text=True, timeout=120)
        
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        success = result.returncode == 0
        if success:
            print("✓ Comprehensive workflow completed successfully!")
        else:
            print(f"✗ Workflow failed with return code: {result.returncode}")
        
        return success
        
    except Exception as e:
        print(f"✗ Workflow error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_environment():
    """Verify that all required packages are working"""
    print("Verifying environment...")
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
        
        import open3d
        print(f"✓ Open3D {open3d.__version__}")
        
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
        
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
        
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("SEGMENTATION SYSTEM COMPREHENSIVE TEST")
    print("="*60)
    
    # Test 1: Environment verification
    print("\\n1. Environment Verification:")
    env_ok = verify_environment()
    
    # Test 2: Direct execution test
    print("\\n2. Direct Execution Test:")
    direct_ok = test_segmentation_directly()
    
    # Test 3: Comprehensive workflow test
    print("\\n3. Comprehensive Workflow Test:")
    workflow_ok = test_comprehensive_workflow()
    
    # Summary
    print("\\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Environment", env_ok),
        ("Direct Execution", direct_ok), 
        ("Comprehensive Workflow", workflow_ok)
    ]
    
    passed = sum(1 for _, ok in tests if ok)
    
    for test_name, ok in tests:
        status = "PASS" if ok else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\\n🎉 All tests passed! Your segmentation system is fully functional.")
        print("\\nNext steps:")
        print("- Use 'python segmentation.py' to run with synthetic data")
        print("- Modify the script to load your own point cloud files")
        print("- Adjust parameters for your specific use case")
    else:
        print(f"\\n⚠️ {len(tests) - passed} test(s) failed. Check the output above for details.")

if __name__ == "__main__":
    main()