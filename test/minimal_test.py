#!/usr/bin/env python3
"""
Minimal test to isolate the import issue
"""

print("Starting minimal test...")

try:
    print("1. Testing basic imports...")
    import numpy as np
    print("✓ NumPy imported")
    
    import open3d as o3d
    print("✓ Open3D imported")
    
    from sklearn.cluster import DBSCAN
    print("✓ scikit-learn imported")
    
    from scipy.spatial.distance import cdist
    print("✓ SciPy imported")
    
    import cv2
    print("✓ OpenCV imported")
    
    print("2. Testing basic functionality...")
    
    # Test numpy
    arr = np.array([1, 2, 3])
    print(f"✓ NumPy array: {arr}")
    
    # Test Open3D basic functionality
    pcd = o3d.geometry.PointCloud()
    print("✓ Open3D PointCloud created")
    
    # Test sklearn
    clustering = DBSCAN(eps=0.1, min_samples=5)
    print("✓ DBSCAN created")
    
    print("3. All basic tests passed!")
    
except Exception as e:
    print(f"✗ Error during basic tests: {e}")
    import traceback
    traceback.print_exc()

print("\nNow testing segmentation.py import...")

try:
    # Try to read the file first
    with open('segmentation.py', 'r') as f:
        content = f.read()
    print(f"✓ segmentation.py file readable ({len(content)} characters)")
    
    # Try to compile it
    compile(content, 'segmentation.py', 'exec')
    print("✓ segmentation.py syntax is valid")
    
    # Try to import it
    print("Attempting import...")
    import segmentation
    print("✓ segmentation.py imported successfully!")
    
except FileNotFoundError:
    print("✗ segmentation.py file not found")
except SyntaxError as e:
    print(f"✗ Syntax error in segmentation.py: {e}")
except Exception as e:
    print(f"✗ Error importing segmentation.py: {e}")
    import traceback
    traceback.print_exc()

print("Minimal test completed.")