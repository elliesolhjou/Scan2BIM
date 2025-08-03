#!/usr/bin/env python3
"""
Inspect what's actually available in segmentation.py
"""

print("Inspecting segmentation.py contents...")

try:
    import segmentation
    print("✓ segmentation.py imported successfully")
    
    print("\nAvailable classes and functions:")
    available_items = []
    
    for name in dir(segmentation):
        if not name.startswith('_'):
            obj = getattr(segmentation, name)
            obj_type = type(obj).__name__
            available_items.append((name, obj_type))
            print(f"  {name}: {obj_type}")
    
    print(f"\nTotal available items: {len(available_items)}")
    
    # Check for the specific classes we need
    required_classes = [
        'RANSACSegmenter',
        'RegionGrowingSegmenter', 
        'DBSCANClusterer',
        'OccupancyGridAnalyzer',
        'GapAnalyzer',
        'DoorDetectionValidator',
        'AdvancedSegmentationPipeline',
        'SegmentedRegion',
        'DetectedDoor'
    ]
    
    print(f"\nChecking for required classes:")
    missing_classes = []
    
    for class_name in required_classes:
        if hasattr(segmentation, class_name):
            print(f"  ✓ {class_name}: Found")
        else:
            print(f"  ✗ {class_name}: Missing")
            missing_classes.append(class_name)
    
    if missing_classes:
        print(f"\n⚠️  Missing classes: {missing_classes}")
        print("This might be due to:")
        print("  1. Syntax errors in the class definitions")
        print("  2. Import errors within the segmentation.py file")
        print("  3. Indentation issues")
    else:
        print(f"\n✅ All required classes are available!")
        
except Exception as e:
    print(f"✗ Error importing segmentation.py: {e}")
    import traceback
    traceback.print_exc()

# Also try to read the file and check for class definitions
print(f"\nChecking file content for class definitions...")
try:
    with open('segmentation.py', 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    class_lines = [line for line in lines if line.strip().startswith('class ')]
    
    print(f"Found {len(class_lines)} class definitions:")
    for line in class_lines:
        print(f"  {line.strip()}")
        
except Exception as e:
    print(f"Error reading file: {e}")