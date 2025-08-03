#!/usr/bin/env python3
"""
Debug the import issue in segmentation.py
"""

import sys
import traceback

print("Debugging segmentation.py import...")

# Method 1: Try to execute the file directly
print("\n1. Testing direct execution...")
try:
    exec(open('segmentation.py').read())
    print("✓ Direct execution successful")
except Exception as e:
    print(f"✗ Direct execution failed: {e}")
    traceback.print_exc()

print("\n" + "="*50)

# Method 2: Try importing with detailed error tracking
print("2. Testing import with error tracking...")
try:
    # Clear any existing import
    if 'segmentation' in sys.modules:
        del sys.modules['segmentation']
    
    import segmentation
    print("✓ Import successful")
    
    # Check what got imported
    attrs = [attr for attr in dir(segmentation) if not attr.startswith('_')]
    print(f"Imported attributes: {len(attrs)}")
    for attr in attrs:
        print(f"  {attr}")
        
except Exception as e:
    print(f"✗ Import failed: {e}")
    traceback.print_exc()

print("\n" + "="*50)

# Method 3: Check for specific import errors in the file
print("3. Checking for problematic imports in segmentation.py...")

try:
    # Read the file and check each import
    with open('segmentation.py', 'r') as f:
        lines = f.readlines()
    
    import_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            import_lines.append((i+1, stripped))
    
    print(f"Found {len(import_lines)} import statements:")
    
    for line_num, import_stmt in import_lines:
        print(f"  Line {line_num}: {import_stmt}")
        
        # Test each import individually
        try:
            exec(import_stmt)
            print(f"    ✓ OK")
        except Exception as e:
            print(f"    ✗ FAILED: {e}")

except Exception as e:
    print(f"Error checking imports: {e}")

print("\n" + "="*50)

# Method 4: Try importing individual components step by step
print("4. Testing step-by-step execution...")

try:
    with open('segmentation.py', 'r') as f:
        content = f.read()
    
    # Split into logical sections
    lines = content.split('\n')
    
    # Find where imports end
    import_end = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            import_end = i
    
    print(f"Imports appear to end around line {import_end + 1}")
    
    # Execute imports first
    import_section = '\n'.join(lines[:import_end + 10])  # Include a few extra lines
    print("Executing imports...")
    exec(import_section)
    print("✓ Imports successful")
    
    # Now try to find the first class definition
    class_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('@dataclass') or line.strip().startswith('class '):
            class_start = i
            break
    
    if class_start >= 0:
        print(f"First class/dataclass starts at line {class_start + 1}")
        
        # Try executing up to the first class
        pre_class_section = '\n'.join(lines[:class_start + 5])
        print("Executing up to first class...")
        exec(pre_class_section)
        print("✓ Pre-class section successful")
        
except Exception as e:
    print(f"✗ Step-by-step execution failed: {e}")
    traceback.print_exc()

print("\nDebugging completed.")