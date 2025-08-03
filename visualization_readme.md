# 🏗️ Scan2BIM - Wall Detection Visualization Guide

This guide shows how to visualize the results from your successful wall detection system.

## 📋 Prerequisites

Ensure you have the required packages installed:
```bash
pip install open3d scikit-learn opencv-python scipy numpy
```

## 📁 Results Structure

After running the segmentation pipeline, you'll find these files in the `results/` directory:
```
results/
├── segment_01_wall.ply           # Individual wall segments
├── segment_02_wall.ply
├── segment_XX_wall.ply
├── segment_01_floor_ceiling.ply  # Floor/ceiling segments
├── segment_02_floor_ceiling.ply
├── segment_XX_floor_ceiling.ply
└── building_segmented_working.ply # Combined view (if available)
```

## 🎨 Visualization Commands

### 🧱 View Individual Wall
```bash
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('results/segment_01_wall.ply')])"
```

### 🏠 View Individual Floor/Ceiling
```bash
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('results/segment_01_floor_ceiling.ply')])"
```

### 🎨 View Combined Building (All Segments)
```bash
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('results/building_segmented_working.ply')])"
```

### 🌈 View Multiple Walls Together
```bash
python -c "import open3d as o3d; walls=[o3d.io.read_point_cloud(f'results/segment_{i:02d}_wall.ply') for i in [1,2,3,4,5]]; o3d.visualization.draw_geometries(walls)"
```

### 🏗️ View All Walls vs All Floors (Color-Coded)
```bash
python -c "
import open3d as o3d
import glob
walls = []
floors = []
for f in glob.glob('results/segment_*_wall.ply'):
    w = o3d.io.read_point_cloud(f)
    w.paint_uniform_color([0.8, 0.2, 0.2])  # Red walls
    walls.append(w)
for f in glob.glob('results/segment_*_floor_ceiling.ply'):
    fl = o3d.io.read_point_cloud(f)
    fl.paint_uniform_color([0.4, 0.2, 0.1])  # Brown floors
    floors.append(fl)
o3d.visualization.draw_geometries(walls + floors)
"
```

## 📊 Utility Commands

### Check Available Files
```bash
ls results/
```

### Count Detected Segments
```bash
# Count walls
ls results/segment_*_wall.ply | wc -l

# Count floors/ceilings
ls results/segment_*_floor_ceiling.ply | wc -l
```

## 🎯 Quick Start

**Best starting command** to see your segmented building:
```bash
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('results/building_segmented_working.ply')])"
```

If that file doesn't exist, start with an individual wall:
```bash
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('results/segment_01_wall.ply')])"
```

## 🎮 Viewer Controls

Once the Open3D viewer opens, you can interact with the 3D model:

| Action | Control |
|--------|---------|
| **Rotate** | Mouse drag |
| **Zoom** | Mouse scroll wheel |
| **Pan** | Shift + Mouse drag |
| **Help** | Press `H` key |
| **Reset View** | Press `R` key |
| **Screenshots** | Press `S` key |

## 🎨 Color Coding

- **Gray**: Wall segments
- **Brown**: Floor/ceiling segments  
- **Red**: Walls (in color-coded view)
- **Dark Brown**: Floors/ceilings (in color-coded view)

## 🔧 Troubleshooting

### No files in results/ directory
```bash
# Run the segmentation pipeline first
python cp_loader.py blk-slam-original.ply
```

### "File not found" errors
```bash
# Check what files actually exist
ls results/segment_*.ply
```

### Import errors
```bash
# Reinstall required packages
pip install open3d scikit-learn opencv-python scipy numpy
```

## 🎊 Success!

If you can see your walls and floors properly segmented, congratulations! Your wall detection system is working perfectly. The aggressive clustering approach successfully detected building structures that traditional RANSAC methods couldn't find.

---

**Note**: This visualization system works with the results from the Scan2BIM aggressive wall detection pipeline that successfully segments building point clouds into walls, floors, and ceilings.