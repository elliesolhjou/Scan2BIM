#!/usr/bin/env python3
"""
Room Visualization and Output System
Save and visualize room separation results in 3D
"""

import numpy as np
import open3d as o3d
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging

# Import our systems
from room_separation import Room, RoomSeparationPipeline
from segmentation import CompleteBuildingSegmentation

logger = logging.getLogger(__name__)

class RoomVisualizationSystem:
    """Complete room visualization and output system"""
    
    def __init__(self):
        self.output_dir = 'room_results'
        self.colors = {
            'wall': [0.7, 0.7, 0.7],           # Gray
            'floor': [0.6, 0.4, 0.2],          # Brown
            'ceiling': [0.9, 0.9, 0.9],        # Light gray
            'low_furniture': [0.2, 0.8, 0.2],  # Green
            'medium_furniture': [0.2, 0.2, 0.8], # Blue
            'high_furniture': [0.8, 0.2, 0.2], # Red
            'room_1': [1.0, 0.6, 0.6],         # Light red
            'room_2': [0.6, 1.0, 0.6],         # Light green
            'room_3': [0.6, 0.6, 1.0],         # Light blue
            'room_4': [1.0, 1.0, 0.6],         # Light yellow
            'room_5': [1.0, 0.6, 1.0],         # Light magenta
        }
    
    def save_room_results(self, rooms: List[Room], all_segments: List, 
                         analysis_report: Dict[str, Any]):
        """Save complete room analysis results"""
        
        print(f"\n💾 SAVING ROOM VISUALIZATION RESULTS...")
        print("=" * 50)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Save individual room point clouds
        self._save_individual_rooms(rooms)
        
        # 2. Save rooms with color coding
        self._save_colored_rooms(rooms)
        
        # 3. Save furniture by room
        self._save_furniture_by_room(rooms)
        
        # 4. Save complete building with room colors
        self._save_complete_building_with_rooms(rooms, all_segments)
        
        # 5. Save analysis reports
        self._save_analysis_reports(rooms, analysis_report)
        
        # 6. Create viewing instructions
        self._create_viewing_instructions()
        
        print(f"✅ All results saved to '{self.output_dir}/' directory")
        print(f"📁 {len(os.listdir(self.output_dir))} files created")
    
    def _save_individual_rooms(self, rooms: List[Room]):
        """Save each room as a separate point cloud file"""
        
        print(f"💾 Saving individual rooms...")
        
        for i, room in enumerate(rooms):
            room_points = []
            room_colors = []
            
            # Add floor points
            if room.floor_area:
                floor_points = room.floor_area.points
                floor_colors = np.tile(self.colors['floor'], (len(floor_points), 1))
                room_points.extend(floor_points)
                room_colors.extend(floor_colors)
            
            # Add wall points
            for wall in room.boundary_walls:
                wall_points = wall.points
                wall_colors = np.tile(self.colors['wall'], (len(wall_points), 1))
                room_points.extend(wall_points)
                room_colors.extend(wall_colors)
            
            # Add furniture points with type-based colors
            for furniture in room.furniture:
                furn_points = furniture.points
                furn_type = furniture.region_type
                color = self.colors.get(furn_type, [0.5, 0.5, 0.5])
                furn_colors = np.tile(color, (len(furn_points), 1))
                room_points.extend(furn_points)
                room_colors.extend(furn_colors)
            
            if room_points:
                # Create point cloud
                room_cloud = o3d.geometry.PointCloud()
                room_cloud.points = o3d.utility.Vector3dVector(room_points)
                room_cloud.colors = o3d.utility.Vector3dVector(room_colors)
                
                # Save to file
                filename = f"{self.output_dir}/room_{i+1}_{room.room_type}.ply"
                o3d.io.write_point_cloud(filename, room_cloud)
                
                print(f"   Room {i+1} ({room.room_type}): {len(room_points):,} points → {filename}")
    
    def _save_colored_rooms(self, rooms: List[Room]):
        """Save rooms with distinct room colors"""
        
        print(f"🎨 Saving color-coded rooms...")
        
        all_points = []
        all_colors = []
        
        for i, room in enumerate(rooms):
            room_color = self.colors.get(f'room_{i+1}', [0.8, 0.8, 0.8])
            
            # Collect all points from this room
            room_all_points = []
            
            # Add floor
            if room.floor_area:
                room_all_points.extend(room.floor_area.points)
            
            # Add walls
            for wall in room.boundary_walls:
                room_all_points.extend(wall.points)
            
            # Add furniture
            for furniture in room.furniture:
                room_all_points.extend(furniture.points)
            
            if room_all_points:
                room_colors = np.tile(room_color, (len(room_all_points), 1))
                all_points.extend(room_all_points)
                all_colors.extend(room_colors)
        
        if all_points:
            # Create combined room cloud
            rooms_cloud = o3d.geometry.PointCloud()
            rooms_cloud.points = o3d.utility.Vector3dVector(all_points)
            rooms_cloud.colors = o3d.utility.Vector3dVector(all_colors)
            
            filename = f"{self.output_dir}/all_rooms_colored.ply"
            o3d.io.write_point_cloud(filename, rooms_cloud)
            
            print(f"   All rooms with distinct colors: {len(all_points):,} points → {filename}")
    
    def _save_furniture_by_room(self, rooms: List[Room]):
        """Save furniture grouped by room"""
        
        print(f"🪑 Saving furniture by room...")
        
        for i, room in enumerate(rooms):
            if not room.furniture:
                continue
            
            furniture_points = []
            furniture_colors = []
            
            for furniture in room.furniture:
                furn_points = furniture.points
                furn_type = furniture.region_type
                color = self.colors.get(furn_type, [0.5, 0.5, 0.5])
                furn_colors = np.tile(color, (len(furn_points), 1))
                
                furniture_points.extend(furn_points)
                furniture_colors.extend(furn_colors)
            
            if furniture_points:
                furniture_cloud = o3d.geometry.PointCloud()
                furniture_cloud.points = o3d.utility.Vector3dVector(furniture_points)
                furniture_cloud.colors = o3d.utility.Vector3dVector(furniture_colors)
                
                filename = f"{self.output_dir}/room_{i+1}_furniture_only.ply"
                o3d.io.write_point_cloud(filename, furniture_cloud)
                
                print(f"   Room {i+1} furniture: {len(furniture_points):,} points → {filename}")
    
    def _save_complete_building_with_rooms(self, rooms: List[Room], all_segments: List):
        """Save complete building with room-based coloring"""
        
        print(f"🏢 Saving complete building...")
        
        all_points = []
        all_colors = []
        
        # Create a mapping of points to rooms
        point_to_room = {}
        
        for i, room in enumerate(rooms):
            room_color = self.colors.get(f'room_{i+1}', [0.8, 0.8, 0.8])
            
            # Map all room points
            if room.floor_area:
                for point in room.floor_area.points:
                    point_to_room[tuple(point)] = room_color
            
            for wall in room.boundary_walls:
                for point in wall.points:
                    point_to_room[tuple(point)] = room_color
            
            for furniture in room.furniture:
                for point in furniture.points:
                    point_to_room[tuple(point)] = room_color
        
        # Process all segments
        for segment in all_segments:
            segment_points = segment.points
            
            # Determine colors
            if segment.region_type == 'wall':
                color = self.colors['wall']
            elif segment.region_type == 'floor_ceiling':
                if segment.centroid[2] < 1.0:
                    color = self.colors['floor']
                else:
                    color = self.colors['ceiling']
            elif segment.region_type == 'ceiling':
                color = self.colors['ceiling']
            elif 'furniture' in segment.region_type:
                color = self.colors.get(segment.region_type, [0.5, 0.5, 0.5])
            else:
                color = [0.5, 0.5, 0.5]  # Default gray
            
            # Override with room color if point belongs to a room
            segment_colors = []
            for point in segment_points:
                point_key = tuple(point)
                if point_key in point_to_room:
                    segment_colors.append(point_to_room[point_key])
                else:
                    segment_colors.append(color)
            
            all_points.extend(segment_points)
            all_colors.extend(segment_colors)
        
        if all_points:
            complete_cloud = o3d.geometry.PointCloud()
            complete_cloud.points = o3d.utility.Vector3dVector(all_points)
            complete_cloud.colors = o3d.utility.Vector3dVector(all_colors)
            
            filename = f"{self.output_dir}/complete_building_with_rooms.ply"
            o3d.io.write_point_cloud(filename, complete_cloud)
            
            print(f"   Complete building: {len(all_points):,} points → {filename}")
    
    def _save_analysis_reports(self, rooms: List[Room], analysis_report: Dict[str, Any]):
        """Save detailed text reports"""
        
        print(f"📋 Saving analysis reports...")
        
        # Detailed room report
        with open(f"{self.output_dir}/detailed_room_analysis.txt", 'w') as f:
            f.write("DETAILED ROOM SEPARATION ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            f.write("BUILDING OVERVIEW:\n")
            f.write(f"Total rooms detected: {analysis_report['total_rooms']}\n")
            f.write(f"Total floor area: {analysis_report['total_area']:.1f} m²\n")
            f.write(f"Furniture items assigned: {analysis_report['total_furniture_assigned']}\n")
            f.write(f"Wall connections found: {analysis_report['wall_connections']}\n")
            f.write(f"Wall groups formed: {analysis_report['wall_groups']}\n\n")
            
            # Room details
            f.write("INDIVIDUAL ROOM DETAILS:\n")
            for i, room in enumerate(rooms):
                f.write(f"\nRoom {i+1} ({room.room_type}):\n")
                f.write(f"  Area: {room.area:.1f} m²\n")
                f.write(f"  Center: ({room.center_point[0]:.1f}, {room.center_point[1]:.1f}, {room.center_point[2]:.1f})\n")
                f.write(f"  Boundary walls: {len(room.boundary_walls)}\n")
                f.write(f"  Total furniture: {len(room.furniture)}\n")
                
                # Furniture breakdown
                furniture_types = {}
                for furniture in room.furniture:
                    ftype = furniture.region_type.replace('_furniture', '')
                    furniture_types[ftype] = furniture_types.get(ftype, 0) + 1
                
                if furniture_types:
                    f.write(f"  Furniture breakdown:\n")
                    for ftype, count in furniture_types.items():
                        f.write(f"    {ftype}: {count} items\n")
        
        # CSV summary
        with open(f"{self.output_dir}/room_summary.csv", 'w') as f:
            f.write("Room_ID,Room_Type,Area_m2,Center_X,Center_Y,Center_Z,Walls,Total_Furniture,Low_Furniture,Medium_Furniture,High_Furniture\n")
            
            for i, room in enumerate(rooms):
                furniture_counts = {'low': 0, 'medium': 0, 'high': 0}
                for furniture in room.furniture:
                    if 'low_furniture' in furniture.region_type:
                        furniture_counts['low'] += 1
                    elif 'medium_furniture' in furniture.region_type:
                        furniture_counts['medium'] += 1
                    elif 'high_furniture' in furniture.region_type:
                        furniture_counts['high'] += 1
                
                f.write(f"{i+1},{room.room_type},{room.area:.1f},"
                       f"{room.center_point[0]:.1f},{room.center_point[1]:.1f},{room.center_point[2]:.1f},"
                       f"{len(room.boundary_walls)},{len(room.furniture)},"
                       f"{furniture_counts['low']},{furniture_counts['medium']},{furniture_counts['high']}\n")
        
        print(f"   Text report → {self.output_dir}/detailed_room_analysis.txt")
        print(f"   CSV summary → {self.output_dir}/room_summary.csv")
    
    def _create_viewing_instructions(self):
        """Create instructions for viewing the results"""
        
        instructions = """
# ROOM SEPARATION VISUALIZATION GUIDE

## 📁 Generated Files:

### Individual Rooms:
- `room_1_[type].ply` - Individual room with furniture
- `room_2_[type].ply` - Individual room with furniture
- etc.

### Combined Views:
- `all_rooms_colored.ply` - All rooms with distinct colors
- `complete_building_with_rooms.ply` - Complete building with room assignments

### Furniture Only:
- `room_1_furniture_only.ply` - Just furniture from room 1
- `room_2_furniture_only.ply` - Just furniture from room 2
- etc.

### Reports:
- `detailed_room_analysis.txt` - Complete text analysis
- `room_summary.csv` - Spreadsheet-compatible summary

## 🎨 Color Coding:

### Room Colors:
- Room 1: Light Red
- Room 2: Light Green  
- Room 3: Light Blue
- Room 4: Light Yellow
- Room 5: Light Magenta

### Furniture Colors:
- Low Furniture (tables, chairs): Green
- Medium Furniture (desks, counters): Blue
- High Furniture (cabinets, shelves): Red
- Walls: Gray
- Floors: Brown
- Ceilings: Light Gray

## 👀 How to View:

### Best Starting Views:
1. **Overview**: `complete_building_with_rooms.ply`
2. **Room Separation**: `all_rooms_colored.ply`
3. **Individual Rooms**: `room_1_[type].ply`, etc.

### Command Line Viewing:
```bash
# View complete building
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('room_results/complete_building_with_rooms.ply')])"

# View room separation
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('room_results/all_rooms_colored.ply')])"

# View individual room
python -c "import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('room_results/room_1_meeting_room.ply')])"

"""


def main():
    """Test the visualization system"""
    
    print("💾 TESTING ROOM VISUALIZATION SYSTEM")
    print("=" * 60)
    
    # Load point cloud and run complete pipeline
    test_file = 'building_segmented_colored.ply'
    print(f"Looking for test file: {test_file}")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.ply'):
                print(f"  - {file}")
        return
    
    try:
        point_cloud = o3d.io.read_point_cloud(test_file)
        print(f"✅ Loaded {len(point_cloud.points):,} points")
        
        # Run Phase 1: Semantic Classification
        print("\n🏷️ Running Phase 1: Semantic Classification...")
        segmentation_pipeline = CompleteBuildingSegmentation()
        segments, doors = segmentation_pipeline.process_point_cloud(point_cloud)
        print(f"✅ Phase 1 complete: {len(segments)} segments")
        
        # Run Phase 2: Room Separation
        print("\n🏠 Running Phase 2: Room Separation...")
        room_pipeline = RoomSeparationPipeline()
        rooms, analysis_report = room_pipeline.process_semantic_segments(segments)
        print(f"✅ Phase 2 complete: {len(rooms)} rooms")
        
        # Run Visualization and Output
        print("\n💾 Running Visualization and Output...")
        viz_system = RoomVisualizationSystem()
        viz_system.save_room_results(rooms, segments, analysis_report)
        
        print(f"\n🎉 VISUALIZATION COMPLETE!")
        print(f"✅ All results saved to 'room_results/' directory")
        print(f"📖 Check 'room_results/HOW_TO_VIEW.md' for viewing instructions")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()