#!/usr/bin/env python3
"""
Phase 2: Room Separation and Space Analysis
Building on the semantic classification results from Phase 1
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging
import os

# Import our existing segmentation system
from segmentation import SegmentedRegion, CompleteBuildingSegmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Room:
    """Represents a detected room space"""
    room_id: int
    boundary_walls: List[SegmentedRegion]
    furniture: List[SegmentedRegion]
    floor_area: Optional[SegmentedRegion]
    ceiling_area: Optional[SegmentedRegion]
    center_point: np.ndarray
    area: float
    room_type: str  # 'office', 'corridor', 'meeting_room', 'unknown'
    adjacent_rooms: List[int]  # IDs of connected rooms

@dataclass
class RoomConnection:
    """Represents connection between two rooms"""
    room1_id: int
    room2_id: int
    connection_type: str  # 'door', 'opening', 'corridor'
    connection_point: np.ndarray
    width: float

class WallConnectivityAnalyzer:
    """Analyze how walls connect to form room boundaries"""
    
    def __init__(self):
        self.connection_threshold = 0.5  # 50cm max gap between connected walls
        self.parallel_threshold = 0.2    # How parallel walls need to be
        self.perpendicular_threshold = 0.8  # How perpendicular walls need to be
    
    def analyze_wall_connectivity(self, wall_segments: List[SegmentedRegion]) -> Dict[str, Any]:
        """Analyze how walls connect to each other"""
        
        logger.info(f"🔗 Analyzing connectivity of {len(wall_segments)} walls...")
        
        if len(wall_segments) < 2:
            logger.warning("Need at least 2 walls for connectivity analysis")
            return {'connections': [], 'wall_groups': []}
        
        connections = []
        
        # Check each pair of walls
        for i, wall1 in enumerate(wall_segments):
            for j, wall2 in enumerate(wall_segments[i+1:], i+1):
                connection = self._analyze_wall_pair(wall1, wall2, i, j)
                if connection:
                    connections.append(connection)
        
        # Group connected walls
        wall_groups = self._group_connected_walls(wall_segments, connections)
        
        logger.info(f"Found {len(connections)} wall connections")
        logger.info(f"Formed {len(wall_groups)} wall groups")
        
        return {
            'connections': connections,
            'wall_groups': wall_groups,
            'connectivity_matrix': self._build_connectivity_matrix(len(wall_segments), connections)
        }
    
    def _analyze_wall_pair(self, wall1: SegmentedRegion, wall2: SegmentedRegion, 
                          id1: int, id2: int) -> Optional[Dict[str, Any]]:
        """Analyze connection between two walls"""
        
        # Get wall endpoints (simplified - use bounding box)
        points1 = wall1.points
        points2 = wall2.points
        
        # Calculate bounding boxes
        min1, max1 = np.min(points1, axis=0), np.max(points1, axis=0)
        min2, max2 = np.min(points2, axis=0), np.max(points2, axis=0)
        
        # Calculate minimum distance between walls
        min_distance = self._calculate_wall_distance(points1, points2)
        
        if min_distance > self.connection_threshold:
            return None  # Too far apart
        
        # Calculate angle between walls
        normal1 = wall1.normal[:2]  # Only X,Y components
        normal2 = wall2.normal[:2]
        
        # Normalize normals
        if np.linalg.norm(normal1) > 0:
            normal1 = normal1 / np.linalg.norm(normal1)
        if np.linalg.norm(normal2) > 0:
            normal2 = normal2 / np.linalg.norm(normal2)
        
        dot_product = np.abs(np.dot(normal1, normal2))
        
        # Determine connection type
        if dot_product > (1 - self.parallel_threshold):
            connection_type = 'parallel'
        elif dot_product < self.perpendicular_threshold:
            connection_type = 'perpendicular'
        else:
            connection_type = 'angled'
        
        return {
            'wall1_id': id1,
            'wall2_id': id2,
            'distance': min_distance,
            'connection_type': connection_type,
            'angle': np.arccos(np.clip(dot_product, 0, 1)) * 180 / np.pi
        }
    
    def _calculate_wall_distance(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """Calculate minimum distance between two wall point clouds"""
        
        # Sample points to speed up calculation
        sample_size = min(1000, len(points1), len(points2))
        
        if len(points1) > sample_size:
            indices1 = np.random.choice(len(points1), sample_size, replace=False)
            sample1 = points1[indices1]
        else:
            sample1 = points1
            
        if len(points2) > sample_size:
            indices2 = np.random.choice(len(points2), sample_size, replace=False)
            sample2 = points2[indices2]
        else:
            sample2 = points2
        
        # Calculate pairwise distances
        distances = cdist(sample1, sample2)
        return np.min(distances)
    
    def _group_connected_walls(self, walls: List[SegmentedRegion], 
                             connections: List[Dict[str, Any]]) -> List[List[int]]:
        """Group walls that are connected to each other"""
        
        # Build adjacency list
        adjacency = {i: set() for i in range(len(walls))}
        
        for conn in connections:
            wall1_id = conn['wall1_id']
            wall2_id = conn['wall2_id']
            adjacency[wall1_id].add(wall2_id)
            adjacency[wall2_id].add(wall1_id)
        
        # Find connected components
        visited = set()
        groups = []
        
        for i in range(len(walls)):
            if i not in visited:
                group = []
                self._dfs_visit(i, adjacency, visited, group)
                if group:
                    groups.append(group)
        
        return groups
    
    def _dfs_visit(self, node: int, adjacency: Dict[int, set], 
                  visited: set, group: List[int]):
        """Depth-first search to find connected components"""
        visited.add(node)
        group.append(node)
        
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                self._dfs_visit(neighbor, adjacency, visited, group)
    
    def _build_connectivity_matrix(self, num_walls: int, 
                                 connections: List[Dict[str, Any]]) -> np.ndarray:
        """Build connectivity matrix between walls"""
        
        matrix = np.zeros((num_walls, num_walls))
        
        for conn in connections:
            i, j = conn['wall1_id'], conn['wall2_id']
            matrix[i, j] = 1
            matrix[j, i] = 1
        
        return matrix

class RoomDetector:
    """Detect individual rooms based on wall boundaries and furniture clustering"""
    
    def __init__(self):
        self.furniture_room_threshold = 10.0  # 2m max distance for furniture to belong to room
        self.min_room_area = 4.0            # 4m² minimum room area
        self.corridor_width_threshold = 3.0   # Corridors are typically < 3m wide
    
    def detect_rooms(self, walls: List[SegmentedRegion], 
                    furniture: List[SegmentedRegion],
                    floors: List[SegmentedRegion],
                    ceilings: List[SegmentedRegion],
                    wall_connectivity: Dict[str, Any]) -> List[Room]:
        """Detect rooms using wall boundaries and furniture clustering"""
        
        logger.info("🏠 Starting room detection...")
        
        # Step 1: Create room spaces based on floor areas
        room_candidates = self._create_room_candidates_from_floors(floors)
        
        # Step 2: Assign walls to rooms
        rooms_with_walls = self._assign_walls_to_rooms(room_candidates, walls)
        
        # Step 3: Assign furniture to rooms
        rooms_with_furniture = self._assign_furniture_to_rooms(rooms_with_walls, furniture)
        
        # Step 4: Classify room types
        classified_rooms = self._classify_room_types(rooms_with_furniture)
        
        # Step 5: Find room connections
        final_rooms = self._find_room_connections(classified_rooms, wall_connectivity)
        
        logger.info(f"Detected {len(final_rooms)} rooms")
        return final_rooms
    
    def _create_room_candidates_from_floors(self, floors: List[SegmentedRegion]) -> List[Room]:
        """Create initial room candidates based on floor segments"""
        
        room_candidates = []
        
        for i, floor in enumerate(floors):
            # Calculate floor properties
            floor_points = floor.points
            center = np.mean(floor_points, axis=0)
            
            # Estimate area (simplified)
            min_coords = np.min(floor_points, axis=0)
            max_coords = np.max(floor_points, axis=0)
            area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
            
            if area >= self.min_room_area:
                room = Room(
                    room_id=i,
                    boundary_walls=[],
                    furniture=[],
                    floor_area=floor,
                    ceiling_area=None,
                    center_point=center,
                    area=area,
                    room_type='unknown',
                    adjacent_rooms=[]
                )
                room_candidates.append(room)
        
        logger.info(f"Created {len(room_candidates)} room candidates from floors")
        return room_candidates
    
    def _assign_walls_to_rooms(self, rooms: List[Room], 
                             walls: List[SegmentedRegion]) -> List[Room]:
        """Assign walls to their closest rooms"""
        
        for wall in walls:
            wall_center = wall.centroid
            closest_room = None
            min_distance = float('inf')
            
            for room in rooms:
                room_center = room.center_point
                distance = np.linalg.norm(wall_center - room_center)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_room = room
            
            if closest_room and min_distance < 5.0:  # 5m max distance
                closest_room.boundary_walls.append(wall)
        
        return rooms
    
    def _assign_furniture_to_rooms(self, rooms: List[Room], 
                                 furniture: List[SegmentedRegion]) -> List[Room]:
        """Assign furniture to their closest rooms"""
        
        for furniture_item in furniture:
            furniture_center = furniture_item.centroid
            closest_room = None
            min_distance = float('inf')
            
            for room in rooms:
                room_center = room.center_point
                distance = np.linalg.norm(furniture_center - room_center)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_room = room
            
            if closest_room and min_distance < self.furniture_room_threshold:
                closest_room.furniture.append(furniture_item)
        
        return rooms
    
    def _classify_room_types(self, rooms: List[Room]) -> List[Room]:
        """Classify room types based on furniture and geometry"""
        
        for room in rooms:
            room.room_type = self._determine_room_type(room)
        
        return rooms
    
    def _determine_room_type(self, room: Room) -> str:
        """Determine room type based on characteristics"""
        
        # Analyze furniture types
        furniture_types = {}
        for furniture in room.furniture:
            ftype = furniture.region_type
            furniture_types[ftype] = furniture_types.get(ftype, 0) + 1
        
        # Simple classification rules
        total_furniture = len(room.furniture)
        
        if total_furniture == 0:
            if room.area > 50:
                return 'corridor'
            else:
                return 'empty_room'
        
        # Check for meeting room characteristics
        if 'medium_furniture' in furniture_types and furniture_types['medium_furniture'] >= 2:
            return 'meeting_room'
        
        # Check for office characteristics
        if 'medium_furniture' in furniture_types and 'high_furniture' in furniture_types:
            return 'office'
        
        # Check for corridor characteristics
        min_coords = np.min(room.floor_area.points, axis=0)
        max_coords = np.max(room.floor_area.points, axis=0)
        width = min(max_coords[0] - min_coords[0], max_coords[1] - min_coords[1])
        
        if width < self.corridor_width_threshold and room.area > 10:
            return 'corridor'
        
        return 'general_room'
    
    def _find_room_connections(self, rooms: List[Room], 
                             wall_connectivity: Dict[str, Any]) -> List[Room]:
        """Find connections between rooms"""
        
        # For now, return rooms as-is
        # This would analyze wall gaps, doors, and openings
        return rooms
    

class RoomSeparationPipeline:
    """Complete Room Separation Pipeline - integrates with Phase 1 results"""
    
    def __init__(self):
        self.connectivity_analyzer = WallConnectivityAnalyzer()
        self.room_detector = RoomDetector()
    
    def process_semantic_segments(self, segments: List[SegmentedRegion]) -> Tuple[List[Room], Dict[str, Any]]:
        """Process semantic segments from Phase 1 to detect rooms"""
        
        logger.info("=" * 60)
        logger.info("🏠 PHASE 2: ROOM SEPARATION AND SPACE ANALYSIS")
        logger.info("=" * 60)
        
        # Step 1: Separate segments by type
        walls = [s for s in segments if s.region_type == 'wall']
        furniture = [s for s in segments if 'furniture' in s.region_type]
        # Better floor detection - separate floors from ceilings more intelligently
        floors = []
        for s in segments:
            if s.region_type == 'floor_ceiling':
                avg_height = s.centroid[2]
                if avg_height < 0.5:  # Definitely floor (below 50cm)
                    floors.append(s)
                elif avg_height < 1.0 and len(s.points) > 50000:  # Large, low segments are likely floors
                    floors.append(s)   
            ceilings = [s for s in segments if s.region_type in ['floor_ceiling', 'ceiling'] and s.centroid[2] > 1.5]
        
        logger.info(f"Separated segments: {len(walls)} walls, {len(furniture)} furniture, {len(floors)} floors, {len(ceilings)} ceilings")
        
        # Step 2: Analyze wall connectivity
        logger.info("Step 2.1: Wall Connectivity Analysis")
        wall_connectivity = self.connectivity_analyzer.analyze_wall_connectivity(walls)
        
        # Step 3: Detect rooms
        logger.info("Step 2.2: Room Detection")
        rooms = self.room_detector.detect_rooms(walls, furniture, floors, ceilings, wall_connectivity)
        
        # Step 4: Create analysis report
        analysis_report = self._create_room_analysis_report(rooms, wall_connectivity)
        
        logger.info("=" * 60)
        logger.info("🎉 ROOM SEPARATION COMPLETE")
        logger.info("=" * 60)
        
        return rooms, analysis_report
    
    def _create_room_analysis_report(self, rooms: List[Room], 
                                   wall_connectivity: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed analysis report"""
        
        # Count room types
        room_type_counts = {}
        total_area = 0
        total_furniture = 0
        
        for room in rooms:
            room_type = room.room_type
            room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
            total_area += room.area
            total_furniture += len(room.furniture)
        
        return {
            'total_rooms': len(rooms),
            'room_types': room_type_counts,
            'total_area': total_area,
            'total_furniture_assigned': total_furniture,
            'wall_connections': len(wall_connectivity.get('connections', [])),
            'wall_groups': len(wall_connectivity.get('wall_groups', []))
        }

def create_room_analysis_report(rooms: List[Room], analysis_report: Dict[str, Any]):
    """Create detailed room analysis report"""
    
    print("\n" + "="*70)
    print("🏠 DETAILED ROOM SEPARATION ANALYSIS")
    print("="*70)
    
    # Overall statistics
    print(f"📊 BUILDING OVERVIEW:")
    print(f"   Total rooms detected: {analysis_report['total_rooms']}")
    print(f"   Total floor area: {analysis_report['total_area']:.1f} m²")
    print(f"   Furniture items assigned: {analysis_report['total_furniture_assigned']}")
    print(f"   Wall connections found: {analysis_report['wall_connections']}")
    print(f"   Wall groups formed: {analysis_report['wall_groups']}")
    
    # Room type breakdown
    print(f"\n🏢 ROOM TYPE ANALYSIS:")
    room_types = analysis_report['room_types']
    for room_type, count in room_types.items():
        print(f"   {room_type.upper().replace('_', ' ')}: {count} rooms")
    
    # Individual room details
    print(f"\n🔍 INDIVIDUAL ROOM DETAILS:")
    for i, room in enumerate(rooms):
        furniture_summary = {}
        for furniture in room.furniture:
            ftype = furniture.region_type.replace('_furniture', '')
            furniture_summary[ftype] = furniture_summary.get(ftype, 0) + 1
        
        furniture_str = ", ".join([f"{count} {ftype}" for ftype, count in furniture_summary.items()])
        if not furniture_str:
            furniture_str = "No furniture"
        
        print(f"   Room {i+1} ({room.room_type}):")
        print(f"      Area: {room.area:.1f} m²")
        print(f"      Center: ({room.center_point[0]:.1f}, {room.center_point[1]:.1f}, {room.center_point[2]:.1f})")
        print(f"      Walls: {len(room.boundary_walls)}")
        print(f"      Furniture: {furniture_str}")
        print()
    
    # Space efficiency analysis
    if analysis_report['total_rooms'] > 0:
        avg_room_size = analysis_report['total_area'] / analysis_report['total_rooms']
        avg_furniture_per_room = analysis_report['total_furniture_assigned'] / analysis_report['total_rooms']
        
        print(f"📈 SPACE EFFICIENCY:")
        print(f"   Average room size: {avg_room_size:.1f} m²")
        print(f"   Average furniture per room: {avg_furniture_per_room:.1f} items")
        
        # Room utilization
        utilized_rooms = sum(1 for room in rooms if len(room.furniture) > 0)
        utilization_rate = (utilized_rooms / len(rooms)) * 100 if rooms else 0
        print(f"   Room utilization rate: {utilization_rate:.1f}% ({utilized_rooms}/{len(rooms)} rooms with furniture)")

def main():
    """Test the room separation system"""
    
    print("🏠 TESTING ROOM SEPARATION SYSTEM")
    print("=" * 50)
    
    # Load point cloud and run Phase 1 (semantic classification)
    test_file = 'building_segmented_colored.ply'
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("Run your segmentation.py first to generate semantic results")
        return
    
    point_cloud = o3d.io.read_point_cloud(test_file)
    print(f"Loaded {len(point_cloud.points):,} points")
    
    # Run Phase 1: Semantic Classification
    print("\n🏷️ Running Phase 1: Semantic Classification...")
    segmentation_pipeline = CompleteBuildingSegmentation()
    segments, doors = segmentation_pipeline.process_point_cloud(point_cloud)
    
    print(f"Phase 1 complete: {len(segments)} semantic segments")
    
    # Run Phase 2: Room Separation
    print("\n🏠 Running Phase 2: Room Separation...")
    room_pipeline = RoomSeparationPipeline()
    rooms, analysis_report = room_pipeline.process_semantic_segments(segments)
    
    # Create detailed analysis
    create_room_analysis_report(rooms, analysis_report)
    
    print(f"\n🎉 ROOM SEPARATION TEST COMPLETE!")
    print(f"Detected {len(rooms)} rooms with detailed analysis")

if __name__ == "__main__":
    main()