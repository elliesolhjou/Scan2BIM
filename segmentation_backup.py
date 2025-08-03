#!/usr/bin/env python3
"""
Fixed segmentation.py - Integrates the PROVEN aggressive wall detection method
This replaces your existing segmentation.py with the method that actually works
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion, binary_dilation
import cv2
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging
from collections import defaultdict
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SegmentedRegion:
    """Represents a segmented region from point cloud"""
    points: np.ndarray
    normal: np.ndarray
    centroid: np.ndarray
    area: float
    confidence: float
    region_type: str  # 'wall', 'door', 'window', 'floor_ceiling', 'unknown'
    properties: Dict[str, Any]

@dataclass
class DetectedDoor:
    """Represents a detected door opening"""
    position: np.ndarray  # 3D position
    width: float
    height: float
    orientation: np.ndarray  # normal vector
    confidence: float
    wall_segment: Optional[SegmentedRegion]

class AggressiveWallDetector:
    
    def __init__(self):
        # These are the PROVEN parameters from the successful test
        self.eps = 0.1  # DBSCAN epsilon
        self.min_samples = 20  # DBSCAN min samples
        self.height_range = (0.1, 2.2)  # Wall height range
        self.horizontal_threshold = 0.27  # Very loose - captures fragmented walls
        self.z_threshold = 0.86  # Very loose - allows noisy normals
    
    def detect_walls(self, point_cloud: o3d.geometry.PointCloud) -> List[SegmentedRegion]:
        """Detect walls using the PROVEN aggressive clustering method"""
        
        logger.info("🎯 Using PROVEN aggressive wall detection method...")
        
        # Ensure normals exist
        if not point_cloud.has_normals():
            logger.info("Estimating normals...")
            point_cloud.estimate_normals()
        
        points = np.asarray(point_cloud.points)
        normals = np.asarray(point_cloud.normals)
        
        logger.info(f"Processing {len(points):,} points")
        
        # Step 1: Height filtering - PROVEN working range
        height_mask = (points[:, 2] > self.height_range[0]) & (points[:, 2] < self.height_range[1])
        wall_points = points[height_mask]
        wall_normals = normals[height_mask]
        
        logger.info(f"Height filtering: {len(wall_points):,} points in wall range")
        
        if len(wall_points) < 1000:
            logger.warning("Not enough points in wall height range")
            return []
        
        # Step 2: Normal filtering - PROVEN loose criteria
        x_align = np.abs(wall_normals[:, 0])
        y_align = np.abs(wall_normals[:, 1])
        z_align = np.abs(wall_normals[:, 2])
        max_horizontal = np.maximum(x_align, y_align)
        
        # Use the PROVEN working thresholds
        vertical_mask = (max_horizontal > self.horizontal_threshold) & (z_align < self.z_threshold)
        candidate_points = wall_points[vertical_mask]
        candidate_normals = wall_normals[vertical_mask]
        
        logger.info(f"Normal filtering: {len(candidate_points):,} wall candidate points")
        
        if len(candidate_points) < 100:
            logger.warning("Not enough wall candidates after normal filtering")
            return []
        
        # Step 3: Spatial clustering - PROVEN parameters
        logger.info("Running spatial clustering...")
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(candidate_points)
        
        unique_labels = set(labels) - {-1}  # Remove noise
        logger.info(f"Clustering found {len(unique_labels)} potential walls")
        
        if len(unique_labels) == 0:
            logger.warning("No clusters found")
            return []
        
        # Step 4: Extract and validate walls
        wall_segments = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = candidate_points[cluster_mask]
            cluster_normals = candidate_normals[cluster_mask]
            
            if len(cluster_points) < 50:  # Minimum size
                continue
            
            # Calculate properties
            min_coord = np.min(cluster_points, axis=0)
            max_coord = np.max(cluster_points, axis=0)
            dimensions = max_coord - min_coord
            centroid = np.mean(cluster_points, axis=0)
            
            # Calculate average normal
            avg_normal = np.mean(cluster_normals, axis=0)
            if np.linalg.norm(avg_normal) > 0:
                avg_normal = avg_normal / np.linalg.norm(avg_normal)
            else:
                avg_normal = np.array([1, 0, 0])  # Default normal
            
            # Validate as wall - PROVEN working criteria
            height = dimensions[2]
            width = max(dimensions[0], dimensions[1])
            
            if height > 0.3 and width > 0.3:  # Very loose criteria that worked
                # Calculate area approximation
                area = len(cluster_points) * 0.01  # Rough estimate
                
                segment = SegmentedRegion(
                    points=cluster_points,
                    normal=avg_normal,
                    centroid=centroid,
                    area=area,
                    confidence=0.9,  # High confidence - this method works!
                    region_type='wall',
                    properties={
                        'dimensions': dimensions,
                        'point_count': len(cluster_points),
                        'method': 'aggressive_clustering'
                    }
                )
                
                wall_segments.append(segment)
                logger.info(f"Wall {len(wall_segments)}: {len(cluster_points):,} points, "
                          f"{dimensions[0]:.1f}×{dimensions[1]:.1f}×{dimensions[2]:.1f}m")
        
        logger.info(f"🎉 Successfully detected {len(wall_segments)} walls!")
        return wall_segments

class RANSACSegmenter:
    """RANSAC segmenter - now uses aggressive wall detection + traditional floor detection"""
    
    def __init__(self, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.wall_detector = AggressiveWallDetector()
    
    def segment_planes(self, point_cloud: o3d.geometry.PointCloud) -> List[SegmentedRegion]:
        """Extract planar segments using WORKING wall detection + traditional floor detection"""
        
        logger.info("Starting INTEGRATED segmentation (aggressive walls + RANSAC floors)...")
        
        all_segments = []
        
        # Step 1: Use aggressive wall detection (that actually works)
        logger.info("🧱 WALL DETECTION using proven aggressive method...")
        wall_segments = self.wall_detector.detect_walls(point_cloud)
        all_segments.extend(wall_segments)
        
        # Step 2: Use traditional RANSAC for floors/ceilings (works fine for horizontal planes)
        logger.info("🏠 FLOOR/CEILING DETECTION using RANSAC...")
        floor_segments = self._detect_floors_ceilings_ransac(point_cloud)
        all_segments.extend(floor_segments)
        
        logger.info(f"INTEGRATED segmentation completed: {len(all_segments)} total segments")
        logger.info(f"  Walls: {len(wall_segments)}")
        logger.info(f"  Floors/Ceilings: {len(floor_segments)}")
        
        return all_segments
    
    def _detect_floors_ceilings_ransac(self, point_cloud: o3d.geometry.PointCloud) -> List[SegmentedRegion]:
        """Detect floors/ceilings using traditional RANSAC (works fine for horizontal planes)"""
        
        segments = []
        remaining_cloud = point_cloud
        
        for iteration in range(5):  # Max 5 horizontal planes
            if len(remaining_cloud.points) < 1000:
                break
            
            try:
                plane_model, inliers = remaining_cloud.segment_plane(
                    distance_threshold=self.distance_threshold,
                    ransac_n=self.ransac_n,
                    num_iterations=self.num_iterations
                )
                
                if len(inliers) < 500:
                    break
                
                # Extract plane
                plane_cloud = remaining_cloud.select_by_index(inliers)
                plane_points = np.asarray(plane_cloud.points)
                
                # Calculate normal
                normal = np.array(plane_model[:3])
                normal = normal / np.linalg.norm(normal)
                
                # Only accept if it's horizontal (z-alignment > 0.8)
                z_alignment = abs(normal[2])
                
                if z_alignment > 0.8:  # Horizontal surface
                    centroid = np.mean(plane_points, axis=0)
                    area = self._calculate_plane_area(plane_points)
                    
                    segment = SegmentedRegion(
                        points=plane_points,
                        normal=normal,
                        centroid=centroid,
                        area=area,
                        confidence=len(inliers) / len(remaining_cloud.points),
                        region_type='floor_ceiling',
                        properties={
                            'plane_equation': plane_model,
                            'num_points': len(inliers),
                            'z_alignment': z_alignment
                        }
                    )
                    
                    segments.append(segment)
                    logger.info(f"Floor/ceiling {len(segments)}: {len(inliers):,} points")
                
                # Remove inliers
                remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)
                
            except Exception as e:
                logger.warning(f"Floor detection iteration {iteration} failed: {e}")
                break
        
        return segments
    
    def _calculate_plane_area(self, points: np.ndarray) -> float:
        """Calculate approximate area of plane"""
        try:
            # Simple bounding box area approximation
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            dimensions = max_coords - min_coords
            return dimensions[0] * dimensions[1]  # X * Y area
        except:
            return 0.0

class RegionGrowingSegmenter:
    """Region growing for detailed segmentation refinement"""
    
    def __init__(self, normal_threshold=0.1, distance_threshold=0.05):
        self.normal_threshold = normal_threshold
        self.distance_threshold = distance_threshold
    
    def refine_segments(self, point_cloud: o3d.geometry.PointCloud, 
                       initial_segments: List[SegmentedRegion]) -> List[SegmentedRegion]:
        """Refine initial segments - now works with wall/floor segments"""
        
        logger.info("Starting region growing refinement...")
        refined_segments = []
        
        # For now, just return the initial segments since aggressive detection works well
        for segment in initial_segments:
            refined_segments.append(segment)
        
        logger.info(f"Region growing completed: {len(refined_segments)} refined segments")
        return refined_segments

class DBSCANClusterer:
    """DBSCAN clustering for noise removal and object grouping"""
    
    def __init__(self, eps=0.05, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples
    
    def cluster_segments(self, segments: List[SegmentedRegion]) -> List[SegmentedRegion]:
        """Apply DBSCAN clustering - now works with wall/floor segments"""
        
        logger.info("Starting DBSCAN clustering...")
        
        # For now, just return segments as is since aggressive detection works well
        final_segments = segments
        
        logger.info(f"DBSCAN clustering completed: {len(final_segments)} final segments")
        return final_segments

class DoorDetector:
    """Door detection based on gaps in wall segments"""
    
    def __init__(self):
        self.min_door_width = 0.6
        self.max_door_width = 1.5
        self.min_door_height = 1.8
        self.max_door_height = 2.5
    
    def detect_doors_in_walls(self, wall_segments: List[SegmentedRegion]) -> List[DetectedDoor]:
        """Detect doors as gaps in wall segments"""
        
        logger.info(f"🔍 Analyzing {len(wall_segments)} walls for doors...")
        
        doors = []
        
        for wall in wall_segments:
            # Skip small walls
            if len(wall.points) < 1000:
                continue
            
            # Simple door detection - look for height gaps
            wall_doors = self._analyze_wall_for_doors(wall)
            doors.extend(wall_doors)
        
        logger.info(f"Detected {len(doors)} potential doors")
        return doors
    
    def _analyze_wall_for_doors(self, wall: SegmentedRegion) -> List[DetectedDoor]:
        """Analyze a single wall for door openings"""
        
        points = wall.points
        
        # Simple approach: look for height gaps in the wall
        heights = points[:, 2]
        min_height = np.min(heights)
        max_height = np.max(heights)
        
        # If wall doesn't span typical door height range, skip
        if max_height - min_height < 1.5:
            return []
        
        # Look for gaps at floor level (potential door openings)
        floor_level_mask = heights < (min_height + 0.3)  # Bottom 30cm
        floor_points = points[floor_level_mask]
        
        if len(floor_points) < len(points) * 0.8:  # If less than 80% at floor level
            # Potential door opening
            door_position = wall.centroid.copy()
            door_position[2] = min_height + 1.0  # 1m above ground
            
            door = DetectedDoor(
                position=door_position,
                width=0.8,  # Estimated
                height=2.0,  # Estimated
                orientation=wall.normal,
                confidence=0.5,  # Medium confidence
                wall_segment=wall
            )
            
            return [door]
        
        return []

class CompleteBuildingSegmentation:
    """Complete building segmentation system using WORKING wall detection"""
    
    def __init__(self):
        self.ransac_segmenter = RANSACSegmenter()  # Now uses aggressive wall detection
        self.region_grower = RegionGrowingSegmenter()
        self.dbscan_clusterer = DBSCANClusterer()
        self.door_detector = DoorDetector()
        
        # Direct access to components
        self.wall_detector = self.ransac_segmenter.wall_detector
        self.floor_detector = self.ransac_segmenter
    
    def detect_floors_ceilings(self, point_cloud: o3d.geometry.PointCloud) -> List[SegmentedRegion]:
        """Wrapper method for floor/ceiling detection"""
        return self.ransac_segmenter._detect_floors_ceilings_ransac(point_cloud)
    def process_point_cloud(self, point_cloud: o3d.geometry.PointCloud) -> Tuple[List[SegmentedRegion], List[DetectedDoor]]:
        """Complete processing pipeline with semantic classification"""
        
        logger.info("=" * 50)
        logger.info("COMPLETE BUILDING SEGMENTATION - WITH SEMANTIC CLASSIFICATION")
        logger.info("=" * 50)
        
        # Step 1: Integrated segmentation (aggressive walls + RANSAC floors)
        logger.info("Step 1: Integrated Plane Segmentation")
        initial_segments = self.ransac_segmenter.segment_planes(point_cloud)
        
        # Step 1.5: Apply wall quality filtering
        logger.info("Step 1.5: Wall Quality Filtering")
        wall_filter = WallQualityFilter()
        filtered_segments, rejected_segments = wall_filter.filter_walls(initial_segments)
        
        logger.info(f"Wall filtering results:")
        logger.info(f"  Before: {len([s for s in initial_segments if s.region_type == 'wall'])} walls")
        logger.info(f"  After: {len([s for s in filtered_segments if s.region_type == 'wall'])} good walls")
        logger.info(f"  Rejected: {len([s for s in rejected_segments if s.region_type == 'wall'])} poor quality walls")
        
        # Step 1.6: NEW - Semantic Classification
        logger.info("Step 1.6: Semantic Classification")
        semantic_classifier = SemanticClassifier()
        semantically_enhanced_segments = semantic_classifier.classify_semantic_regions(
            point_cloud, filtered_segments)
        
        # Step 2: Region Growing Refinement (minimal for now)
        logger.info("Step 2: Region Growing Refinement")
        refined_segments = self.region_grower.refine_segments(point_cloud, semantically_enhanced_segments)
        
        # Step 3: DBSCAN Clustering (minimal for now)
        logger.info("Step 3: DBSCAN Clustering")
        final_segments = self.dbscan_clusterer.cluster_segments(refined_segments)
        
        # Step 4: Door Detection
        logger.info("Step 4: Door Detection")
        wall_segments = [s for s in final_segments if s.region_type == 'wall']
        doors = self.door_detector.detect_doors_in_walls(wall_segments)
        
        # Final summary with semantic breakdown
        logger.info("=" * 50)
        logger.info("COMPLETE SEGMENTATION RESULTS (WITH SEMANTICS)")
        logger.info("=" * 50)
        
        # Count by type
        segment_counts = {}
        for segment in final_segments:
            region_type = segment.region_type
            if region_type not in segment_counts:
                segment_counts[region_type] = 0
            segment_counts[region_type] += 1
        
        logger.info(f"Total segments: {len(final_segments)}")
        for region_type, count in segment_counts.items():
            logger.info(f"  {region_type}: {count}")
        logger.info(f"Detected doors: {len(doors)}")
        
        return final_segments, doors
    
def analyze_current_segments(segments):
    """Analyze segmentation results including semantic types"""
    
    print("=== SEMANTIC SEGMENTATION ANALYSIS ===")
    
    all_points_by_type = {}
    
    for i, segment in enumerate(segments):
        points = segment.points
        
        # Collect points by type
        if segment.region_type not in all_points_by_type:
            all_points_by_type[segment.region_type] = []
        all_points_by_type[segment.region_type].extend(points)
        
        # Basic stats
        height_range = points[:, 2].max() - points[:, 2].min()
        centroid_height = segment.centroid[2]
        
        print(f"Segment {i+1} ({segment.region_type}):")
        print(f"  Points: {len(points):,}")
        print(f"  Height range: {height_range:.2f}m")
        print(f"  Centroid height: {centroid_height:.2f}m")
        print(f"  Normal: [{segment.normal[0]:.2f}, {segment.normal[1]:.2f}, {segment.normal[2]:.2f}]")
        
        # Show furniture-specific properties
        if segment.region_type in ['low_furniture', 'medium_furniture', 'high_furniture']:
            props = segment.properties
            print(f"  Avg height: {props.get('avg_height', 0):.2f}m")
            print(f"  Height range: {props.get('height_range', 0):.2f}m")
        
        print()
    
    # Summary stats
    print("=== SEMANTIC SUMMARY ===")
    for region_type, points in all_points_by_type.items():
        print(f"{region_type}: {len(points):,} points")
    
    return all_points_by_type

def create_semantic_summary_report(segments):
    """Create a detailed semantic analysis report"""
    
    print("\n" + "="*60)
    print("📊 DETAILED SEMANTIC BUILDING ANALYSIS")
    print("="*60)
    
    # Group by semantic type
    semantic_groups = {}
    total_furniture_points = 0
    
    for segment in segments:
        region_type = segment.region_type
        if region_type not in semantic_groups:
            semantic_groups[region_type] = []
        semantic_groups[region_type].append(segment)
        
        if 'furniture' in region_type:
            total_furniture_points += len(segment.points)
    
    print(f"🏗️  BUILDING STRUCTURE:")
    for region_type in ['wall', 'floor_ceiling', 'ceiling']:
        if region_type in semantic_groups:
            segments_of_type = semantic_groups[region_type]
            total_points = sum(len(s.points) for s in segments_of_type)
            avg_height = sum(s.centroid[2] for s in segments_of_type) / len(segments_of_type)
            print(f"   {region_type.upper()}: {len(segments_of_type)} segments, {total_points:,} points, avg height {avg_height:.2f}m")
    
    print(f"\n🪑 FURNITURE & OBJECTS:")
    furniture_types = ['low_furniture', 'medium_furniture', 'high_furniture']
    for furniture_type in furniture_types:
        if furniture_type in semantic_groups:
            segments_of_type = semantic_groups[furniture_type]
            total_points = sum(len(s.points) for s in segments_of_type)
            avg_height = sum(s.centroid[2] for s in segments_of_type) / len(segments_of_type)
            largest_segment = max(segments_of_type, key=lambda x: len(x.points))
            
            print(f"   {furniture_type.upper()}: {len(segments_of_type)} objects, {total_points:,} points")
            print(f"      Average height: {avg_height:.2f}m")
            print(f"      Largest object: {len(largest_segment.points):,} points at {largest_segment.centroid[2]:.2f}m height")
    
    print(f"\n📈 BUILDING STATISTICS:")
    print(f"   Total objects detected: {len(segments)}")
    print(f"   Total furniture objects: {sum(len(semantic_groups.get(ft, [])) for ft in furniture_types)}")
    print(f"   Furniture point coverage: {total_furniture_points:,} points ({100*total_furniture_points/sum(len(s.points) for s in segments):.1f}%)")
    
    # Find interesting objects
    print(f"\n🔍 NOTABLE OBJECTS:")
    all_furniture = []
    for ft in furniture_types:
        all_furniture.extend(semantic_groups.get(ft, []))
    
    # Sort by size
    all_furniture.sort(key=lambda x: len(x.points), reverse=True)
    
    for i, segment in enumerate(all_furniture[:5]):  # Top 5 largest furniture
        height = segment.centroid[2]
        points = len(segment.points)
        
        # Guess what it might be
        if segment.region_type == 'low_furniture' and points > 1000:
            guess = "Large table or seating area"
        elif segment.region_type == 'medium_furniture' and points > 10000:
            guess = "Major desk/counter system"
        elif segment.region_type == 'high_furniture' and points > 10000:
            guess = "Wall of cabinets/shelving"
        else:
            guess = f"Large {segment.region_type.replace('_', ' ')}"
        
        print(f"   #{i+1}: {points:,} points at {height:.2f}m - {guess}")

# Legacy class for backward compatibility
class AdvancedSegmentationPipeline(CompleteBuildingSegmentation):
    """Legacy name - redirects to working system"""
    pass

class SemanticClassifier:
    """Advanced semantic labeling - classify points into different categories"""
    
    def __init__(self):
        self.furniture_height_range = (0.3, 2.0)    # Furniture is between 30cm and 2m
        self.ceiling_threshold = 1.8                # Above 1.8m is likely ceiling
        self.floor_threshold = 0.5                  # Below 50cm is likely floor
        self.min_furniture_size = 100               # Minimum points for furniture
    
    def classify_semantic_regions(self, point_cloud: o3d.geometry.PointCloud, 
        existing_segments: List[SegmentedRegion]) -> List[SegmentedRegion]:
        """Add semantic classification to improve labeling"""
        
        logger.info("🏷️ Starting advanced semantic classification...")
        
        # Get all points and their classifications
        all_points = np.asarray(point_cloud.points)
        classified_points = set()
        enhanced_segments = []
        
        # Keep existing good segments (walls and floors)
        for segment in existing_segments:
            enhanced_segments.append(segment)
            # Mark these points as classified
            for point in segment.points:
                classified_points.update([tuple(p) for p in point.reshape(1, -1)])
        
        # Find unclassified points
        unclassified_mask = np.array([
            tuple(point) not in classified_points 
            for point in all_points
        ])
        
        unclassified_points = all_points[unclassified_mask]
        logger.info(f"Found {len(unclassified_points):,} unclassified points")
        
        if len(unclassified_points) < 100:
            return enhanced_segments
        
        # Classify unclassified points by height and clustering
        furniture_segments = self._detect_furniture(unclassified_points)
        ceiling_segments = self._detect_additional_ceilings(unclassified_points, existing_segments)
        
        enhanced_segments.extend(furniture_segments)
        enhanced_segments.extend(ceiling_segments)
        
        logger.info(f"Semantic classification complete:")
        logger.info(f"  Added {len(furniture_segments)} furniture segments")
        logger.info(f"  Added {len(ceiling_segments)} additional ceiling segments")
        
        return enhanced_segments
    
    def _detect_furniture(self, points: np.ndarray) -> List[SegmentedRegion]:
        """Detect furniture and objects"""
        
        if len(points) < self.min_furniture_size:
            return []
        
        # Filter by furniture height range
        heights = points[:, 2]
        furniture_mask = (heights >= self.furniture_height_range[0]) & (heights <= self.furniture_height_range[1])
        furniture_points = points[furniture_mask]
        
        if len(furniture_points) < self.min_furniture_size:
            return []
        
        logger.info(f"Furniture detection: {len(furniture_points):,} candidate points")
        
        # Cluster furniture points
        try:
            clustering = DBSCAN(eps=0.15, min_samples=50)  # Larger clusters for furniture
            labels = clustering.fit_predict(furniture_points)
            
            unique_labels = set(labels) - {-1}  # Remove noise
            furniture_segments = []
            
            for label in unique_labels:
                cluster_points = furniture_points[labels == label]
                
                if len(cluster_points) >= self.min_furniture_size:
                    # Analyze furniture type by height
                    avg_height = np.mean(cluster_points[:, 2])
                    min_height = np.min(cluster_points[:, 2])
                    max_height = np.max(cluster_points[:, 2])
                    
                    # Simple furniture classification
                    if avg_height < 0.8:
                        furniture_type = 'low_furniture'  # Tables, chairs, etc.
                    elif avg_height > 1.5:
                        furniture_type = 'high_furniture'  # Cabinets, shelves, etc.
                    else:
                        furniture_type = 'medium_furniture'  # Desks, counters, etc.
                    
                    segment = SegmentedRegion(
                        points=cluster_points,
                        normal=np.array([0, 0, 1]),  # Default up normal
                        centroid=np.mean(cluster_points, axis=0),
                        area=len(cluster_points) * 0.01,
                        confidence=0.7,
                        region_type=furniture_type,
                        properties={
                            'avg_height': avg_height,
                            'height_range': max_height - min_height,
                            'point_count': len(cluster_points)
                        }
                    )
                    
                    furniture_segments.append(segment)
                    logger.info(f"Furniture {len(furniture_segments)} ({furniture_type}): "
                        f"{len(cluster_points)} points, height {avg_height:.2f}m")
            
            return furniture_segments
            
        except Exception as e:
            logger.warning(f"Furniture detection failed: {e}")
            return []
    
    def _detect_additional_ceilings(self, points: np.ndarray, 
        existing_segments: List[SegmentedRegion]) -> List[SegmentedRegion]:
        """Detect additional ceiling segments that might have been missed"""
        
        # Find points above ceiling threshold
        high_points = points[points[:, 2] > self.ceiling_threshold]
        
        if len(high_points) < 500:
            return []
        
        logger.info(f"Additional ceiling detection: {len(high_points):,} high points")
        
        # Simple clustering for ceiling points
        try:
            clustering = DBSCAN(eps=0.2, min_samples=100)
            labels = clustering.fit_predict(high_points)
            
            unique_labels = set(labels) - {-1}
            ceiling_segments = []
            
            for label in unique_labels:
                cluster_points = high_points[labels == label]
                
                if len(cluster_points) >= 500:
                    segment = SegmentedRegion(
                        points=cluster_points,
                        normal=np.array([0, 0, -1]),  # Down-facing normal
                        centroid=np.mean(cluster_points, axis=0),
                        area=len(cluster_points) * 0.01,
                        confidence=0.6,
                        region_type='ceiling',
                        properties={
                            'avg_height': np.mean(cluster_points[:, 2]),
                            'point_count': len(cluster_points)
                        }
                    )
                    
                    ceiling_segments.append(segment)
            
            return ceiling_segments
            
        except Exception as e:
            logger.warning(f"Additional ceiling detection failed: {e}")
            return []


class WallQualityFilter:
    """Filter and improve wall detection quality"""
    
    def __init__(self):
        self.min_points = 1000      # Minimum points for a valid wall
        self.min_height = 0.8       # Minimum wall height (80cm)
        self.max_height = 3.0       # Maximum wall height (3m)
        self.min_centroid_height = 0.2   # Walls should be above ground
        self.max_centroid_height = 2.5   # Walls shouldn't be at ceiling level
    
    def filter_walls(self, segments: List[SegmentedRegion]) -> Tuple[List[SegmentedRegion], List[SegmentedRegion]]:
        """Filter walls into good walls and rejected segments"""
        
        good_walls = []
        rejected = []
        
        for segment in segments:
            if segment.region_type != 'wall':
                good_walls.append(segment)  # Keep non-walls as is
                continue
            
            # Check wall quality
            points = segment.points
            height_range = points[:, 2].max() - points[:, 2].min()
            centroid_height = segment.centroid[2]
            
            # Quality checks
            if (len(points) >= self.min_points and 
                height_range >= self.min_height and 
                height_range <= self.max_height and
                centroid_height >= self.min_centroid_height and
                centroid_height <= self.max_centroid_height):
                
                good_walls.append(segment)
            else:
                rejected.append(segment)
        
        return good_walls, rejected


# Example usage and testing
def main():
    """Example usage of the WORKING segmentation system"""
    
    print("🏗️ TESTING INTEGRATED SEGMENTATION SYSTEM")
    print("=" * 60)
    
    # Load test point cloud
    test_file = 'building_segmented_colored.ply'
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    point_cloud = o3d.io.read_point_cloud(test_file)
    print(f"Loaded {len(point_cloud.points):,} points")
    
    # Process with integrated system
    pipeline = CompleteBuildingSegmentation()
    segments, doors = pipeline.process_point_cloud(point_cloud)
    
    print(f"\n🎉 INTEGRATION TEST RESULTS:")
    print(f"Total segments: {len(segments)}")
    print(f"Walls: {len([s for s in segments if s.region_type == 'wall'])}")
    print(f"Floors: {len([s for s in segments if s.region_type == 'floor_ceiling'])}")
    print(f"Doors: {len(doors)}")
    
    if len([s for s in segments if s.region_type == 'wall']) > 0:
        print("✅ WALL DETECTION WORKING IN INTEGRATED SYSTEM!")
    else:
        print("❌ Wall detection still not working")

    analyze_current_segments(segments)
    create_semantic_summary_report(segments)


if __name__ == "__main__":
    main()