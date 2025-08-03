#!/usr/bin/env python3
"""
Phase 3: Advanced Door & Window Detection System
Integrates with your proven wall detection to add comprehensive door/window detection
"""

import numpy as np
import open3d as o3d
import cv2
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectedOpening:
    """Represents a detected door or window opening"""
    opening_type: str  # 'door' or 'window'
    position_3d: np.ndarray  # 3D center position
    width: float
    height: float
    bottom_height: float  # Height from floor
    top_height: float     # Height to top
    wall_normal: np.ndarray
    confidence: float
    bounding_box_2d: Tuple[int, int, int, int]  # x1, y1, x2, y2 in 2D image
    wall_segment_id: int
    properties: Dict[str, Any]

@dataclass
class Wall2DProjection:
    """Represents a wall projected to 2D image"""
    wall_id: int
    wall_points_3d: np.ndarray
    wall_normal: np.ndarray
    wall_centroid: np.ndarray
    projection_plane: str  # 'XZ' or 'YZ'
    image_2d: np.ndarray  # 2D occupancy image
    grid_size: float
    image_bounds: Tuple[float, float, float, float]  # min_x, max_x, min_z, max_z
    point_to_pixel_mapping: np.ndarray  # Maps 3D points to 2D pixel coordinates

class Phase3DoorWindowDetector:
    """Complete Phase 3 implementation for door and window detection"""
    
    def __init__(self, grid_size=0.05, wall_buffer_distance=1.0):
        """
        Initialize Phase 3 detector
        
        Args:
            grid_size: Grid sampling size for 2D image creation (5cm default)
            wall_buffer_distance: Extract points within this distance of walls (1m default)
        """
        self.grid_size = grid_size
        self.wall_buffer_distance = wall_buffer_distance
        
        # Detection parameters
        self.min_door_width = 0.6
        self.max_door_width = 2.0
        self.min_door_height = 1.8
        self.max_door_height = 2.5
        
        self.min_window_width = 0.4
        self.max_window_width = 3.0
        self.min_window_height = 0.5
        self.max_window_height = 2.0
        self.min_window_bottom_height = 0.5  # Windows start above ground
        
        # Image processing parameters
        self.erosion_kernel_size = 3
        self.dilation_kernel_size = 5
        
        logger.info("Phase 3 Door & Window Detection System initialized")
        logger.info(f"Grid size: {grid_size}m, Wall buffer: {wall_buffer_distance}m")
    
    def process_building(self, point_cloud: o3d.geometry.PointCloud, 
                        wall_segments: List) -> List[DetectedOpening]:
        """
        Complete Phase 3 processing pipeline
        
        Args:
            point_cloud: Original point cloud
            wall_segments: List of detected wall segments from your existing system
            
        Returns:
            List of detected doors and windows
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: ADVANCED DOOR & WINDOW DETECTION")
        logger.info("=" * 60)
        
        # Step 3.1: 3D to 2D Conversion
        logger.info("Step 3.1: Converting walls to 2D projections...")
        wall_projections = self._convert_walls_to_2d(point_cloud, wall_segments)
        
        # Step 3.2: Gap-based Detection (replaces AI until YOLOv8 is trained)
        logger.info("Step 3.2: Detecting openings using advanced gap analysis...")
        detected_openings = []
        
        for projection in wall_projections:
            openings = self._detect_openings_in_2d_projection(projection)
            detected_openings.extend(openings)
        
        # Step 3.3: 3D Integration and Classification
        logger.info("Step 3.3: Converting 2D detections back to 3D and classifying...")
        final_openings = self._integrate_2d_to_3d(detected_openings, wall_projections)
        
        # Results summary
        doors = [o for o in final_openings if o.opening_type == 'door']
        windows = [o for o in final_openings if o.opening_type == 'window']
        
        logger.info("=" * 60)
        logger.info("PHASE 3 DETECTION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total openings detected: {len(final_openings)}")
        logger.info(f"Doors: {len(doors)}")
        logger.info(f"Windows: {len(windows)}")
        
        # Save debug visualizations
        self._save_debug_visualizations(wall_projections, final_openings)
        
        return final_openings
    
    def _convert_walls_to_2d(self, point_cloud: o3d.geometry.PointCloud, 
                            wall_segments: List) -> List[Wall2DProjection]:
        """
        Step 3.1: Convert 3D wall segments to 2D image projections
        """
        logger.info(f"Converting {len(wall_segments)} walls to 2D projections...")
        
        points_3d = np.asarray(point_cloud.points)
        projections = []
        
        for wall_id, wall_segment in enumerate(wall_segments):
            if wall_segment.region_type != 'wall':
                continue
                
            # Extract wall properties
            wall_points = wall_segment.points
            wall_normal = wall_segment.normal
            wall_centroid = wall_segment.centroid
            
            # Find points within buffer distance of this wall
            wall_nearby_points = self._extract_points_near_wall(
                points_3d, wall_points, wall_centroid, wall_normal
            )
            
            if len(wall_nearby_points) < 100:
                logger.warning(f"Wall {wall_id}: Not enough nearby points ({len(wall_nearby_points)})")
                continue
            
            # Determine projection plane based on wall normal
            projection_plane = self._determine_projection_plane(wall_normal)
            
            # Create 2D projection
            projection = self._create_2d_projection(
                wall_id, wall_nearby_points, wall_normal, wall_centroid, projection_plane
            )
            
            if projection is not None:
                projections.append(projection)
                logger.info(f"Wall {wall_id}: Created {projection_plane} projection "
                          f"({projection.image_2d.shape[1]}x{projection.image_2d.shape[0]} pixels)")
        
        logger.info(f"Successfully created {len(projections)} wall projections")
        return projections
    
    def _extract_points_near_wall(self, all_points: np.ndarray, wall_points: np.ndarray,
                                 wall_centroid: np.ndarray, wall_normal: np.ndarray) -> np.ndarray:
        """Extract points within buffer distance of wall"""
        
        # Calculate distances from all points to wall plane
        wall_normal_normalized = wall_normal / np.linalg.norm(wall_normal)
        
        # Distance from point to plane = |n·(p - p0)| where n is normal, p0 is point on plane
        distances = np.abs(np.dot(all_points - wall_centroid, wall_normal_normalized))
        
        # Select points within buffer distance
        nearby_mask = distances <= self.wall_buffer_distance
        nearby_points = all_points[nearby_mask]
        
        return nearby_points
    
    def _determine_projection_plane(self, wall_normal: np.ndarray) -> str:
        """Determine best projection plane (XZ or YZ) based on wall normal"""
        
        abs_normal = np.abs(wall_normal)
        
        # If wall normal is more aligned with X-axis, project to YZ plane
        if abs_normal[0] > abs_normal[1]:
            return 'YZ'
        else:
            return 'XZ'
    
    def _create_2d_projection(self, wall_id: int, points_3d: np.ndarray, 
                             wall_normal: np.ndarray, wall_centroid: np.ndarray,
                             projection_plane: str) -> Optional[Wall2DProjection]:
        """Create 2D occupancy image from 3D points"""
        
        # Select coordinates based on projection plane
        if projection_plane == 'XZ':
            coords_2d = points_3d[:, [0, 2]]  # X and Z coordinates
            plane_indices = [0, 2]
        else:  # YZ
            coords_2d = points_3d[:, [1, 2]]  # Y and Z coordinates  
            plane_indices = [1, 2]
        
        # Calculate bounds
        min_coord = np.min(coords_2d, axis=0)
        max_coord = np.max(coords_2d, axis=0)
        
        # Add padding
        padding = 0.5  # 50cm padding
        min_coord -= padding
        max_coord += padding
        
        # Calculate image dimensions
        image_size = max_coord - min_coord
        image_width = int(np.ceil(image_size[0] / self.grid_size))
        image_height = int(np.ceil(image_size[1] / self.grid_size))
        
        if image_width < 10 or image_height < 10:
            logger.warning(f"Wall {wall_id}: Image too small ({image_width}x{image_height})")
            return None
        
        # Limit maximum image size
        max_size = 2000
        if image_width > max_size or image_height > max_size:
            logger.warning(f"Wall {wall_id}: Image too large, skipping")
            return None
        
        # Create occupancy grid
        occupancy_grid = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # Convert 3D points to 2D pixel coordinates
        pixel_coords = ((coords_2d - min_coord) / self.grid_size).astype(int)
        
        # Clip to image bounds
        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, image_width - 1)
        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, image_height - 1)
        
        # Fill occupancy grid (Y coordinate is flipped for image coordinates)
        occupancy_grid[image_height - 1 - pixel_coords[:, 1], pixel_coords[:, 0]] = 255
        
        # Create projection object
        projection = Wall2DProjection(
            wall_id=wall_id,
            wall_points_3d=points_3d,
            wall_normal=wall_normal,
            wall_centroid=wall_centroid,
            projection_plane=projection_plane,
            image_2d=occupancy_grid,
            grid_size=self.grid_size,
            image_bounds=(min_coord[0], max_coord[0], min_coord[1], max_coord[1]),
            point_to_pixel_mapping=pixel_coords
        )
        
        return projection
    
    def _detect_openings_in_2d_projection(self, projection: Wall2DProjection) -> List[DetectedOpening]:
        """
        Step 3.2: Detect openings in 2D projection using advanced gap analysis
        """
        
        # Apply morphological operations to clean up the image
        kernel_erosion = np.ones((self.erosion_kernel_size, self.erosion_kernel_size), np.uint8)
        kernel_dilation = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
        
        # Erode to remove noise, then dilate to fill small gaps
        processed_image = cv2.erode(projection.image_2d, kernel_erosion, iterations=1)
        processed_image = cv2.dilate(processed_image, kernel_dilation, iterations=1)
        
        # Invert image to find gaps (gaps become white)
        inverted_image = 255 - processed_image
        
        # Find connected components (potential openings)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_image, connectivity=8)
        
        openings = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            # Get component properties
            x, y, width, height, area = stats[i]
            centroid_x, centroid_y = centroids[i]
            
            # Convert pixel dimensions to real-world dimensions
            width_real = width * projection.grid_size
            height_real = height * projection.grid_size
            
            # Basic size filtering
            if width_real < 0.3 or height_real < 0.3:  # Too small
                continue
            if width_real > 5.0 or height_real > 4.0:  # Too large
                continue
            if area < 50:  # Too small area in pixels
                continue
            
            # Convert centroid from image coordinates to real-world coordinates
            real_x = projection.image_bounds[0] + centroid_x * projection.grid_size
            real_z = projection.image_bounds[3] - centroid_y * projection.grid_size  # Flip Y
            
            # Convert back to 3D coordinates
            if projection.projection_plane == 'XZ':
                position_3d = np.array([real_x, projection.wall_centroid[1], real_z])
            else:  # YZ
                position_3d = np.array([projection.wall_centroid[0], real_x, real_z])
            
            # Calculate confidence based on shape regularity
            aspect_ratio = max(width_real, height_real) / min(width_real, height_real)
            regularity_score = min(1.0, 2.0 / aspect_ratio)  # More square = higher score
            size_score = min(1.0, area / 1000.0)  # Larger = higher score (up to limit)
            confidence = (regularity_score + size_score) / 2.0
            
            # Create detected opening
            opening = DetectedOpening(
                opening_type='unknown',  # Will be classified in next step
                position_3d=position_3d,
                width=width_real,
                height=height_real,
                bottom_height=real_z - height_real/2,
                top_height=real_z + height_real/2,
                wall_normal=projection.wall_normal,
                confidence=confidence,
                bounding_box_2d=(x, y, x + width, y + height),
                wall_segment_id=projection.wall_id,
                properties={
                    'area_pixels': int(area),
                    'aspect_ratio': aspect_ratio,
                    'projection_plane': projection.projection_plane
                }
            )
            
            openings.append(opening)
        
        logger.info(f"Wall {projection.wall_id}: Found {len(openings)} potential openings")
        return openings
    
    def _integrate_2d_to_3d(self, detected_openings: List[DetectedOpening],
                           projections: List[Wall2DProjection]) -> List[DetectedOpening]:
        """
        Step 3.3: Integrate 2D detections back to 3D and classify as doors/windows
        """
        
        classified_openings = []
        
        for opening in detected_openings:
            # Height-based classification
            bottom_height = opening.bottom_height
            height = opening.height
            width = opening.width
            
            # Door classification criteria
            is_door_height = (self.min_door_height <= height <= self.max_door_height)
            is_door_width = (self.min_door_width <= width <= self.max_door_width)
            is_at_floor_level = (bottom_height <= 0.3)  # Doors start near floor
            
            # Window classification criteria
            is_window_height = (self.min_window_height <= height <= self.max_window_height)
            is_window_width = (self.min_window_width <= width <= self.max_window_width)
            is_above_floor = (bottom_height >= self.min_window_bottom_height)
            
            # Classify
            if is_door_height and is_door_width and is_at_floor_level:
                opening.opening_type = 'door'
                opening.confidence *= 1.2  # Boost confidence for good door match
            elif is_window_height and is_window_width and is_above_floor:
                opening.opening_type = 'window'
                opening.confidence *= 1.1  # Boost confidence for good window match
            else:
                opening.opening_type = 'unknown'
                opening.confidence *= 0.8  # Reduce confidence for unclear classification
            
            # Cap confidence at 1.0
            opening.confidence = min(1.0, opening.confidence)
            
            # Add standard dimensions matching
            opening.properties['standard_match'] = self._match_standard_dimensions(opening)
            
            classified_openings.append(opening)
        
        # Filter by confidence threshold
        confident_openings = [o for o in classified_openings if o.confidence > 0.3]
        
        logger.info(f"Classification results: {len(confident_openings)}/{len(detected_openings)} above confidence threshold")
        
        return confident_openings
    
    def _match_standard_dimensions(self, opening: DetectedOpening) -> Dict[str, Any]:
        """Match opening dimensions to standard door/window sizes"""
        
        width = opening.width
        height = opening.height
        
        # Standard door sizes (width x height in meters)
        standard_doors = [
            (0.6, 2.0), (0.7, 2.0), (0.8, 2.0), (0.9, 2.0),
            (1.0, 2.0), (1.2, 2.0), (1.5, 2.0)
        ]
        
        # Standard window sizes
        standard_windows = [
            (0.6, 1.2), (0.8, 1.2), (1.0, 1.2), (1.2, 1.2),
            (1.5, 1.2), (1.8, 1.2), (2.0, 1.2), (1.0, 1.5)
        ]
        
        standards = standard_doors if opening.opening_type == 'door' else standard_windows
        
        # Find closest match
        min_distance = float('inf')
        best_match = None
        
        for std_width, std_height in standards:
            distance = ((width - std_width) ** 2 + (height - std_height) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                best_match = (std_width, std_height)
        
        return {
            'best_standard_size': best_match,
            'distance_to_standard': min_distance,
            'is_standard_size': min_distance < 0.2  # Within 20cm of standard
        }
    
    def _save_debug_visualizations(self, projections: List[Wall2DProjection], 
                                  openings: List[DetectedOpening]):
        """Save debug visualizations for analysis"""
        
        debug_dir = Path('debug_phase3')
        debug_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving debug visualizations to {debug_dir}/")
        
        for projection in projections:
            # Save original projection
            plt.figure(figsize=(12, 8))
            plt.imshow(projection.image_2d, cmap='gray', origin='lower')
            plt.title(f'Wall {projection.wall_id} - {projection.projection_plane} Projection')
            plt.xlabel('X (pixels)')
            plt.ylabel('Z (pixels)')
            
            # Overlay detected openings
            wall_openings = [o for o in openings if o.wall_segment_id == projection.wall_id]
            
            for opening in wall_openings:
                x1, y1, x2, y2 = opening.bounding_box_2d
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                
                # Add label
                color = 'red' if opening.opening_type == 'door' else 'blue'
                plt.text(x1, y1-5, f'{opening.opening_type}\n{opening.width:.1f}x{opening.height:.1f}m', 
                        color=color, fontsize=8, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(debug_dir / f'wall_{projection.wall_id}_{projection.projection_plane}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save summary report
        self._save_detection_report(openings, debug_dir)
    
    def _save_detection_report(self, openings: List[DetectedOpening], output_dir: Path):
        """Save detailed detection report"""
        
        report_data = {
            'total_openings': len(openings),
            'doors': len([o for o in openings if o.opening_type == 'door']),
            'windows': len([o for o in openings if o.opening_type == 'window']),
            'unknown': len([o for o in openings if o.opening_type == 'unknown']),
            'detections': []
        }
        
        for i, opening in enumerate(openings):
            # Convert properties to JSON-safe format
            json_safe_properties = {}
            for key, value in opening.properties.items():
                if isinstance(value, np.ndarray):
                    json_safe_properties[key] = value.tolist()
                elif isinstance(value, (np.bool_, bool)):
                    json_safe_properties[key] = bool(value)
                elif isinstance(value, (np.integer, int)):
                    json_safe_properties[key] = int(value)
                elif isinstance(value, (np.floating, float)):
                    json_safe_properties[key] = float(value)
                else:
                    json_safe_properties[key] = str(value)
            
            report_data['detections'].append({
                'id': i,
                'type': opening.opening_type,
                'position_3d': opening.position_3d.tolist(),
                'width': float(opening.width),
                'height': float(opening.height),
                'bottom_height': float(opening.bottom_height),
                'top_height': float(opening.top_height),
                'confidence': float(opening.confidence),
                'wall_id': int(opening.wall_segment_id),
                'properties': json_safe_properties
            })
        
        # Save JSON report
        with open(output_dir / 'detection_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Save human-readable report
        with open(output_dir / 'detection_summary.txt', 'w') as f:
            f.write("PHASE 3: DOOR & WINDOW DETECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Openings Detected: {report_data['total_openings']}\n")
            f.write(f"Doors: {report_data['doors']}\n")
            f.write(f"Windows: {report_data['windows']}\n") 
            f.write(f"Unknown: {report_data['unknown']}\n\n")
            
            f.write("DETAILED DETECTIONS:\n")
            f.write("-" * 30 + "\n")
            
            for detection in report_data['detections']:
                f.write(f"\n{detection['type'].upper()} #{detection['id']}\n")
                f.write(f"  Position: ({detection['position_3d'][0]:.2f}, "
                       f"{detection['position_3d'][1]:.2f}, {detection['position_3d'][2]:.2f})\n")
                f.write(f"  Dimensions: {detection['width']:.2f}m × {detection['height']:.2f}m\n")
                f.write(f"  Height: {detection['bottom_height']:.2f}m to {detection['top_height']:.2f}m\n")
                f.write(f"  Confidence: {detection['confidence']:.3f}\n")
                f.write(f"  Wall ID: {detection['wall_id']}\n")
        
        logger.info(f"Detection report saved to {output_dir}")

# Integration function for your existing system
def add_phase3_to_existing_system(point_cloud: o3d.geometry.PointCloud, 
                                 wall_segments: List) -> Tuple[List, List[DetectedOpening]]:
    """
    Integration function to add Phase 3 to your existing segmentation system
    
    Args:
        point_cloud: Your original point cloud
        wall_segments: Wall segments from your existing system
        
    Returns:
        Tuple of (original_segments, detected_openings)
    """
    
    # Initialize Phase 3 detector
    phase3_detector = Phase3DoorWindowDetector()
    
    # Run Phase 3 detection
    detected_openings = phase3_detector.process_building(point_cloud, wall_segments)
    
    return wall_segments, detected_openings

# Example usage with your existing system
def main():
    """Example of integrating Phase 3 with your existing system"""
    
    print("🚪 PHASE 3: DOOR & WINDOW DETECTION TEST")
    print("=" * 60)
    
    # This would be replaced with your actual segmentation system
    test_file = 'building_segmented_colored.ply'
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("Place your point cloud file and update the filename")
        return
    
    # Load point cloud
    point_cloud = o3d.io.read_point_cloud(test_file)
    print(f"Loaded {len(point_cloud.points):,} points")
    
    # TODO: Replace this with your actual wall detection system
    # For now, we'll create dummy wall segments for testing
    print("⚠️  Using dummy wall segments for testing")
    print("   Replace this with your actual CompleteBuildingSegmentation system")
    
    # Mock wall segments (replace with your actual system)
    mock_wall_segments = []
    
    if len(mock_wall_segments) == 0:
        print("❌ No wall segments provided")
        print("   Please integrate with your actual wall detection system")
        return
    
    # Run Phase 3 detection
    wall_segments, detected_openings = add_phase3_to_existing_system(
        point_cloud, mock_wall_segments
    )
    
    # Results
    doors = [o for o in detected_openings if o.opening_type == 'door']
    windows = [o for o in detected_openings if o.opening_type == 'window']
    
    print(f"\n🎉 PHASE 3 RESULTS:")
    print(f"Total openings: {len(detected_openings)}")
    print(f"Doors: {len(doors)}")
    print(f"Windows: {len(windows)}")
    
    if len(detected_openings) > 0:
        print("✅ PHASE 3 DOOR & WINDOW DETECTION WORKING!")
        print("📁 Check debug_phase3/ directory for visualizations")
    else:
        print("❌ No openings detected")

if __name__ == "__main__":
    main()