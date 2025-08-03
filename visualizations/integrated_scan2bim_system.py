#!/usr/bin/env python3
"""
Complete Integrated Scan2BIM System
Combines your proven wall detection with Phase 3 door/window detection
"""

import numpy as np
import open3d as o3d
import sys
import os
import time
from typing import List, Tuple
from pathlib import Path

# Import your existing proven segmentation system
from segmentation import CompleteBuildingSegmentation, SegmentedRegion, DetectedDoor

# Import the new Phase 3 system
from phase3_door_window_detection import Phase3DoorWindowDetector, DetectedOpening, add_phase3_to_existing_system

class CompleteScan2BIMPipeline:
    """
    Complete Scan2BIM pipeline combining:
    - Your proven wall/floor detection 
    - Advanced Phase 3 door/window detection
    """
    
    def __init__(self):
        self.building_segmentation = CompleteBuildingSegmentation()
        self.phase3_detector = Phase3DoorWindowDetector()
        
        print("🏗️ Complete Scan2BIM Pipeline Initialized")
        print("✅ Proven wall detection system loaded")
        print("✅ Phase 3 door/window detection loaded")
    
    def process_building_complete(self, point_cloud_file: str) -> Tuple[List[SegmentedRegion], List[DetectedOpening]]:
        """
        Complete building processing pipeline
        
        Args:
            point_cloud_file: Path to point cloud file
            
        Returns:
            Tuple of (wall_floor_segments, door_window_openings)
        """
        
        print("=" * 80)
        print("COMPLETE SCAN2BIM PROCESSING PIPELINE")
        print("=" * 80)
        
        # Step 1: Load point cloud
        print(f"Step 1: Loading point cloud from {point_cloud_file}...")
        point_cloud = self._load_point_cloud(point_cloud_file)
        
        if point_cloud is None:
            return [], []
        
        # Step 2: Your proven wall/floor detection
        print("Step 2: Running proven wall and floor detection...")
        wall_floor_segments, legacy_doors = self.building_segmentation.process_point_cloud(point_cloud)
        
        # Step 3: Phase 3 advanced door/window detection
        print("Step 3: Running Phase 3 door and window detection...")
        wall_segments = [s for s in wall_floor_segments if s.region_type == 'wall']
        
        if len(wall_segments) == 0:
            print("⚠️  No walls detected - skipping Phase 3")
            return wall_floor_segments, []
        
        door_window_openings = self.phase3_detector.process_building(point_cloud, wall_segments)
        
        # Step 4: Save comprehensive results
        print("Step 4: Saving comprehensive results...")
        self._save_complete_results(wall_floor_segments, door_window_openings, point_cloud)
        
        return wall_floor_segments, door_window_openings
    
    def _load_point_cloud(self, filename: str) -> o3d.geometry.PointCloud:
        """Load and validate point cloud"""
        
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            return None
        
        try:
            point_cloud = o3d.io.read_point_cloud(filename)
            
            if len(point_cloud.points) == 0:
                print(f"❌ Point cloud is empty!")
                return None
            
            points = np.asarray(point_cloud.points)
            
            print(f"✅ Loaded {len(points):,} points")
            print(f"   Bounds: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
                  f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
                  f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
            
            # Light downsampling for very large point clouds
            if len(point_cloud.points) > 5000000:
                print(f"🔽 Light downsampling for very large point cloud...")
                point_cloud = point_cloud.voxel_down_sample(0.01)  # 1cm voxel
                print(f"   Downsampled to {len(point_cloud.points):,} points")
            
            return point_cloud
            
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            return None
    
    def _save_complete_results(self, segments: List[SegmentedRegion], 
                              openings: List[DetectedOpening],
                              point_cloud: o3d.geometry.PointCloud):
        """Save comprehensive results including visualizations"""
        
        print(f"\n💾 Saving comprehensive results...")
        
        # Create results directory
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Save wall and floor segments (your existing method)
        self._save_wall_floor_segments(segments, results_dir)
        
        # Save door and window detections
        self._save_door_window_results(openings, results_dir)
        
        # Create comprehensive 3D visualization
        self._create_comprehensive_visualization(segments, openings, results_dir)
        
        # Generate final report
        self._generate_final_report(segments, openings, results_dir)
        
        print(f"✅ All results saved to {results_dir}/")
    
    def _save_wall_floor_segments(self, segments: List[SegmentedRegion], results_dir: Path):
        """Save wall and floor segments as PLY files"""
        
        wall_count = 0
        floor_count = 0
        
        for segment in segments:
            # Create point cloud
            segment_cloud = o3d.geometry.PointCloud() 
            segment_cloud.points = o3d.utility.Vector3dVector(segment.points)
            
            # Color and save based on type
            if segment.region_type == 'wall':
                wall_count += 1
                segment_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for walls
                filename = results_dir / f'segment_{wall_count:02d}_wall.ply'
            else:  # floor_ceiling
                floor_count += 1
                segment_cloud.paint_uniform_color([0.6, 0.4, 0.2])  # Brown for floors
                filename = results_dir / f'segment_{floor_count:02d}_floor_ceiling.ply'
            
            o3d.io.write_point_cloud(str(filename), segment_cloud)
        
        print(f"   Saved {wall_count} walls, {floor_count} floors/ceilings")
    
    def _save_door_window_results(self, openings: List[DetectedOpening], results_dir: Path):
        """Save door and window detection results"""
        
        if len(openings) == 0:
            print("   No door/window openings to save")
            return
        
        # Create door/window markers as small spheres
        door_spheres = []
        window_spheres = []
        
        for opening in openings:
            # Create sphere at opening position
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.translate(opening.position_3d)
            
            if opening.opening_type == 'door':
                sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red for doors
                door_spheres.append(sphere)
            elif opening.opening_type == 'window':
                sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for windows
                window_spheres.append(sphere)
        
        # Save door markers
        if door_spheres:
            combined_doors = door_spheres[0]
            for sphere in door_spheres[1:]:
                combined_doors += sphere
            o3d.io.write_triangle_mesh(str(results_dir / 'detected_doors.ply'), combined_doors)
        
        # Save window markers
        if window_spheres:
            combined_windows = window_spheres[0]
            for sphere in window_spheres[1:]:
                combined_windows += sphere
            o3d.io.write_triangle_mesh(str(results_dir / 'detected_windows.ply'), combined_windows)
        
        doors = [o for o in openings if o.opening_type == 'door']
        windows = [o for o in openings if o.opening_type == 'window']
        print(f"   Saved {len(doors)} doors, {len(windows)} windows")
    
    def _create_comprehensive_visualization(self, segments: List[SegmentedRegion], 
                                          openings: List[DetectedOpening], results_dir: Path):
        """Create comprehensive 3D visualization with everything"""
        
        all_geometries = []
        
        # Add wall and floor segments
        for segment in segments:
            segment_cloud = o3d.geometry.PointCloud() 
            segment_cloud.points = o3d.utility.Vector3dVector(segment.points)
            
            if segment.region_type == 'wall':
                segment_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray for walls
            else:
                segment_cloud.paint_uniform_color([0.4, 0.2, 0.1])  # Dark brown for floors
            
            all_geometries.append(segment_cloud)
        
        # Add door and window markers
        for opening in openings:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            sphere.translate(opening.position_3d)
            
            if opening.opening_type == 'door':
                sphere.paint_uniform_color([1.0, 0.2, 0.2])  # Bright red for doors
            elif opening.opening_type == 'window':
                sphere.paint_uniform_color([0.2, 0.2, 1.0])  # Bright blue for windows
            else:
                sphere.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow for unknown
            
            all_geometries.append(sphere)
        
        # Save combined visualization
        if all_geometries:
            # Combine all point clouds
            point_clouds = [g for g in all_geometries if isinstance(g, o3d.geometry.PointCloud)]
            meshes = [g for g in all_geometries if isinstance(g, o3d.geometry.TriangleMesh)]
            
            if point_clouds:
                combined_points = []
                combined_colors = []
                
                for pc in point_clouds:
                    points = np.asarray(pc.points)
                    colors = np.asarray(pc.colors)
                    combined_points.extend(points)
                    combined_colors.extend(colors)
                
                final_cloud = o3d.geometry.PointCloud()
                final_cloud.points = o3d.utility.Vector3dVector(combined_points)
                final_cloud.colors = o3d.utility.Vector3dVector(combined_colors)
                
                o3d.io.write_point_cloud(str(results_dir / 'complete_building_with_openings.ply'), final_cloud)
        
        print("   Created comprehensive 3D visualization")
    
    def _generate_final_report(self, segments: List[SegmentedRegion], 
                              openings: List[DetectedOpening], results_dir: Path):
        """Generate comprehensive final report"""
        
        # Count segments
        walls = [s for s in segments if s.region_type == 'wall']
        floors = [s for s in segments if s.region_type == 'floor_ceiling']
        doors = [o for o in openings if o.opening_type == 'door']
        windows = [o for o in openings if o.opening_type == 'window']
        
        # Calculate total areas
        wall_area = sum(s.area for s in walls)
        floor_area = sum(s.area for s in floors)
        
        report_file = results_dir / 'COMPLETE_SCAN2BIM_REPORT.txt'
        
        with open(report_file, 'w') as f:
            f.write("COMPLETE SCAN2BIM PROCESSING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("STRUCTURAL ELEMENTS DETECTED:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Wall segments: {len(walls)}\n")
            f.write(f"Floor/ceiling segments: {len(floors)}\n")
            f.write(f"Total wall area: {wall_area:.2f} m²\n")
            f.write(f"Total floor area: {floor_area:.2f} m²\n\n")
            
            f.write("ARCHITECTURAL OPENINGS DETECTED:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Doors: {len(doors)}\n")
            f.write(f"Windows: {len(windows)}\n")
            f.write(f"Total openings: {len(openings)}\n\n")
            
            if doors:
                f.write("DOOR DETAILS:\n")
                for i, door in enumerate(doors, 1):
                    f.write(f"  Door {i}: {door.width:.2f}m × {door.height:.2f}m "
                           f"at ({door.position_3d[0]:.1f}, {door.position_3d[1]:.1f}, {door.position_3d[2]:.1f})\n")
                f.write("\n")
            
            if windows:
                f.write("WINDOW DETAILS:\n")
                for i, window in enumerate(windows, 1):
                    f.write(f"  Window {i}: {window.width:.2f}m × {window.height:.2f}m "
                           f"at ({window.position_3d[0]:.1f}, {window.position_3d[1]:.1f}, {window.position_3d[2]:.1f})\n")
                f.write("\n")
            
            f.write("OUTPUT FILES:\n")
            f.write("-" * 30 + "\n")
            f.write("• segment_XX_wall.ply - Individual wall segments\n")
            f.write("• segment_XX_floor_ceiling.ply - Floor/ceiling segments\n")
            f.write("• detected_doors.ply - Door position markers\n")
            f.write("• detected_windows.ply - Window position markers\n") 
            f.write("• complete_building_with_openings.ply - Complete visualization\n")
            f.write("• debug_phase3/ - Detailed door/window detection analysis\n\n")
            
            f.write("VIEWING INSTRUCTIONS:\n")
            f.write("-" * 30 + "\n")
            f.write("Best overview: complete_building_with_openings.ply\n")
            f.write("- Gray points: Walls\n")
            f.write("- Brown points: Floors/ceilings\n")
            f.write("- Red spheres: Detected doors\n")
            f.write("- Blue spheres: Detected windows\n\n")
            
            f.write("Quick view command:\n")
            f.write('python -c "import open3d as o3d; ')
            f.write("o3d.visualization.draw_geometries([o3d.io.read_point_cloud('results/complete_building_with_openings.ply')])\"\n")
        
        print(f"   Generated comprehensive report: {report_file}")

def main():
    """Main entry point for complete Scan2BIM processing"""
    
    if len(sys.argv) != 2:
        print("Usage: python integrated_scan2bim_system.py <point_cloud_file.ply>")
        print("\nExample:")
        print("  python integrated_scan2bim_system.py building_scan.ply")
        return
    
    point_cloud_file = sys.argv[1]
    
    # Initialize complete pipeline
    pipeline = CompleteScan2BIMPipeline()
    
    # Process building
    start_time = time.time()
    segments, openings = pipeline.process_building_complete(point_cloud_file)
    processing_time = time.time() - start_time
    
    # Final summary
    if segments:
        walls = [s for s in segments if s.region_type == 'wall']
        floors = [s for s in segments if s.region_type == 'floor_ceiling']
        doors = [o for o in openings if o.opening_type == 'door']
        windows = [o for o in openings if o.opening_type == 'window']
        
        print("\n" + "=" * 80)
        print("COMPLETE SCAN2BIM PROCESSING RESULTS")
        print("=" * 80)
        print(f"✅ Successfully processed {point_cloud_file}")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        print(f"📊 Results:")
        print(f"   • {len(walls)} walls detected")
        print(f"   • {len(floors)} floors/ceilings detected") 
        print(f"   • {len(doors)} doors detected")
        print(f"   • {len(windows)} windows detected")
        print(f"📁 All results saved to: results/")
        print("\n🎉 COMPLETE SCAN2BIM PIPELINE SUCCESS!")
        
        print(f"\n👀 VIEW RESULTS:")
        print(f"python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('results/complete_building_with_openings.ply')])\"")
        
    else:
        print("\n❌ Processing failed - no segments detected")

if __name__ == "__main__":
    main()