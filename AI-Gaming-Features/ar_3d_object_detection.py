"""
3D Object Detection System for Augmented Reality Gaming
Uses computer vision for real-time object detection and AR overlay.
"""

import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import List, Tuple, Dict, Optional
import math
import time


class ObjectDetector:
    """Detects objects in real-time using color-based detection and contour analysis"""
    
    def __init__(self):
        self.detected_objects = []
        
        # Color ranges for object detection (HSV)
        self.color_ranges = {
            "red": [(0, 100, 100), (10, 255, 255)],
            "red2": [(170, 100, 100), (180, 255, 255)],  # Red wraps around in HSV
            "green": [(40, 50, 50), (80, 255, 255)],
            "blue": [(100, 50, 50), (130, 255, 255)],
            "yellow": [(20, 100, 100), (30, 255, 255)],
        }
        
        # ArUco marker detector for pose estimation
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Camera calibration (approximate values, calibrate for better results)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1))
    
    def detect_colored_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect colored objects in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected = []
        
        for color_name, (lower, upper) in self.color_ranges.items():
            if "red2" in color_name:
                continue  # Handle red separately
            
            # Create mask
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            mask = cv2.inRange(hsv, lower_np, upper_np)
            
            # Handle red (special case)
            if color_name == "red":
                lower2 = np.array(self.color_ranges["red2"][0])
                upper2 = np.array(self.color_ranges["red2"][1])
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask, mask2)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate center
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Estimate 3D position (simplified)
                    # Distance estimation based on object size
                    estimated_distance = 1000 / math.sqrt(area)
                    
                    detected.append({
                        "color": color_name,
                        "bbox": (x, y, w, h),
                        "center": (center_x, center_y),
                        "area": area,
                        "distance": estimated_distance,
                        "position_3d": self.estimate_3d_position(center_x, center_y, estimated_distance)
                    })
        
        return detected
    
    def estimate_3d_position(self, x: int, y: int, distance: float) -> Tuple[float, float, float]:
        """Estimate 3D position from 2D coordinates and distance"""
        # Simple perspective projection inverse
        focal_length = self.camera_matrix[0, 0]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # Calculate 3D coordinates
        z = distance
        x_3d = (x - cx) * z / focal_length
        y_3d = (y - cy) * z / focal_length
        
        return (x_3d, y_3d, z)
    
    def detect_aruco_markers(self, frame: np.ndarray) -> List[Dict]:
        """Detect ArUco markers for precise pose estimation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        markers = []
        if ids is not None:
            # Estimate pose for each marker
            marker_size = 0.05  # 5cm markers
            
            for i, corner in enumerate(corners):
                # Estimate pose
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corner, marker_size, self.camera_matrix, self.dist_coeffs
                )
                
                markers.append({
                    "id": int(ids[i][0]),
                    "corners": corner[0],
                    "rvec": rvec[0],
                    "tvec": tvec[0],
                    "center": np.mean(corner[0], axis=0).astype(int)
                })
        
        return markers
    
    def draw_detected_objects(self, frame: np.ndarray, objects: List[Dict]):
        """Draw bounding boxes and labels for detected objects"""
        for obj in objects:
            x, y, w, h = obj["bbox"]
            color = self.get_color_bgr(obj["color"])
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{obj['color']}: {obj['distance']:.1f} units"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)
            
            # Draw center point
            cv2.circle(frame, obj["center"], 5, color, -1)
    
    def draw_aruco_markers(self, frame: np.ndarray, markers: List[Dict]):
        """Draw ArUco markers and their axes"""
        for marker in markers:
            corners = marker["corners"]
            
            # Draw marker border
            corners_int = corners.astype(int)
            cv2.polylines(frame, [corners_int], True, (0, 255, 0), 2)
            
            # Draw ID
            center = marker["center"]
            cv2.putText(frame, f"ID: {marker['id']}", tuple(center),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw 3D axis
            if "rvec" in marker and "tvec" in marker:
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                 marker["rvec"], marker["tvec"], 0.03)
    
    def get_color_bgr(self, color_name: str) -> Tuple[int, int, int]:
        """Get BGR color for drawing"""
        colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255)
        }
        return colors.get(color_name, (255, 255, 255))


class AR3DObject:
    """Represents a 3D object to be rendered in AR"""
    
    def __init__(self, position: Tuple[float, float, float], 
                 obj_type: str = "cube", color: Tuple[float, float, float] = (1, 0, 0)):
        self.position = position
        self.obj_type = obj_type
        self.color = color
        self.rotation = [0, 0, 0]
        self.scale = 1.0
        self.velocity = [0, 0, 0]
    
    def update(self, dt: float):
        """Update object physics"""
        # Apply velocity
        self.position = (
            self.position[0] + self.velocity[0] * dt,
            self.position[1] + self.velocity[1] * dt,
            self.position[2] + self.velocity[2] * dt
        )
        
        # Auto-rotate for effect
        self.rotation[1] += 50 * dt
    
    def render(self):
        """Render the 3D object"""
        glPushMatrix()
        
        # Apply transformations
        glTranslatef(*self.position)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(self.scale, self.scale, self.scale)
        
        # Set color
        glColor3f(*self.color)
        
        # Render based on type
        if self.obj_type == "cube":
            self.render_cube()
        elif self.obj_type == "sphere":
            self.render_sphere()
        elif self.obj_type == "pyramid":
            self.render_pyramid()
        
        glPopMatrix()
    
    def render_cube(self):
        """Render a cube"""
        glBegin(GL_QUADS)
        
        # Front face
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        
        # Back face
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        
        # Top face
        glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, 0.5, -0.5)
        
        # Bottom face
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        
        # Right face
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        
        # Left face
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(-0.5, 0.5, 0.5)
        glVertex3f(-0.5, 0.5, -0.5)
        
        glEnd()
    
    def render_pyramid(self):
        """Render a pyramid"""
        glBegin(GL_TRIANGLES)
        
        # Front face
        glVertex3f(0, 0.5, 0)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        
        # Right face
        glVertex3f(0, 0.5, 0)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, -0.5)
        
        # Back face
        glVertex3f(0, 0.5, 0)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, -0.5)
        
        # Left face
        glVertex3f(0, 0.5, 0)
        glVertex3f(-0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, 0.5)
        
        glEnd()
        
        # Base
        glBegin(GL_QUADS)
        glVertex3f(-0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, -0.5, -0.5)
        glVertex3f(-0.5, -0.5, -0.5)
        glEnd()
    
    def render_sphere(self):
        """Render a sphere"""
        quadric = gluNewQuadric()
        gluSphere(quadric, 0.5, 20, 20)
        gluDeleteQuadric(quadric)


class ARGamingApp:
    """AR Gaming Application with 3D object detection"""
    
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.width = 1280
        self.height = 720
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("AR 3D Object Detection Game")
        
        # Initialize OpenGL
        self.init_opengl()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize detector
        self.detector = ObjectDetector()
        
        # AR objects
        self.ar_objects = []
        self.spawn_timer = 0
        
        # Game state
        self.score = 0
        self.running = True
        self.clock = pygame.time.Clock()
        self.last_time = time.time()
        
        # Font for UI
        self.font = pygame.font.Font(None, 36)
    
    def init_opengl(self):
        """Initialize OpenGL settings"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 50.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
    
    def spawn_ar_object(self, position: Tuple[float, float, float]):
        """Spawn a new AR object"""
        obj_types = ["cube", "pyramid", "sphere"]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
        
        obj = AR3DObject(
            position=position,
            obj_type=np.random.choice(obj_types),
            color=colors[np.random.randint(0, len(colors))]
        )
        
        # Add some random velocity
        obj.velocity = [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.2, 0.2)
        ]
        
        self.ar_objects.append(obj)
    
    def update(self, dt: float):
        """Update game state"""
        # Process camera frame
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        
        # Detect objects
        detected_objects = self.detector.detect_colored_objects(frame)
        markers = self.detector.detect_aruco_markers(frame)
        
        # Draw detections
        self.detector.draw_detected_objects(frame, detected_objects)
        self.detector.draw_aruco_markers(frame, markers)
        
        # Spawn AR objects on detected objects
        self.spawn_timer += dt
        if self.spawn_timer > 2.0 and len(detected_objects) > 0:
            # Spawn on random detected object
            obj = np.random.choice(detected_objects)
            pos_3d = obj["position_3d"]
            # Scale position for better AR effect
            self.spawn_ar_object((pos_3d[0] / 100, pos_3d[1] / 100, -3))
            self.spawn_timer = 0
        
        # Update AR objects
        for ar_obj in self.ar_objects[:]:
            ar_obj.update(dt)
            
            # Remove objects that are too far
            if abs(ar_obj.position[2]) > 10:
                self.ar_objects.remove(ar_obj)
        
        # Display info on frame
        cv2.putText(frame, f"Detected: {len(detected_objects)} objects", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"AR Objects: {len(self.ar_objects)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def render_3d(self):
        """Render 3D AR objects"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glLoadIdentity()
        gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
        
        # Render all AR objects
        for ar_obj in self.ar_objects:
            ar_obj.render()
    
    def render_2d_overlay(self, frame: np.ndarray):
        """Render 2D overlay with camera feed"""
        if frame is not None:
            # Convert OpenCV frame to Pygame surface
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            
            # Scale to fit window
            frame_surface = pygame.transform.scale(frame_surface, (self.width // 3, self.height // 3))
            
            # Blit to screen
            self.screen.blit(frame_surface, (10, 10))
        
        # Render UI text
        score_text = self.font.render(f"AR Objects: {len(self.ar_objects)}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.width - 300, 20))
        
        help_text = pygame.font.Font(None, 24).render(
            "Show colored objects to the camera!", True, (200, 200, 200)
        )
        self.screen.blit(help_text, (self.width // 2 - 150, self.height - 30))
    
    def run(self):
        """Main application loop"""
        print("AR 3D Object Detection Game")
        print("Show colored objects (red, green, blue, yellow) to the camera")
        print("AR objects will spawn on detected objects!")
        print("Press ESC to quit")
        
        while self.running:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Update
            frame = self.update(dt)
            
            # Render 3D
            self.render_3d()
            
            # Convert OpenGL to Pygame surface for 2D overlay
            glReadBuffer(GL_BACK)
            pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            gl_surface = pygame.image.fromstring(pixels, (self.width, self.height), 'RGB')
            gl_surface = pygame.transform.flip(gl_surface, False, True)
            self.screen.blit(gl_surface, (0, 0))
            
            # Render 2D overlay
            self.render_2d_overlay(frame)
            
            pygame.display.flip()
            self.clock.tick(60)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    app = ARGamingApp()
    app.run()
