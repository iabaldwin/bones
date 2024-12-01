#!/usr/bin/env python3

import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from scipy.spatial.transform import Rotation
import math
from enum import Enum
import sys

class FrameType(Enum):
    ECI = "Earth Centered Inertial"
    ECEF = "Earth Centered Earth Fixed"

class ReferenceFrame:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.rotation = Rotation.from_euler('xyz', [0, 0, 0])
        self.position = np.zeros(3)
        
        if parent:
            parent.add_child(self)
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    
    def get_transform(self):
        """Get the complete transform from this frame to root"""
        # Start with this frame's transform
        position = self.position.copy()
        rotation = self.rotation
        
        # Walk up the tree to accumulate transforms
        current = self.parent
        while current is not None:
            # Transform position by parent's rotation and add parent's position
            position = current.rotation.apply(position) + current.position
            # Compose rotations
            rotation = current.rotation * rotation
            current = current.parent
            
        return position, rotation
    
    def push_transform(self):
        """Push this frame's complete transform to OpenGL matrix stack"""
        position, rotation = self.get_transform()
        
        glPushMatrix()
        
        # Apply accumulated transform
        glTranslatef(*position)
        rot_matrix = rotation.as_matrix()
        gl_matrix = np.eye(4)
        gl_matrix[:3, :3] = rot_matrix
        glMultMatrixf(gl_matrix.T.flatten())
    
    def pop_transform(self):
        """Pop this frame's transform from OpenGL matrix stack"""
        glPopMatrix()

class FramedObject:
    """Base class for objects that exist in a reference frame"""
    def __init__(self, frame):
        self.frame = frame
    
    def draw(self):
        self.frame.push_transform()
        self._draw()
        self.frame.pop_transform()
    
    def _draw(self):
        """Override this method to implement specific drawing"""
        pass

class Earth(FramedObject):
    def __init__(self, frame):
        super().__init__(frame)
        self.radius = 1.0
    
    def _draw(self):
        glColor3f(0.0, 0.0, 1.0)
        self.draw_wireframe_sphere()
    
    def draw_wireframe_sphere(self, lats=30, longs=30):
        for i in range(lats):
            lat0 = math.pi * (-0.5 + float(i) / lats)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + float(i + 1) / lats)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)
            
            glBegin(GL_LINE_STRIP)
            for j in range(longs + 1):
                lng = 2 * math.pi * float(j) / longs
                x = math.cos(lng)
                y = math.sin(lng)
                
                glVertex3f(self.radius * x * zr0, self.radius * y * zr0, self.radius * z0)
                glVertex3f(self.radius * x * zr1, self.radius * y * zr1, self.radius * z1)
            glEnd()

class Satellite(FramedObject):
    def __init__(self, frame, orbit_radius=2.0):
        super().__init__(frame)
        self.orbit_radius = orbit_radius
        self.orbit_angle = 0.0
        self.orbit_speed = 0.01
    
    def _draw(self):
        # Draw orbit trajectory (centered on Earth)
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_LINE_LOOP)
        for i in range(100):
            angle = 2 * np.pi * i / 100
            x = self.orbit_radius * np.cos(angle)
            y = self.orbit_radius * np.sin(angle)
            glVertex3f(x, y, 0)
        glEnd()
        
        # Draw satellite at current position
        glColor3f(0.0, 0.0, 1.0)
        glPointSize(10.0)
        glBegin(GL_POINTS)
        x = self.orbit_radius * np.cos(self.orbit_angle)
        y = self.orbit_radius * np.sin(self.orbit_angle)
        glVertex3f(x, y, 0)
        glEnd()
    
    def update(self):
        self.orbit_angle += self.orbit_speed

class Axes(FramedObject):
    def __init__(self, frame, length=1.0, is_inertial=False):
        super().__init__(frame)
        self.length = length
        self.is_inertial = is_inertial
    
    def draw_dashed_line(self, start, end, color):
        glColor3f(*color)
        dash_length = 0.1  # Length of each dash
        direction = end - start
        total_length = np.linalg.norm(direction)
        num_segments = int(total_length / dash_length)
        
        if num_segments == 0:
            return
            
        direction = direction / total_length  # Normalize
        
        glBegin(GL_LINES)
        for i in range(0, num_segments, 2):
            # Calculate start and end of this dash
            dash_start = start + (i * dash_length) * direction
            dash_end = start + min((i + 1) * dash_length, total_length) * direction
            
            glVertex3f(*dash_start)
            glVertex3f(*dash_end)
        glEnd()
    
    def _draw(self):
        glLineWidth(1.0)
        
        if self.is_inertial:
            # Draw dashed axes for inertial frame
            # X axis - Red
            self.draw_dashed_line(
                np.array([0.0, 0.0, 0.0]),
                np.array([self.length, 0.0, 0.0]),
                (1.0, 0.0, 0.0)  # Red
            )
            
            # Y axis - Green
            self.draw_dashed_line(
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, self.length, 0.0]),
                (0.0, 1.0, 0.0)  # Green
            )
            
            # Z axis - Blue
            self.draw_dashed_line(
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, self.length]),
                (0.0, 0.0, 1.0)  # Blue
            )
        else:
            # Solid axes for non-inertial frames
            # X axis - Red
            glBegin(GL_LINES)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(self.length, 0.0, 0.0)
            glEnd()
            
            # Y axis - Green
            glBegin(GL_LINES)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(0.0, self.length, 0.0)
            glEnd()
            
            # Z axis - Blue
            glBegin(GL_LINES)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(0.0, 0.0, self.length)
            glEnd()

class Camera:
    def __init__(self):
        self.distance = 5.0  # Distance from target
        self.azimuth = 0.0   # Horizontal angle
        self.elevation = 0.0  # Vertical angle
        self.rotation = Rotation.from_euler('xyz', [0, 0, 0])
        # For mouse control
        self.last_mouse = None
        self.mouse_sensitivity = 0.01
        
    def update_rotation(self):
        # Convert azimuth and elevation to rotation matrix
        rot_az = Rotation.from_euler('y', np.radians(self.azimuth))
        rot_el = Rotation.from_euler('x', np.radians(self.elevation))
        self.rotation = rot_az * rot_el
        
    def orbit(self, dx, dy):
        # Update azimuth and elevation based on mouse movement
        self.azimuth += dx * self.mouse_sensitivity * 50
        self.elevation = np.clip(self.elevation + dy * self.mouse_sensitivity * 50, -89, 89)
        
        # Update camera rotation
        self.update_rotation()

class GlobeVisualizer:
    def __init__(self):
        # Create frame hierarchy
        self.root_frame = ReferenceFrame("ECI")  # Inertial frame
        self.earth_frame = ReferenceFrame("ECEF", self.root_frame)  # Earth-fixed frame
        self.orbit_frame = ReferenceFrame("Orbit", self.earth_frame)
        
        # Create objects in their respective frames
        self.earth = Earth(self.earth_frame)
        self.satellite = Satellite(self.orbit_frame)
        
        # Add coordinate axes (ECI is inertial)
        self.eci_axes = Axes(self.root_frame, length=1.5, is_inertial=True)
        self.ecef_axes = Axes(self.earth_frame, length=1.5)
        self.orbit_axes = Axes(self.orbit_frame, length=0.5)
        
        # Initialize window and camera
        self.width = 800
        self.height = 600
        self.camera = Camera()
        self.mouse_pressed = False
        
        # Track current frame
        self.frame = FrameType.ECI
        
        # Rotation rate for Earth
        self.rotation_speed = 0.5
        
        # Add pause state
        self.paused = False
    
    def init_gl(self):
        glClearColor(0.95, 0.95, 0.95, 1.0)  # Slightly off-white background
        glEnable(GL_DEPTH_TEST)
        
        # Enable anti-aliasing
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width/self.height), 0.1, 50.0)
    
    def idle(self):
        if not self.paused:
            if self.frame == FrameType.ECI:
                # Rotate around Z axis (north-south)
                rot = Rotation.from_euler('z', np.radians(self.rotation_speed))
                self.earth_frame.rotation = rot * self.earth_frame.rotation
            
            # Always update satellite
            self.satellite.update()
            
        glutPostRedisplay()
    
    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Apply camera transform
        glTranslatef(0.0, 0.0, -self.camera.distance)
        rot_matrix = self.camera.rotation.as_matrix()
        gl_matrix = np.eye(4)
        gl_matrix[:3, :3] = rot_matrix
        glMultMatrixf(gl_matrix.T.flatten())
        
        # Draw all objects with their axes
        self.eci_axes.draw()
        self.earth.draw()
        self.ecef_axes.draw()
        self.satellite.draw()
        self.orbit_axes.draw()
        
        glutSwapBuffers()
    
    def keyboard(self, key, x, y):
        if key == b'f':  # Press 'f' to toggle reference frame
            self.toggle_frame()
        elif key == b' ':  # Spacebar to toggle pause
            self.paused = not self.paused
            print("Simulation " + ("paused" if self.paused else "resumed"))
        elif key == b'\x1b':  # ESC key
            glutDestroyWindow(glutGetWindow())
            sys.exit(0)
        glutPostRedisplay()
    
    def mouse(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self.mouse_pressed = True
                self.camera.last_mouse = (x, y)
            else:
                self.mouse_pressed = False
                self.camera.last_mouse = None
    
    def motion(self, x, y):
        if self.mouse_pressed and self.camera.last_mouse is not None:
            dx = x - self.camera.last_mouse[0]
            dy = y - self.camera.last_mouse[1]
            self.camera.orbit(dx, dy)
            self.camera.last_mouse = (x, y)
            glutPostRedisplay()
    
    def run(self):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow(b"Earth Globe Visualization")
        
        self.init_gl()
        glutDisplayFunc(self.display)
        glutIdleFunc(self.idle)
        glutKeyboardFunc(self.keyboard)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)
        
        print("Press 'f' to toggle between ECI and ECEF frames")
        glutMainLoop()
    
    def toggle_frame(self):
        if self.frame == FrameType.ECI:
            # Switch to ECEF frame
            self.frame = FrameType.ECEF
        else:
            # Switch to ECI frame
            self.frame = FrameType.ECI
            
        print(f"Switched to {self.frame.value}")

if __name__ == "__main__":
    visualizer = GlobeVisualizer()
    visualizer.run()
