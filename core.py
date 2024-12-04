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
        """Initialize a reference frame.

        Naming convention: T_A_B represents transformation from frame A to frame B
        where A is the parent frame and B is the current frame.
        """
        self.name = name
        self.parent = parent
        self.children = []
        # T_parent_self represents the transformation from parent frame to this frame
        self.T_parent_self = {
            'rotation': Rotation.from_euler('xyz', [0, 0, 0]),
            'position': np.zeros(3)
        }

        if parent:
            parent.add_child(self)

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def get_transform(self):
        """Get the complete transform from root to this frame (T_root_self)"""
        # Start with this frame's transform
        position = self.T_parent_self['position'].copy()
        rotation = self.T_parent_self['rotation']

        # Walk up the tree to accumulate transforms
        current = self.parent
        while current is not None:
            # Transform position by parent's rotation and add parent's position
            position = current.T_parent_self['rotation'].apply(position) + current.T_parent_self['position']
            # Compose rotations
            rotation = current.T_parent_self['rotation'] * rotation
            current = current.parent

        return position, rotation

    def push_transform(self):
        """Push this frame's complete transform to OpenGL matrix stack"""
        position, rotation = self.get_transform()

        glPushMatrix()
        # Apply position first, then rotation
        glTranslatef(*position)
        rot_matrix = rotation.as_matrix()
        gl_matrix = np.eye(4)
        gl_matrix[:3, :3] = rot_matrix
        glMultMatrixf(gl_matrix.T.flatten())

    def pop_transform(self):
        """Pop this frame's transform from OpenGL matrix stack"""
        glPopMatrix()

    def draw_transform_hierarchy(self):
        """Draw vectors showing the transform hierarchy"""
        if self.parent:
            parent_pos, _ = self.parent.get_transform()
            current_pos, _ = self.get_transform()

            # Draw vector from parent to current frame
            glLineWidth(1.0)
            glColor4f(0.5, 0.5, 0.5, 0.7)  # Grey, slightly transparent
            glBegin(GL_LINES)
            glVertex3f(*parent_pos)
            glVertex3f(*current_pos)
            glEnd()

            # Draw text label at midpoint
            mid_point = (parent_pos + current_pos) / 2
            glRasterPos3f(*mid_point)
            for c in self.name:
                glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(c))

        # Recursively draw children's transform vectors
        for child in self.children:
            child.draw_transform_hierarchy()

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
        glColor4f(0.0, 0.0, 1.0, 0.6)  # Blue with transparency
        quad = gluNewQuadric()
        gluQuadricDrawStyle(quad, GLU_LINE)
        gluQuadricOrientation(quad, GLU_OUTSIDE)
        gluQuadricNormals(quad, GLU_SMOOTH)
        gluSphere(quad, self.radius, 30, 30)
        gluDeleteQuadric(quad)

class Satellite(FramedObject):
    def __init__(self, frame, orbit_radius=2.0):
        super().__init__(frame)
        self.orbit_radius = orbit_radius
        self.orbit_angle = 0.0
        self.orbit_speed = 0.01
        # Create a frame that follows the satellite's position
        self.body_frame = ReferenceFrame("Satellite Body", parent=frame)

    def _draw(self):
        # Draw orbit trajectory (centered on Earth)
        glColor4f(0.8, 0.8, 0.8, 0.3)  # Light grey with transparency
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
        # Update body frame position to follow satellite
        x = self.orbit_radius * np.cos(self.orbit_angle)
        y = self.orbit_radius * np.sin(self.orbit_angle)
        self.body_frame.T_parent_self['position'] = np.array([x, y, 0.0])

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

class Sun(FramedObject):
    def __init__(self, frame):
        super().__init__(frame)
        self.radius = 3.0  # Even larger

    def _draw(self):
        glColor3f(1.0, 1.0, 0.0)  # Yellow
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

class Camera:
    def __init__(self):
        self.distance = 15.0  # Distance from target
        self.min_distance = 5.0  # Minimum zoom
        self.max_distance = 30.0  # Maximum zoom
        self.azimuth = 0.0   # Angle in XY plane
        self.elevation = 15.0  # Start with 15-degree elevation (changed from 0.0)
        self.rotation = Rotation.from_euler('xyz', [0, 0, 0])
        # For mouse control
        self.last_mouse = None
        self.mouse_sensitivity = 0.01
        self.zoom_sensitivity = 0.5
        # For target tracking
        self.target_frame = None

    def set_target(self, frame):
        self.target_frame = frame
        print(f"Camera now targeting {frame.name}")

    def get_view_matrix(self):
        if not self.target_frame:
            return

        # Get target's world position
        target_pos, _ = self.target_frame.get_transform()

        # Calculate camera position in spherical coordinates
        x = self.distance * np.cos(np.radians(self.elevation)) * np.cos(np.radians(self.azimuth))
        y = self.distance * np.cos(np.radians(self.elevation)) * np.sin(np.radians(self.azimuth))
        z = self.distance * np.sin(np.radians(self.elevation))

        # Look at the target from the calculated position
        glLoadIdentity()
        gluLookAt(x, y, z,  # Camera position
                  0, 0, 0,   # Look at target (origin)
                  0, 0, 1)   # Up vector (Z-up)

        # Move everything relative to target
        glTranslatef(-target_pos[0], -target_pos[1], -target_pos[2])

    def orbit(self, dx, dy):
        # dx changes azimuth (rotation in XY plane)
        # dy changes elevation (up from XY plane)
        self.azimuth += dx * self.mouse_sensitivity * 50
        self.elevation = np.clip(self.elevation + dy * self.mouse_sensitivity * 50, -89, 89)

    def zoom(self, direction):
        """Adjust camera distance based on scroll direction"""
        self.distance = np.clip(
            self.distance + direction * self.zoom_sensitivity,
            self.min_distance,
            self.max_distance
        )

class LocalLevelFrame(FramedObject):
    def __init__(self, frame, latitude, longitude, size=0.2):
        # Create the frame first
        self.llf_frame = ReferenceFrame("ENU Frame", parent=frame)

        # Initialize FramedObject with our LLF frame
        super().__init__(self.llf_frame)

        self.latitude = np.radians(latitude)
        self.longitude = np.radians(longitude)
        self.size = size
        self.earth_radius = 1.0

        # Calculate position on Earth's surface
        x = self.earth_radius * np.cos(self.latitude) * np.cos(self.longitude)
        y = self.earth_radius * np.cos(self.latitude) * np.sin(self.longitude)
        z = self.earth_radius * np.sin(self.latitude)
        self.frame.T_parent_self['position'] = np.array([x, y, z])

        # Rotation sequence for ENU orientation
        R_z1 = Rotation.from_euler('z', self.longitude)
        R_y = Rotation.from_euler('y', np.pi/2 - self.latitude)
        R_z2 = Rotation.from_euler('z', np.pi/2)

        # Combine rotations in correct order
        self.frame.T_parent_self['rotation'] = R_z1 * R_y * R_z2

        # Draw axes directly in _draw method instead of using Axes object
        self.axis_length = 0.2

    def _draw(self):
        # Draw coordinate axes
        glLineWidth(1.0)

        # X axis - Red (East)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(self.axis_length, 0.0, 0.0)
        glEnd()

        # Y axis - Green (North)
        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, self.axis_length, 0.0)
        glEnd()

        # Z axis - Blue (Up)
        glBegin(GL_LINES)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, self.axis_length)
        glEnd()

        # Draw a red square in the tangent plane (East-North plane)
        glColor3f(1.0, 0.0, 0.0)  # Red
        glBegin(GL_LINE_LOOP)
        s = self.size / 2
        glVertex3f(-s, -s, 0)
        glVertex3f(s, -s, 0)
        glVertex3f(s, s, 0)
        glVertex3f(-s, s, 0)
        glEnd()

        # Draw a point at the origin
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glVertex3f(0, 0, 0)
        glEnd()

class GlobeVisualizer:
    def __init__(self):
        # Create frame hierarchy
        self.root_frame = ReferenceFrame("Root")  # Global fixed frame
        self.helio_frame = ReferenceFrame("Heliocentric", self.root_frame)  # Sun-centered frame
        self.eci_frame = ReferenceFrame("ECI", self.helio_frame)  # ECI relative to helio
        self.ecef_frame = ReferenceFrame("ECEF", self.eci_frame)  # ECEF relative to ECI
        self.orbit_frame = ReferenceFrame("Orbit", self.ecef_frame)  # Satellite orbit frame

        # Create a list to store all drawable objects
        self.drawables = []

        # Create objects and add them to drawables list
        self.sun = Sun(self.helio_frame)
        self.earth = Earth(self.ecef_frame)
        self.satellite = Satellite(self.orbit_frame)

        # Add coordinate axes
        self.helio_axes = Axes(self.helio_frame, length=5.0, is_inertial=True)
        self.eci_axes = Axes(self.eci_frame, length=1.5, is_inertial=True)
        self.ecef_axes = Axes(self.ecef_frame, length=1.5)
        self.orbit_axes = Axes(self.orbit_frame, length=0.5)
        self.satellite_axes = Axes(self.satellite.body_frame, length=0.5)

        # Add a local level frame at a specific location (e.g., 45°N, 45°E)
        self.llf = LocalLevelFrame(self.ecef_frame, latitude=45, longitude=45)

        # Add to drawables list
        self.drawables.extend([
            self.helio_axes,
            self.sun,
            self.earth,
            self.eci_axes,
            self.ecef_axes,
            self.satellite,
            self.orbit_axes,
            self.satellite_axes,
            self.llf
        ])

        # Earth's orbital parameters (now applied to ECI frame)
        self.earth_orbit_radius = 10.0
        self.earth_orbit_angle = 0.0
        self.earth_orbit_speed = 0.01

        # Initialize window and camera
        self.width = 800
        self.height = 600
        self.camera = Camera()
        self.camera.distance = 15.0
        self.mouse_pressed = False

        # Track current frame
        self.frame = FrameType.ECI

        # Rotation rates
        self.rotation_speed = 0.5

        # Add pause state
        self.paused = False

        # Set initial camera target
        self.camera.set_target(self.eci_frame)  # Start with Earth view

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

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)

        # Apply camera view
        self.camera.get_view_matrix()

        # Draw transform hierarchy first (so it's behind other elements)
        self.root_frame.draw_transform_hierarchy()

        # Draw Earth's orbital trajectory
        glLineWidth(1.0)
        glColor4f(0.8, 0.8, 0.8, 0.3)  # Light grey with transparency
        glBegin(GL_LINE_LOOP)
        for i in range(100):
            angle = 2 * np.pi * i / 100
            x = self.earth_orbit_radius * np.cos(angle)
            y = self.earth_orbit_radius * np.sin(angle)
            glVertex3f(x, y, 0)
        glEnd()

        # Draw all objects
        for drawable in self.drawables:
            drawable.draw()

        glutSwapBuffers()

    def idle(self):
        if not self.paused:
            if self.frame == FrameType.ECI:
                # Earth rotates around its axis (increase speed for visibility)
                self.rotation_speed = 2.0
                rot = Rotation.from_euler('z', np.radians(self.rotation_speed))
                self.ecef_frame.T_parent_self['rotation'] = self.ecef_frame.T_parent_self['rotation'] * rot

            # Earth orbits the Sun (move the ECI frame)
            self.earth_orbit_angle += self.earth_orbit_speed
            x = self.earth_orbit_radius * np.cos(self.earth_orbit_angle)
            y = self.earth_orbit_radius * np.sin(self.earth_orbit_angle)
            self.eci_frame.T_parent_self['position'] = np.array([x, y, 0.0])

            # Update satellite
            self.satellite.update()

        glutPostRedisplay()

    def keyboard(self, key, x, y):
        if key == b'f':  # Press 'f' to toggle reference frame
            self.toggle_frame()
        elif key == b' ':  # Spacebar to toggle pause
            self.paused = not self.paused
            print("Simulation " + ("paused" if self.paused else "resumed"))
        elif key == b't':  # 't' to cycle camera target
            if self.camera.target_frame == self.eci_frame:
                self.camera.set_target(self.satellite.body_frame)
            elif self.camera.target_frame == self.satellite.body_frame:
                self.camera.set_target(self.helio_frame)
            else:
                self.camera.set_target(self.eci_frame)
        elif key == b'i':  # 'i' to zoom in
            self.camera.zoom(-1)
            glutPostRedisplay()
        elif key == b'o':  # 'o' to zoom out
            self.camera.zoom(1)
            glutPostRedisplay()
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
        elif button == 0 and state == 0:  # Trackpad scroll up
            self.camera.zoom(-1)
            glutPostRedisplay()
        elif button == 1 and state == 0:  # Trackpad scroll down
            self.camera.zoom(1)
            glutPostRedisplay()
        elif button == 3:  # Mouse wheel up
            self.camera.zoom(-1)
            glutPostRedisplay()
        elif button == 4:  # Mouse wheel down
            self.camera.zoom(1)
            glutPostRedisplay()

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
        print("Use 'i' and 'o' to zoom in/out")
        glutMainLoop()

    def passive_motion(self, x, y):
        print(f"Passive motion: x={x}, y={y}")
        # We'll add scroll handling here once we see the events

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
