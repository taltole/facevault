"""
FaceVault Utils — OpenGL 3D Store Attention Heatmap.

Renders a bird's-eye view of the store floor plan with height-mapped
attention bars per zone using GLUT/GLU.

Zone height = total customer dwell time (seconds) in that zone.
Colour gradient: blue (low) -> yellow -> red (high attention).

Based on actual utils_gl.py by Tal Toledano.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger("facevault.gl")


def render_attention_heatmap_3d(zone_data: dict,
                                 title: str = "FaceVault — 3D Attention Heatmap",
                                 auto_close_sec: float = 0.0) -> None:
    """
    Open a GLUT window showing the store floor plan as a 3D bar chart.

    Args:
        zone_data:       {zone_name: {"bbox": (x1,y1,x2,y2), "dwell": float}}
        title:           Window title
        auto_close_sec:  If > 0, close window automatically after N seconds

    Each zone is rendered as a 3D bar:
      - XZ footprint = zone bounding box (normalised to [-1, 1])
      - Y height     = normalised dwell time
      - Colour       = heat gradient (cold blue -> hot red)
    """
    try:
        from OpenGL.GL   import (glClear, glClearColor, glColor3f, glBegin, glEnd,
                                  glVertex3f, glEnable, glDepthFunc, glMatrixMode,
                                  glLoadIdentity, glOrtho, glLineWidth,
                                  GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                                  GL_TRIANGLES, GL_LINES, GL_DEPTH_TEST,
                                  GL_LESS, GL_MODELVIEW, GL_PROJECTION, GL_QUADS)
        from OpenGL.GLUT import (glutInit, glutInitDisplayMode, glutInitWindowSize,
                                  glutCreateWindow, glutDisplayFunc, glutIdleFunc,
                                  glutMainLoop, glutSwapBuffers,
                                  GLUT_DOUBLE, GLUT_RGB, GLUT_DEPTH,
                                  glutPostRedisplay)
        from OpenGL.GLU  import gluPerspective, gluLookAt
        import time

        if not zone_data:
            logger.warning("render_attention_heatmap_3d: no zone data")
            return

        # Normalise dwell values to [0, 1]
        max_dwell = max((v["dwell"] for v in zone_data.values()), default=1.0)
        max_dwell = max(max_dwell, 1.0)

        start_time = time.monotonic()

        def _heat_colour(t: float):
            """Map t in [0,1] to RGB heat (blue->cyan->green->yellow->red)."""
            if t < 0.25:
                r, g, b = 0.0, t*4, 1.0
            elif t < 0.5:
                r, g, b = 0.0, 1.0, 1.0 - (t-0.25)*4
            elif t < 0.75:
                r, g, b = (t-0.5)*4, 1.0, 0.0
            else:
                r, g, b = 1.0, 1.0 - (t-0.75)*4, 0.0
            return r, g, b

        def _draw_bar(x1, z1, x2, z2, height, colour):
            """Draw a solid 3D bar (box) for one zone."""
            r, g, b = colour
            glColor3f(r, g, b)
            glBegin(GL_QUADS)
            # Top face
            glVertex3f(x1, height, z1)
            glVertex3f(x2, height, z1)
            glVertex3f(x2, height, z2)
            glVertex3f(x1, height, z2)
            # Front face
            glVertex3f(x1, 0, z1)
            glVertex3f(x2, 0, z1)
            glVertex3f(x2, height, z1)
            glVertex3f(x1, height, z1)
            # Side faces
            glVertex3f(x2, 0, z1)
            glVertex3f(x2, 0, z2)
            glVertex3f(x2, height, z2)
            glVertex3f(x2, height, z1)
            glVertex3f(x1, 0, z2)
            glVertex3f(x1, 0, z1)
            glVertex3f(x1, height, z1)
            glVertex3f(x1, height, z2)
            # Back face
            glVertex3f(x1, 0, z2)
            glVertex3f(x2, 0, z2)
            glVertex3f(x2, height, z2)
            glVertex3f(x1, height, z2)
            glEnd()

        def display():
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            gluLookAt(0, 2.5, 3,    # camera position
                      0, 0,   0,    # look at
                      0, 1,   0)    # up vector

            # Draw floor grid
            glColor3f(0.25, 0.25, 0.35)
            glLineWidth(1.0)
            glBegin(GL_LINES)
            for i in range(-10, 11):
                glVertex3f(i*0.1, 0, -1)
                glVertex3f(i*0.1, 0,  1)
                glVertex3f(-1, 0, i*0.1)
                glVertex3f( 1, 0, i*0.1)
            glEnd()

            # Draw each zone bar
            all_bboxes = [v["bbox"] for v in zone_data.values()]
            frame_w = max(b[2] for b in all_bboxes) or 1280
            frame_h = max(b[3] for b in all_bboxes) or 540

            for zone_name, zd in zone_data.items():
                bx1, by1, bx2, by2 = zd["bbox"]
                dwell  = zd["dwell"]
                height = (dwell / max_dwell) * 1.5   # max bar height = 1.5 units

                # Normalise bbox to [-1, 1]
                nx1 = (bx1 / frame_w) * 2 - 1
                nx2 = (bx2 / frame_w) * 2 - 1
                nz1 = (by1 / frame_h) * 2 - 1
                nz2 = (by2 / frame_h) * 2 - 1

                colour = _heat_colour(dwell / max_dwell)
                _draw_bar(nx1, nz1, nx2, nz2, max(height, 0.02), colour)

            glutSwapBuffers()

            # Auto-close
            if auto_close_sec > 0:
                if time.monotonic() - start_time > auto_close_sec:
                    import sys
                    sys.exit(0)

            glutPostRedisplay()

        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(800, 600)
        glutCreateWindow(title.encode())

        glClearColor(0.1, 0.1, 0.15, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, 800/600, 0.1, 10.0)

        glutDisplayFunc(display)
        glutMainLoop()

    except ImportError:
        logger.warning("PyOpenGL not installed — rendering ASCII fallback")
        _ascii_heatmap(zone_data)
    except Exception as e:
        logger.error(f"OpenGL render failed: {e}")
        _ascii_heatmap(zone_data)


def draw_face(vertices: np.ndarray, faces: np.ndarray,
              texture: Optional[np.ndarray] = None) -> None:
    """
    Render a triangulated 3D face mesh using OpenGL.
    Used for real-time 3D face visualisation in the staff dashboard.

    Args:
        vertices: (N, 3) float32 array of vertex positions
        faces:    (M, 3) int32 array of triangle indices
        texture:  Optional (H, W, 3) uint8 texture image
    """
    try:
        from OpenGL.GL import (glClear, glColor3f, glBegin, glEnd, glVertex3f,
                                GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_TRIANGLES)
        from OpenGL.GLUT import glutSwapBuffers

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor3f(1.0, 0.9, 0.7)   # skin tone

        glBegin(GL_TRIANGLES)
        for face in faces:
            for idx in face:
                v = vertices[idx]
                glVertex3f(float(v[0]), float(v[1]), float(v[2]))
        glEnd()
        glutSwapBuffers()

    except ImportError:
        pass


def _ascii_heatmap(zone_data: dict) -> None:
    """Terminal fallback when PyOpenGL is unavailable."""
    max_d = max((v["dwell"] for v in zone_data.values()), default=1)
    max_d = max(max_d, 1)
    print("\n  FaceVault — Zone Attention (ASCII)")
    print("  " + "─" * 42)
    for name, zd in zone_data.items():
        bar_len = int((zd["dwell"] / max_d) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {name:<12} {bar}  {zd['dwell']:.0f}s")
    print("  " + "─" * 42 + "\n")
