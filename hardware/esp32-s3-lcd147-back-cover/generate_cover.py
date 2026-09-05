#!/usr/bin/env python3
"""
Parametric 3D-printable back cover for the Waveshare ESP32-S3-Touch-LCD-1.47
(1.47" 172x320 display board that ships with a front shell and an open back).

The cover is a "phone case" style sleeve: the board drops in from the front and
is held by a thin lip that overlaps the front bezel.  This only relies on the
outer dimensions of the front shell (24.55 x 44.50 x 10.60, R5.75), which are
the numbers given in the official Waveshare outline drawing, so it fits without
knowing anything about the inside of the front shell.

Run:  python3 generate_cover.py
Deps: numpy, trimesh, manifold3d  (matplotlib only for the preview image)
"""

import math
import os

import numpy as np
import trimesh

# --------------------------------------------------------------------------
# Device dimensions -- from the Waveshare mechanical drawing (mm)
# --------------------------------------------------------------------------
DEV_W = 24.55          # front shell outer width
DEV_H = 44.50          # front shell outer height
DEV_R = 5.75           # front shell corner radius
DEV_T = 10.60          # total thickness, front glass -> tallest part on the back

PCB_H = 39.00          # PCB height; shell overhangs it by 2.75 top and bottom
PCB_TOP_GAP = (DEV_H - PCB_H) / 2.0

M2_DX = 17.78          # M2 mounting hole pattern, horizontal
M2_DY = 25.40          # M2 mounting hole pattern, vertical
BTN_FROM_PCB_TOP = 5.00   # BOOT / RST buttons, from the top edge of the PCB

# Active display area, for reference: 17.75 x 32.93, centred.
# -> 3.40 mm of bezel left/right, 5.79 mm top/bottom.  The lip must stay below
#    those numbers so it never covers the screen.
SCREEN_MARGIN_X = (DEV_W - 17.75) / 2.0
SCREEN_MARGIN_Y = (DEV_H - 32.93) / 2.0

# --------------------------------------------------------------------------
# Cover parameters -- tune these
# --------------------------------------------------------------------------
CLEAR = 0.30           # gap around the device (increase for a looser fit)
WALL = 1.20            # side wall thickness
FLOOR = 1.20           # back floor thickness
LIP_W = 1.10           # how far the front lip overlaps the bezel
LIP_H = 1.10           # height of the lip; == LIP_W keeps the underside at 45 deg

TOP_NOTCH_W = 13.00    # opening in the top wall for USB-C + microSD
RELIEF_LEN = 18.00     # lip is cut away over this length mid-way down each long
                       # side, so the board can be snapped in and thumbed out

BUTTON_ACCESS = "floor"   # "floor" | "side" | "none"  -- BOOT / RST access
BUTTON_HOLE_D = 3.50
BUTTON_DX = M2_DX / 2.0   # buttons sit roughly above the M2 hole columns

HEADER_SLOTS = False   # cut two slots in the floor for soldered 2.54 mm headers
HEADER_DX = 10.00      # slot centre distance from the middle of the board
HEADER_SLOT_W = 3.20
HEADER_SLOT_L = 29.00

EXTRA_DEPTH = 0.0      # extra space between the board and the floor (battery)
BOTTOM_CHAMFER = 0.60  # chamfer on the outer bottom edge; hides elephant foot

SEG = 24               # arc segments per rounded corner

# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def rounded_rect(w, h, r, seg=SEG):
    """Closed CCW outline of a rounded rectangle centred on the origin.

    Always returns 4 * (seg + 1) points in the same order regardless of `r`,
    so two outlines can be lofted vertex-to-vertex.
    """
    cx, cy = w / 2.0 - r, h / 2.0 - r
    pts = []
    for ox, oy, a0 in ((cx, cy, 0.0), (-cx, cy, 90.0), (-cx, -cy, 180.0), (cx, -cy, 270.0)):
        for i in range(seg + 1):
            a = math.radians(a0 + 90.0 * i / seg)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    return pts


def _skin(bottom, top):
    """Solid between two matching convex rings, with fan-triangulated caps."""
    n = len(bottom)
    verts = list(bottom) + list(top)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    # caps: both rings are convex, so a fan from vertex 0 is valid
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])                      # bottom, facing -Z
        faces.append([n, n + i, n + i + 1])              # top, facing +Z
    mesh = trimesh.Trimesh(vertices=np.array(verts, dtype=float),
                           faces=np.array(faces, dtype=np.int64), process=True)
    mesh.fix_normals()
    return mesh


def prism(w, h, r, z0, z1):
    ring = rounded_rect(w, h, r)
    return _skin([(x, y, z0) for x, y in ring], [(x, y, z1) for x, y in ring])


def taper(w0, h0, r0, z0, w1, h1, r1, z1):
    return _skin([(x, y, z0) for x, y in rounded_rect(w0, h0, r0)],
                 [(x, y, z1) for x, y in rounded_rect(w1, h1, r1)])


def box(x0, x1, y0, y1, z0, z1):
    m = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    m.apply_translation(((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0))
    return m


def cyl(d, x, y, z0, z1):
    m = trimesh.creation.cylinder(radius=d / 2.0, height=z1 - z0, sections=48)
    m.apply_translation((x, y, (z0 + z1) / 2.0))
    return m


def cut(solid, *tools):
    return trimesh.boolean.difference([solid, *tools], engine="manifold")


# --------------------------------------------------------------------------
# The cover
# --------------------------------------------------------------------------


def build(button_access=BUTTON_ACCESS, header_slots=HEADER_SLOTS,
          extra_depth=EXTRA_DEPTH):
    # cavity the device sits in
    cav_w = DEV_W + 2 * CLEAR
    cav_h = DEV_H + 2 * CLEAR
    cav_r = DEV_R + CLEAR
    cav_d = DEV_T + CLEAR + extra_depth

    # front opening, i.e. the cavity minus the retaining lip
    win_w = cav_w - 2 * LIP_W
    win_h = cav_h - 2 * LIP_W
    win_r = max(cav_r - LIP_W, 0.6)

    out_w = cav_w + 2 * WALL
    out_h = cav_h + 2 * WALL
    out_r = cav_r + WALL
    out_d = FLOOR + cav_d + LIP_H

    z_cav_top = FLOOR + cav_d          # front face of the device sits here
    eps = 0.01

    part = trimesh.boolean.union([
        taper(out_w - 2 * BOTTOM_CHAMFER, out_h - 2 * BOTTOM_CHAMFER,
              out_r - BOTTOM_CHAMFER, 0.0,
              out_w, out_h, out_r, BOTTOM_CHAMFER),
        prism(out_w, out_h, out_r, BOTTOM_CHAMFER, out_d),
    ], engine="manifold")

    tools = [
        # main pocket
        prism(cav_w, cav_h, cav_r, FLOOR, z_cav_top + eps),
        # 45 deg chamfer under the lip: self-supporting when printed floor down
        # and it guides the board in
        taper(cav_w, cav_h, cav_r, z_cav_top,
              win_w, win_h, win_r, z_cav_top + LIP_H),
        # front opening
        prism(win_w, win_h, win_r, z_cav_top + LIP_H - eps, out_d + 1.0),
        # USB-C + microSD notch in the top wall
        box(-TOP_NOTCH_W / 2, TOP_NOTCH_W / 2,
            cav_h / 2 - 2.0, out_h, FLOOR, out_d + 1.0),
    ]

    # lip relief half way down both long sides
    for sx in (-1.0, 1.0):
        x0, x1 = sorted((sx * (win_w / 2 - eps), sx * (cav_w / 2 + eps)))
        tools.append(box(x0, x1, -RELIEF_LEN / 2, RELIEF_LEN / 2,
                         z_cav_top - eps, out_d + eps))

    # BOOT / RST access
    y_btn = DEV_H / 2 - (PCB_TOP_GAP + BTN_FROM_PCB_TOP)
    if button_access == "floor":
        for sx in (-1.0, 1.0):
            tools.append(cyl(BUTTON_HOLE_D, sx * BUTTON_DX, y_btn, -1.0, FLOOR + 1.0))
    elif button_access == "side":
        for sx in (-1.0, 1.0):
            m = trimesh.creation.cylinder(radius=BUTTON_HOLE_D / 2.0, height=6.0,
                                          sections=48)
            m.apply_transform(trimesh.transformations.rotation_matrix(
                math.pi / 2, [0, 1, 0]))
            m.apply_translation((sx * (cav_w / 2 + WALL / 2), y_btn,
                                 FLOOR + cav_d / 2))
            tools.append(m)

    # slots for soldered 2.54 mm headers
    if header_slots:
        for sx in (-1.0, 1.0):
            tools.append(box(sx * HEADER_DX - HEADER_SLOT_W / 2,
                             sx * HEADER_DX + HEADER_SLOT_W / 2,
                             -HEADER_SLOT_L / 2, HEADER_SLOT_L / 2,
                             -1.0, FLOOR + 1.0))

    part = cut(part, *tools)
    part.merge_vertices()
    part.fix_normals()
    return part


VARIANTS = {
    "back-cover-standard": dict(),
    "back-cover-headers": dict(header_slots=True),
    "back-cover-battery": dict(extra_depth=5.0),
}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "stl")
    os.makedirs(out_dir, exist_ok=True)
    meshes = {}
    for name, kwargs in VARIANTS.items():
        mesh = build(**kwargs)
        path = os.path.join(out_dir, name + ".stl")
        mesh.export(path)
        meshes[name] = mesh
        bb = mesh.bounds[1] - mesh.bounds[0]
        print("%-22s %5d tri  watertight=%-5s  %.2f x %.2f x %.2f mm  %.2f cm3"
              % (name, len(mesh.faces), mesh.is_watertight,
                 bb[0], bb[1], bb[2], mesh.volume / 1000.0))
    return meshes


if __name__ == "__main__":
    main()
