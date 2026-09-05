#!/usr/bin/env python3
"""Render preview.png -- a few shaded views of the cover.

Uses a small orthographic z-buffer rasteriser instead of matplotlib's 3D
painter's algorithm, which cannot depth-sort a concave shell correctly.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import generate_cover as gc

W = H = 620
BG = np.array([245, 246, 248], dtype=float)
LIGHT = np.array([0.40, -0.55, 0.74])
LIGHT /= np.linalg.norm(LIGHT)
BASE = np.array([76, 104, 148], dtype=float)


def rot(elev, azim):
    a, e = math.radians(azim), math.radians(elev)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    return rx @ rz


def render(mesh, elev, azim, roll_flip=False):
    R = rot(elev, azim)
    v = mesh.vertices @ R.T
    n = mesh.face_normals @ R.T
    lo, hi = v.min(0), v.max(0)
    c = (lo + hi) / 2
    scale = (W * 0.80) / max(hi[0] - lo[0], hi[2] - lo[2])

    px = (v[:, 0] - c[0]) * scale + W / 2
    py = H / 2 - (v[:, 2] - c[2]) * scale
    depth = v[:, 1]

    img = np.repeat(BG[None, None, :], H, 0).repeat(W, 1)
    zbuf = np.full((H, W), 1e9)
    shade = np.clip(n @ LIGHT, 0, 1) * 0.72 + 0.28

    for f, s in zip(mesh.faces, shade):
        x, y, z = px[f], py[f], depth[f]
        x0, x1 = int(max(0, np.floor(x.min()))), int(min(W - 1, np.ceil(x.max())))
        y0, y1 = int(max(0, np.floor(y.min()))), int(min(H - 1, np.ceil(y.max())))
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = ((y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2]))
        if abs(d) < 1e-9:
            continue
        w0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
        w1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not m.any():
            continue
        zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        hit = m & (zz < sub)
        sub[hit] = zz[hit]
        img[y0:y1 + 1, x0:x1 + 1][hit] = np.clip(BASE * s, 0, 255)

    return Image.fromarray(img.astype(np.uint8))


def label(im, text):
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    for i, line in enumerate(text.split("\n")):
        d.text((14, 12 + i * 21), line, fill=(40, 44, 52), font=font)
    return im


mesh = gc.build()
half = gc.trimesh.boolean.difference(
    [mesh, gc.box(-30, 0, -30, 30, -5, 25)], engine="manifold")

views = [
    (mesh, -28, 205, "outside / back\nBOOT + RST pin holes,\nnotch for USB-C at the top"),
    (mesh, 26, 210, "inside\nthe board drops in from the front\nand the lip snaps over the bezel"),
    (mesh, 6, 180, "top edge\nnotch for the USB-C cable\nand the microSD card"),
    (half, 18, 200, "cut in half\nfloor 1.2 / cavity 10.9 / 45° lip"),
]
tiles = [label(render(m, e, a), t) for m, e, a, t in views]
sheet = Image.new("RGB", (W * 2, H * 2), tuple(BG.astype(int)))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 2) * W, (i // 2) * H))
sheet.save("preview.png")
print("wrote preview.png  %.2f x %.2f x %.2f mm" % tuple(mesh.bounds[1] - mesh.bounds[0]))
