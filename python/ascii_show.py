#!/usr/bin/env python3
"""
ASCII Show - detailed full-color ASCII animations for the terminal.

Scenes
  1  3D      ray-marched spinning torus + bouncing sphere, soft shadows, reflections
  2  Fire    classic fire simulation with wind and drifting embers
  3  Fireworks  rockets, seven burst types, glowing trails over a city skyline

Keys: 1 / 2 / 3 switch scene, space = next scene, q = quit
Run:  python ascii_show.py            (needs: pip install numpy)
      python ascii_show.py --scene 3  (start on fireworks)
      python ascii_show.py --cycle 20 (auto-switch every 20 seconds)
"""
import argparse
import math
import os
import random
import shutil
import sys
import time

import numpy as np

RAMP = np.array(list(" .,:;-=+*cox#%&@"))


# --------------------------------------------------------------------------
# Terminal helpers
# --------------------------------------------------------------------------
class Keys:
    """Non-blocking single-key reader for Windows and Unix terminals."""

    def __init__(self):
        self.win = os.name == "nt"
        self.old = None
        if self.win:
            import msvcrt  # noqa: F401
            os.system("")  # enables ANSI escape codes on Windows 10+
        elif sys.stdin.isatty():
            import termios
            import tty
            self.old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def get(self):
        if self.win:
            import msvcrt
            return msvcrt.getwch() if msvcrt.kbhit() else None
        if not sys.stdin.isatty():
            return None
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def close(self):
        if self.old is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)


def to_ansi(chars, rgb):
    """chars: (H, W) array of str, rgb: (H, W, 3) uint8 -> one ANSI frame string."""
    q = (rgb // 6 * 6).astype(np.int32)  # quantize a little so fewer color codes are emitted
    key = (q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]
    out = ["\x1b[H"]
    H, W = chars.shape
    for y in range(H):
        row_c, row_k, row_q = chars[y], key[y], q[y]
        last = -1
        parts = []
        for x in range(W):
            k = row_k[x]
            if k != last:
                r, g, b = row_q[x]
                parts.append(f"\x1b[38;2;{r};{g};{b}m")
                last = k
            parts.append(row_c[x])
        out.append("".join(parts))
        out.append("\n" if y < H - 1 else "")
    return "".join(out)


def shade_to_chars(lum):
    idx = np.clip((lum * (len(RAMP) - 1)).astype(int), 0, len(RAMP) - 1)
    return RAMP[idx]


# --------------------------------------------------------------------------
# Scene 1: ray-marched 3D
# --------------------------------------------------------------------------
class Raymarch:
    name = "3D"

    def __init__(self, W, H):
        self.resize(W, H)

    def resize(self, W, H):
        self.W, self.H = W, H
        jj, ii = np.mgrid[0:H, 0:W].astype(np.float32)
        # terminal cells are ~2x taller than wide, so one column is half a row wide
        u = (ii - W / 2) / H * 0.5 * 2
        v = -(jj - H / 2) / H * 2
        self.u, self.v = u.ravel(), v.ravel()

    @staticmethod
    def rot(p, a, b):
        x, y, z = p
        ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
        y, z = y * ca - z * sa, y * sa + z * ca
        x, z = x * cb + z * sb, -x * sb + z * cb
        return x, y, z

    def scene(self, p, t):
        x, y, z = p
        # spinning torus
        tx, ty, tz = self.rot((x, y - 1.25, z), t * 0.9, t * 0.6)
        q = np.sqrt(tx * tx + tz * tz) - 1.0
        torus = np.sqrt(q * q + ty * ty) - 0.38
        # bouncing sphere orbiting the torus
        sx, sz = 2.3 * math.cos(t * 0.7), 2.3 * math.sin(t * 0.7)
        sy = 0.5 + abs(math.sin(t * 2.2)) * 1.4
        sphere = np.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) - 0.5
        floor = y
        d = np.minimum(np.minimum(torus, sphere), floor)
        mat = np.where(d == torus, 1, np.where(d == sphere, 2, 0))
        return d, mat

    def march(self, ro, rd, t, steps=64, maxd=20.0):
        n = rd[0].shape[0]
        dist = np.zeros(n, np.float32)
        alive = np.ones(n, bool)
        for _ in range(steps):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            p = tuple(ro[k][idx] + rd[k][idx] * dist[idx] for k in range(3))
            d, _ = self.scene(p, t)
            dist[idx] += d
            alive[idx] = (d > 0.002) & (dist[idx] < maxd)
        return dist

    def normal(self, p, t):
        e = 0.002
        x, y, z = p
        dx = self.scene((x + e, y, z), t)[0] - self.scene((x - e, y, z), t)[0]
        dy = self.scene((x, y + e, z), t)[0] - self.scene((x, y - e, z), t)[0]
        dz = self.scene((x, y, z + e), t)[0] - self.scene((x, y, z - e), t)[0]
        l = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
        return dx / l, dy / l, dz / l

    def soft_shadow(self, p, L, t):
        res = np.ones(p[0].shape, np.float32)
        s = np.full(p[0].shape, 0.05, np.float32)
        for _ in range(28):
            q = tuple(p[k] + L[k] * s for k in range(3))
            d, _ = self.scene(q, t)
            res = np.minimum(res, 8 * d / s)
            s += np.clip(d, 0.02, 0.4)
        return np.clip(res, 0, 1)

    def shade(self, ro, rd, t, depth=0):
        dist = self.march(ro, rd, t)
        hit = dist < 20
        p = tuple(ro[k] + rd[k] * dist for k in range(3))
        d, mat = self.scene(p, t)
        N = self.normal(p, t)
        L = np.array([-0.55, 0.8, -0.45]); L = L / np.linalg.norm(L)
        diff = np.clip(N[0] * L[0] + N[1] * L[1] + N[2] * L[2], 0, 1)
        offs = tuple(p[k] + N[k] * 0.02 for k in range(3))
        shadow = self.soft_shadow(offs, L, t) if depth == 0 else np.ones_like(diff)
        # Blinn-Phong highlight
        hx, hy, hz = L[0] - rd[0], L[1] - rd[1], L[2] - rd[2]
        hl = np.sqrt(hx * hx + hy * hy + hz * hz) + 1e-9
        spec = np.clip((N[0] * hx + N[1] * hy + N[2] * hz) / hl, 0, 1) ** 40

        n = dist.shape[0]
        base = np.zeros((n, 3), np.float32)
        check = ((np.floor(p[0] * 1.2) + np.floor(p[2] * 1.2)) % 2).astype(np.float32)
        floor_col = np.stack([0.10 + 0.22 * check, 0.08 + 0.18 * check, 0.14 + 0.22 * check], 1)
        base[:] = floor_col
        base[mat == 1] = (1.0, 0.45, 0.72)   # pink torus
        base[mat == 2] = (0.55, 0.85, 1.0)   # ice-blue sphere
        fade = np.exp(-dist * 0.08)[:, None]
        light = (0.12 + 0.88 * diff * shadow)[:, None]
        col = base * light + spec[:, None] * shadow[:, None] * 0.9
        col = col * fade

        # glossy floor reflections (one bounce)
        if depth == 0:
            fl = hit & (mat == 0)
            if fl.any():
                idx = np.nonzero(fl)[0]
                rd2 = (rd[0][idx], -rd[1][idx], rd[2][idx])
                ro2 = (p[0][idx], np.full(idx.size, 0.01, np.float32), p[2][idx])
                rcol, rhit = self.shade(ro2, rd2, t, depth=1)
                col[idx] += rcol * 0.3 * rhit[:, None]

        sky_t = np.clip(rd[1] * 1.5 + 0.3, 0, 1)[:, None]
        sky = (1 - sky_t) * np.array([0.22, 0.10, 0.20]) + sky_t * np.array([0.04, 0.03, 0.10])
        col = np.where(hit[:, None], col, sky)
        return col, hit

    def frame(self, t):
        W, H = self.W, self.H
        ang = t * 0.15
        ro_p = np.array([math.sin(ang) * 5.2, 3.3, -math.cos(ang) * 5.2], np.float32)
        target = np.array([0, 1.05, 0], np.float32)
        fwd = target - ro_p; fwd /= np.linalg.norm(fwd)
        right = np.cross([0, 1, 0], fwd); right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        rdx = fwd[0] + self.u * right[0] * 0.55 + self.v * up[0] * 0.55
        rdy = fwd[1] + self.u * right[1] * 0.55 + self.v * up[1] * 0.55
        rdz = fwd[2] + self.u * right[2] * 0.55 + self.v * up[2] * 0.55
        l = np.sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
        rd = (rdx / l, rdy / l, rdz / l)
        n = rdx.shape[0]
        ro = tuple(np.full(n, ro_p[k], np.float32) for k in range(3))
        col, hit = self.shade(ro, rd, t)
        col = np.clip(col, 0, 1)
        lum = col @ np.array([0.3, 0.55, 0.15])
        chars = shade_to_chars(np.clip(lum * 1.6, 0, 1))
        chars[~hit] = " "
        chars = chars.reshape(H, W)
        # sparse twinkling stars in the sky
        star = (~hit) & (np.random.rand(n) < 0.004)
        chars.ravel()[star] = "."
        bright = np.clip(col / np.maximum(col.max(1, keepdims=True), 1e-3), 0, 1) * np.clip(lum[:, None] * 1.3 + 0.25, 0, 1)
        rgb = (bright * 255).astype(np.uint8).reshape(H, W, 3)
        rgb.reshape(-1, 3)[star] = (200, 190, 230)
        return chars, rgb


# --------------------------------------------------------------------------
# Scene 2: fire
# --------------------------------------------------------------------------
class Fire:
    name = "Fire"

    def __init__(self, W, H):
        self.resize(W, H)
        stops = [(0, (0, 0, 0)), (0.2, (90, 8, 6)), (0.4, (200, 40, 10)), (0.6, (255, 120, 20)),
                 (0.8, (255, 210, 80)), (1.0, (255, 255, 230))]
        self.pal = np.zeros((37, 3), np.uint8)
        for i in range(37):
            f = i / 36
            for (a, ca), (b, cb) in zip(stops, stops[1:]):
                if a <= f <= b:
                    k = (f - a) / (b - a)
                    self.pal[i] = [int(ca[c] + (cb[c] - ca[c]) * k) for c in range(3)]
                    break
        self.embers = []

    def resize(self, W, H):
        self.W, self.H = W, H
        self.f = np.zeros((H, W), np.int32)
        self.f[-1, :] = 36

    def frame(self, t):
        W, H, f = self.W, self.H, self.f
        # breathing base so the flames swell and calm
        base = (30 + 6 * np.sin(np.arange(W) * 0.09 + t * 1.7) + 4 * np.sin(np.arange(W) * 0.23 - t)).astype(np.int32)
        f[-1, :] = np.clip(base, 0, 36)
        wind = int(round(math.sin(t * 0.5) * 1.2))
        for y in range(H - 1):
            r = np.random.randint(0, 4, W)
            dst = (np.arange(W) - r + 1 + wind) % W
            f[y, dst] = np.maximum(f[y + 1] - (r & 1) - (np.random.rand(W) < 0.12), 0)
        # floating embers
        if random.random() < 0.6:
            self.embers.append([random.uniform(0, W), H - 3.0, random.uniform(-0.3, 0.3), random.uniform(0.4, 1.0), 1.0])
        lum = f / 36.0
        chars = shade_to_chars(lum ** 0.8)
        rgb = self.pal[f].copy()
        alive = []
        for e in self.embers:
            e[0] += e[2] + math.sin(t * 3 + e[1]) * 0.2
            e[1] -= e[3]
            e[4] -= 0.02
            x, y = int(e[0]) % W, int(e[1])
            if 0 <= y < H and e[4] > 0:
                chars[y, x] = "*" if e[4] > 0.6 else "."
                rgb[y, x] = (255, int(150 + 100 * e[4]), int(60 * e[4]))
                alive.append(e)
        self.embers = alive[-200:]
        return chars, rgb


# --------------------------------------------------------------------------
# Scene 3: fireworks
# --------------------------------------------------------------------------
class Fireworks:
    name = "Fireworks"
    PAL = np.array([(255, 143, 200), (255, 92, 138), (201, 160, 255), (255, 210, 122),
                    (255, 242, 248), (143, 227, 255), (70, 50, 62), (255, 207, 138)], np.uint8)
    FW_RAMP = np.array(list(" .,:-=+*#%@"))

    def __init__(self, W, H):
        self.resize(W, H)
        self.rockets, self.sparks = [], []
        self.next, self.finale, self.last = 0.0, 12.0, None
        self.queue = []

    def resize(self, W, H):
        self.W, self.H = W, H
        self.heat = np.zeros((H, W), np.float32)
        self.hcol = np.zeros((H, W), np.uint8)
        self.sky = []
        x, maxh = 0, max(4, int(H * 0.22))
        while x < W:
            w, h = random.randint(4, 10), random.randint(2, maxh)
            self.sky.append((x, w, h, np.random.rand(h, w) < 0.35))
            x += w + (1 if random.random() < 0.3 else 0)
        self.stars = [(random.randrange(W), random.randrange(max(1, int(H * 0.6))), random.random() * 6) for _ in range(W // 2)]

    def stamp(self, x, y, I, c):
        rx, ry = int(round(x)), int(round(y))
        if 0 <= rx < self.W and 0 <= ry < self.H and I > self.heat[ry, rx]:
            self.heat[ry, rx] = I
            self.hcol[ry, rx] = c

    def launch(self, x, ty):
        y0 = self.H - 1
        ty = max(2, min(self.H * 0.6, ty))
        G = 22
        self.rockets.append(dict(x=x, y=y0, vx=random.uniform(-1.5, 1.5), vy=-math.sqrt(2 * G * (y0 - ty)), G=G, c=random.choice([3, 4])))

    def spark(self, x, y, ang, speed, **o):
        s = dict(x=x, y=y, vx=math.cos(ang) * speed * 2, vy=math.sin(ang) * speed, life=0,
                 max=random.uniform(1.4, 2.0), drag=1.2, g=9, c=0, tw=False, split=False)
        s.update(o)
        self.sparks.append(s)

    def explode(self, r):
        kind = random.choice(["peony", "ring", "willow", "heart", "crossette", "glitter", "double"])
        c = random.choice([0, 1, 2, 4, 5]); big = random.uniform(0.8, 1.3)
        x, y = r["x"], r["y"]
        for a in np.arange(0, 6.28, 0.5):
            self.stamp(x + math.cos(a) * 2, y + math.sin(a), 1, 4)
        if kind in ("peony", "double"):
            for _ in range(110):
                self.spark(x, y, random.uniform(0, 6.28), random.uniform(6, 15) * big, c=c)
            if kind == "double":
                for i in range(50):
                    self.spark(x, y, i / 50 * 6.28, 5 * big, c=random.choice([2, 3, 5]), max=1.2)
        elif kind == "ring":
            tilt = random.uniform(0.4, 1.0)
            for i in range(64):
                a = i / 64 * 6.28
                self.sparks.append(dict(x=x, y=y, vx=math.cos(a) * 13 * big * 2, vy=math.sin(a) * 13 * big * tilt,
                                        life=0, max=1.5, drag=1.4, g=6, c=c, tw=False, split=False))
        elif kind == "willow":
            for _ in range(90):
                self.spark(x, y, random.uniform(0, 6.28), random.uniform(5, 12) * big, c=3,
                           max=random.uniform(3, 4), drag=2.4, g=5)
        elif kind == "heart":
            for i in range(90):
                tt = i / 90 * 6.28
                hx = 16 * math.sin(tt) ** 3
                hy = -(13 * math.cos(tt) - 5 * math.cos(2 * tt) - 2 * math.cos(3 * tt) - math.cos(4 * tt))
                self.sparks.append(dict(x=x, y=y, vx=hx * 0.85 * big * 2, vy=hy * 0.85 * big, life=0, max=1.8,
                                        drag=2.2, g=2.5, c=1, tw=False, split=False))
        elif kind == "crossette":
            for i in range(14):
                self.spark(x, y, i / 14 * 6.28, 11 * big, c=c, max=0.8, split=True)
        else:  # glitter
            for _ in range(120):
                self.spark(x, y, random.uniform(0, 6.28), random.uniform(4, 14) * big,
                           c=random.choice([3, 4]), max=2.2, tw=True, drag=1.8)

    def frame(self, t):
        dt = 0.033 if self.last is None else min(0.05, t - self.last)
        self.last = t
        W, H = self.W, self.H
        if t > self.next:
            self.launch(random.uniform(0.1, 0.9) * W, random.uniform(0.1, 0.45) * H)
            self.next = t + random.uniform(0.5, 1.6)
        if t > self.finale:
            self.queue += [t + i * 0.14 for i in range(6)]
            self.finale = t + random.uniform(20, 30)
        while self.queue and self.queue[0] <= t:
            self.queue.pop(0)
            self.launch(random.uniform(0.1, 0.9) * W, random.uniform(0.08, 0.38) * H)

        self.heat *= math.exp(-dt * 4.2)
        for r in self.rockets:
            r["vy"] += r["G"] * dt; r["x"] += r["vx"] * dt; r["y"] += r["vy"] * dt
            self.stamp(r["x"], r["y"], 1, r["c"])
            if random.random() < 0.6:
                self.sparks.append(dict(x=r["x"], y=r["y"] + 0.5, vx=random.uniform(-2, 2), vy=random.uniform(2, 5),
                                        life=0, max=0.35, drag=2, g=4, c=3, tw=False, split=False))
            if r["vy"] >= -1:
                r["dead"] = True
                self.explode(r)
        self.rockets = [r for r in self.rockets if not r.get("dead")]

        born, keep = [], []
        for p in self.sparks:
            p["life"] += dt
            dr = math.exp(-p["drag"] * dt)
            p["vx"] *= dr; p["vy"] = p["vy"] * dr + p["g"] * dt
            p["x"] += p["vx"] * dt; p["y"] += p["vy"] * dt
            f = 1 - p["life"] / p["max"]
            if f <= 0 or p["y"] > H + 2:
                continue
            if p["split"] and p["life"] > p["max"] * 0.6:
                a0 = random.uniform(0, 3.14)
                for i in range(4):
                    a = a0 + i * math.pi / 2
                    born.append(dict(x=p["x"], y=p["y"], vx=math.cos(a) * 12, vy=math.sin(a) * 6, life=0,
                                     max=0.8, drag=1.5, g=6, c=4, tw=False, split=False))
                continue
            I = f ** 0.6
            if p["tw"]:
                I *= 1 if random.random() < 0.35 else 0.15
            self.stamp(p["x"], p["y"], I, p["c"])
            keep.append(p)
        self.sparks = (keep + born)[-4000:]

        # compose
        chars = np.full((H, W), " ", dtype="<U1")
        g = np.linspace(0, 1, H)[:, None]
        rgb = np.zeros((H, W, 3), np.float32)
        for x, y, ph in self.stars:
            if math.sin(t * 1.3 + ph) > 0.2:
                chars[y, x] = "+" if ph > 5 else "."
                rgb[y, x] = (150, 140, 170)
        lit = self.heat > 0.04
        idx = np.clip(np.ceil(self.heat * (len(self.FW_RAMP) - 1)).astype(int), 0, len(self.FW_RAMP) - 1)
        chars[lit] = self.FW_RAMP[idx[lit]]
        col = self.PAL[self.hcol].astype(np.float32) * np.clip(0.3 + self.heat * 0.9, 0, 1)[..., None]
        rgb[lit] = col[lit]
        for (x0, w, h, wins) in self.sky:
            top = H - h
            for yy in range(h):
                for xx in range(w):
                    X, Y = x0 + xx, top + yy
                    if X >= W:
                        continue
                    if yy == 0:
                        ch = "." if xx in (0, w - 1) else "_"
                    elif xx in (0, w - 1):
                        ch = "|"
                    elif yy % 2 == 1 and xx % 2 == 1:
                        on = bool(wins[yy, xx]) != (math.sin(t * 0.3 + x0 * 3.1 + xx * 7.7 + yy) > 0.97)
                        ch = "o" if on else "."
                        if on:
                            chars[Y, X] = ch; rgb[Y, X] = self.PAL[7]
                            continue
                    else:
                        continue
                    chars[Y, X] = ch
                    rgb[Y, X] = self.PAL[6]
        return chars, rgb.astype(np.uint8)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
SCENES = [Raymarch, Fire, Fireworks]


def term_size():
    s = shutil.get_terminal_size((100, 32))
    return max(40, s.columns), max(16, s.lines - 1)


def main():
    ap = argparse.ArgumentParser(description="Detailed ASCII animations in your terminal.")
    ap.add_argument("--scene", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--cycle", type=float, default=0, help="auto-switch scenes every N seconds")
    ap.add_argument("--fps", type=float, default=30)
    args = ap.parse_args()

    W, H = term_size()
    cur = args.scene - 1
    scene = SCENES[cur](W, H)
    keys = Keys()
    sys.stdout.write("\x1b[?25l\x1b[2J")  # hide cursor, clear
    t0 = time.time(); last_switch = t0; fps_t, frames, fps = t0, 0, 0.0
    try:
        while True:
            k = keys.get()
            if k in ("q", "Q", "\x1b"):
                break
            nxt = None
            if k in ("1", "2", "3"):
                nxt = int(k) - 1
            elif k == " " or (args.cycle and time.time() - last_switch > args.cycle):
                nxt = (cur + 1) % len(SCENES)
            if nxt is not None and nxt != cur:
                cur, last_switch = nxt, time.time()
                scene = SCENES[cur](W, H)
                sys.stdout.write("\x1b[2J")

            nW, nH = term_size()
            if (nW, nH) != (W, H):
                W, H = nW, nH
                scene.resize(W, H)
                sys.stdout.write("\x1b[2J")

            start = time.time()
            chars, rgb = scene.frame(start - t0)
            status = f" [1] 3D  [2] Fire  [3] Fireworks   space next  q quit   {scene.name}  {fps:4.1f} fps"
            sys.stdout.write(to_ansi(chars, rgb) + "\n\x1b[0m\x1b[38;2;255;143;200m" + status[:W].ljust(W) + "\x1b[0m")
            sys.stdout.flush()

            frames += 1
            if start - fps_t > 0.5:
                fps = frames / (start - fps_t); frames, fps_t = 0, start
            time.sleep(max(0, 1 / args.fps - (time.time() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        keys.close()
        sys.stdout.write("\x1b[0m\x1b[?25h\x1b[2J\x1b[H")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
