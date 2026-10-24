import numpy as np


class Track:
    def __init__(self, name, pts, width=70):
        self.name = name
        self.pts = np.asarray(pts, float)
        self.width = width
        self.seg = np.roll(self.pts, -1, axis=0) - self.pts
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        self.seg_len2 = self.seg_len ** 2
        self.cum = np.concatenate([[0], np.cumsum(self.seg_len)[:-1]])
        self.length = self.seg_len.sum()
        self.angle0 = float(np.arctan2(self.seg[0][1], self.seg[0][0]))

    def dist_to_center(self, P):
        """P: (M,2) -> (M,) distance to the centerline."""
        a = P[:, None, :] - self.pts[None]
        t = np.clip((a * self.seg).sum(2) / self.seg_len2, 0, 1)
        c = self.pts[None] + self.seg[None] * t[..., None]
        return np.linalg.norm(P[:, None, :] - c, axis=2).min(1)

    def project(self, p):
        """-> (distance to centerline, arc-length position)."""
        a = p - self.pts
        t = np.clip((a * self.seg).sum(1) / self.seg_len2, 0, 1)
        c = self.pts + self.seg * t[:, None]
        d = np.linalg.norm(p - c, axis=1)
        i = d.argmin()
        return d[i], self.cum[i] + t[i] * self.seg_len[i]

    def point_at(self, s):
        """Arc-length position -> (point on centerline, heading angle)."""
        s %= self.length
        i = np.searchsorted(self.cum, s, "right") - 1
        t = (s - self.cum[i]) / self.seg_len[i]
        p = self.pts[i] + self.seg[i] * t
        return p, float(np.arctan2(self.seg[i][1], self.seg[i][0]))

    def edges(self):
        """-> (left edge points, right edge points)."""
        n = np.c_[-self.seg[:, 1], self.seg[:, 0]] / self.seg_len[:, None]
        return self.pts + n * self.width / 2, self.pts - n * self.width / 2


def _make(name, base=400, sx=1.0, sy=1.0, harmonics=(), width=70, n=240):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.full(n, float(base))
    for k, amp, ph in harmonics:
        r += amp * np.sin(k * th + ph)
    pts = np.c_[r * np.cos(th) * sx, r * np.sin(th) * sy]
    return Track(name, pts, width)


TRACKS = {
    "oval":   _make("oval", sx=1.6, sy=1.0),
    "wavy":   _make("wavy", harmonics=[(3, 60, 0), (5, 30, 1)]),
    "twisty": _make("twisty", harmonics=[(2, 50, 0), (4, 40, 1), (7, 20, 2)], width=65),
    "zigzag": _make("zigzag", sx=1.3, harmonics=[(3, 80, 0), (6, 40, 2)], width=60),
}


if __name__ == "__main__":
    # Quick sanity check: python track.py
    for name, t in TRACKS.items():
        d, s = t.project(t.pts[0])
        print(f"{name:7s} length={t.length:8.1f}  width={t.width}  start_dist={d:.2f}  start_s={s:.2f}")