#!/usr/bin/env python3
"""
Dirt Happens: turn a folder of GPX files into a static, searchable route map.

    python build_map.py            # build ./site from ./routes
    python build_map.py --serve    # build, then preview at http://127.0.0.1:8000
    python build_map.py --strict   # exit non-zero if any GPX is invalid (used in CI)

Standard library only: nothing to install.

What the build does for every GPX file in routes/:
  * validates it (size, XML safety, coordinates) and skips or rejects bad files
  * computes distance, climb, descent, high/low point, loop / out-and-back / point-to-point
  * removes the first and last N metres of the PUBLISHED track (default 200 m) so your
    front door is not on the map
  * writes a clean GPX for download: no timestamps, heart rate or other sensor data
  * embeds everything in site/index.html, so the result is plain static files
"""
from __future__ import annotations

import argparse
import bisect
import functools
import html
import http.server
import json
import math
import re
import shutil
import socketserver
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

ROOT = Path(__file__).resolve().parent

# --- limits and tuning -------------------------------------------------------
MAX_FILE_BYTES = 15 * 1024 * 1024   # reject absurdly large uploads
MAX_POINTS = 200_000
EARTH_RADIUS_M = 6_371_008.8
MIN_ROUTE_M = 100                   # shorter than this is not a route
LOOP_MAX_GAP_M = 300                # start and finish this close = round trip
SHAPE_MIN_LENGTH_M = 1_000          # do not label very short tracks
RETRACE_TOLERANCE_M = 75            # second half this close to the first = out-and-back
ELE_SMOOTH_WINDOW = 5               # points in the moving average
ELE_NOISE_M = 3.0                   # ignore up/down wiggles smaller than this (GPS noise)
SIMPLIFY_TOLERANCE_M = 4.0          # map line detail (the GPX download is not simplified)
PROFILE_SAMPLES = 120
MAX_TAGS = 8


class GpxError(Exception):
    """A route file that cannot be used. The message is shown to the person who added it."""


# --- text helpers ------------------------------------------------------------
def clean_text(value, max_len: int) -> str:
    if not isinstance(value, str):
        return ""
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()[:max_len]


def clean_tags(value) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,#]", value)
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        tag = re.sub(r"[^a-z0-9 +\-]", "", item.lower())
        tag = re.sub(r"\s+", " ", tag).strip()[:24]
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:MAX_TAGS]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "route"


def safe_json(obj) -> str:
    """JSON that is safe to place inside a <script> tag (no '<', '>' or '&')."""
    text = json.dumps(obj, separators=(",", ":"))  # ensure_ascii escapes the rest
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


# --- GPX parsing -------------------------------------------------------------
def local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def child_text(el, name: str):
    for child in el:
        if local(child.tag) == name and child.text and child.text.strip():
            return child.text.strip()
    return None


def read_point(el):
    try:
        lat, lon = float(el.get("lat")), float(el.get("lon"))
    except (TypeError, ValueError):
        raise GpxError("a track point has a missing or invalid lat/lon") from None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):  # also rejects NaN and inf
        raise GpxError("a track point has coordinates outside the valid range")
    ele = None
    raw = child_text(el, "ele")
    if raw is not None:
        try:
            ele = float(raw)
        except ValueError:
            ele = None
        if ele is not None and not math.isfinite(ele):
            ele = None
    return lat, lon, ele


def parse_gpx(path: Path) -> dict:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise GpxError(f"file is larger than {MAX_FILE_BYTES // 2**20} MB")
    data = path.read_bytes()
    lowered = data.lower()
    # Friends may send files through pull requests, so refuse XML tricks
    # (entity expansion) outright instead of relying on the parser.
    if b"\x00" in data:
        raise GpxError("file must be UTF-8 text")
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise GpxError("DTD and entity declarations are not allowed")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise GpxError(f"not valid XML ({exc})") from None
    if local(root.tag) != "gpx":
        raise GpxError("not a GPX file (root element is not <gpx>)")

    trk = next((e for e in root if local(e.tag) == "trk"), None)
    elements = [
        p
        for t in root
        if local(t.tag) == "trk"
        for seg in t
        if local(seg.tag) == "trkseg"
        for p in seg
        if local(p.tag) == "trkpt"
    ]
    if not elements:  # some apps export a planned route instead of a recorded track
        elements = [
            p for r in root if local(r.tag) == "rte" for p in r if local(p.tag) == "rtept"
        ]
    if len(elements) > MAX_POINTS:
        raise GpxError(f"more than {MAX_POINTS:,} points")
    points = [read_point(p) for p in elements]

    meta = next((e for e in root if local(e.tag) == "metadata"), None)
    name = (child_text(trk, "name") if trk is not None else None) or (
        child_text(meta, "name") if meta is not None else None
    )
    desc = (child_text(trk, "desc") if trk is not None else None) or (
        child_text(meta, "desc") if meta is not None else None
    )
    stamp = (child_text(elements[0], "time") if elements else None) or (
        child_text(meta, "time") if meta is not None else None
    )
    day = None
    match = re.match(r"(\d{4}-\d{2}-\d{2})", stamp or "")
    if match:
        try:
            day = date.fromisoformat(match.group(1)).isoformat()
        except ValueError:
            day = None
    return {"points": points, "name": name, "desc": desc, "date": day}


def load_sidecar(gpx_path: Path) -> dict:
    """Optional route details in a JSON file next to the GPX (same name, .json)."""
    side = gpx_path.with_suffix(".json")
    if side.is_symlink() or not side.is_file():
        return {}
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GpxError(f"{side.name} is not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise GpxError(f"{side.name} must contain a JSON object")
    return data


# --- geometry and statistics -------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def cumulative_distances(points) -> list[float]:
    cum = [0.0]
    for a, b in zip(points, points[1:]):
        cum.append(cum[-1] + haversine_m(a[0], a[1], b[0], b[1]))
    return cum


def smoothed_elevations(points):
    """Moving average of the elevation, or None when the file has no usable elevation."""
    values = [p[2] for p in points]
    if sum(v is not None for v in values) < 0.9 * len(values):
        return None
    last = next(v for v in values if v is not None)
    filled = []
    for v in values:
        last = last if v is None else v
        filled.append(last)
    half = ELE_SMOOTH_WINDOW // 2
    out = []
    for i in range(len(filled)):
        lo, hi = max(0, i - half), min(len(filled), i + half + 1)
        out.append(sum(filled[lo:hi]) / (hi - lo))
    return out


def gain_loss(ele) -> tuple[float, float]:
    """Total climb and descent, ignoring changes smaller than ELE_NOISE_M."""
    gain = loss = 0.0
    ref = ele[0]
    for e in ele[1:]:
        d = e - ref
        if d >= ELE_NOISE_M:
            gain += d
            ref = e
        elif d <= -ELE_NOISE_M:
            loss -= d
            ref = e
    return gain, loss


def classify_shape(points, cum):
    total = cum[-1]
    if total < SHAPE_MIN_LENGTH_M:
        return None
    if haversine_m(points[0][0], points[0][1], points[-1][0], points[-1][1]) > LOOP_MAX_GAP_M:
        return "point-to-point"

    def at(target):
        return points[min(bisect.bisect_left(cum, target), len(points) - 1)]

    half = total / 2
    first_half = [at(half * i / 150) for i in range(151)]
    samples, hits = 60, 0
    for j in range(samples):
        p = at(half + half * j / (samples - 1))
        nearest = min(haversine_m(p[0], p[1], q[0], q[1]) for q in first_half)
        hits += nearest <= RETRACE_TOLERANCE_M
    return "out-and-back" if hits / samples >= 0.7 else "loop"


def trim_range(cum, trim_m: float) -> tuple[int, int, float]:
    """Index range (inclusive) left after removing trim_m from each end."""
    total = cum[-1]
    trim = min(max(trim_m, 0.0), total / 4)
    if trim <= 0:
        return 0, len(cum) - 1, 0.0
    lo = bisect.bisect_left(cum, trim)
    hi = bisect.bisect_right(cum, total - trim) - 1
    if hi - lo < 1:
        return 0, len(cum) - 1, 0.0
    return lo, hi, trim


def simplify_indices(xy, tolerance: float) -> list[int]:
    """Douglas-Peucker line simplification, iterative so long tracks cannot overflow the stack."""
    n = len(xy)
    if n < 3:
        return list(range(n))
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        ax, ay = xy[a]
        dx, dy = xy[b][0] - ax, xy[b][1] - ay
        seg = dx * dx + dy * dy
        worst, worst_i = -1.0, -1
        for i in range(a + 1, b):
            px, py = xy[i]
            if seg == 0:
                d = math.hypot(px - ax, py - ay)
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg))
                d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if d > worst:
                worst, worst_i = d, i
        if worst > tolerance:
            keep[worst_i] = True
            stack.append((a, worst_i))
            stack.append((worst_i, b))
    return [i for i, k in enumerate(keep) if k]


def project(points):
    lat0 = math.radians(sum(p[0] for p in points) / len(points))
    kx = EARTH_RADIUS_M * math.cos(lat0)
    return [(math.radians(p[1]) * kx, math.radians(p[0]) * EARTH_RADIUS_M) for p in points]


def elevation_profile(points, ele, cum, lo: int, hi: int):
    if ele is None or cum[hi] - cum[lo] <= 0:
        return None
    start, end = cum[lo], cum[hi]
    out = []
    for j in range(PROFILE_SAMPLES):
        target = start + (end - start) * j / (PROFILE_SAMPLES - 1)
        i = min(max(bisect.bisect_left(cum, target, lo, hi + 1), lo), hi)
        out.append(
            [round(cum[i] / 1000, 3), round(ele[i], 1), round(points[i][0], 5), round(points[i][1], 5)]
        )
    return out


# --- output ------------------------------------------------------------------
def gpx_document(name: str, desc: str, points) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="Dirt Happens" xmlns="http://www.topografix.com/GPX/1/1">',
        "  <trk>",
        f"    <name>{xml_escape(name)}</name>",
    ]
    if desc:
        lines.append(f"    <desc>{xml_escape(desc)}</desc>")
    lines.append("    <trkseg>")
    for lat, lon, ele in points:
        if ele is None:
            lines.append(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}"/>')
        else:
            lines.append(f'      <trkpt lat="{lat:.6f}" lon="{lon:.6f}"><ele>{ele:.1f}</ele></trkpt>')
    lines += ["    </trkseg>", "  </trk>", "</gpx>", ""]
    return "\n".join(lines)


def process_route(path: Path, routes_dir: Path, trim_m: float, used_ids: set, out_dir: Path) -> dict:
    parsed = parse_gpx(path)
    points = parsed["points"]
    if len(points) < 2:
        raise GpxError("fewer than 2 track points")
    cum = cumulative_distances(points)
    total = cum[-1]
    if total < MIN_ROUTE_M:
        raise GpxError(f"route is shorter than {MIN_ROUTE_M} m")

    side = load_sidecar(path)
    rel = path.relative_to(routes_dir)
    name = clean_text(
        side.get("name") or parsed["name"] or path.stem.replace("_", " ").replace("-", " "), 120
    )
    description = clean_text(side.get("description") or parsed["desc"] or "", 600)
    author = clean_text(side.get("author") or (rel.parts[0] if len(rel.parts) > 1 else ""), 40)

    ele = smoothed_elevations(points)
    gain = loss = low = high = None
    if ele is not None:
        gain, loss = gain_loss(ele)
        low, high = min(ele), max(ele)

    shape = classify_shape(points, cum)
    tags = ([shape] if shape else []) + [t for t in clean_tags(side.get("tags")) if t != shape]

    lo, hi, trimmed = trim_range(cum, trim_m)
    shown = points[lo : hi + 1]
    keep = simplify_indices(project(shown), SIMPLIFY_TOLERANCE_M)
    coords = [[round(shown[i][0], 5), round(shown[i][1], 5)] for i in keep]
    lats, lons = [c[0] for c in coords], [c[1] for c in coords]

    base = slugify("-".join(rel.with_suffix("").parts))
    route_id, n = base, 2
    while route_id in used_ids:
        route_id, n = f"{base}-{n}", n + 1
    used_ids.add(route_id)

    (out_dir / "gpx" / f"{route_id}.gpx").write_text(gpx_document(name, description, shown), encoding="utf-8")

    return {
        "id": route_id,
        "name": name,
        "author": author,
        "description": description,
        "tags": tags[:MAX_TAGS],
        "date": parsed["date"],
        "distance_km": round(total / 1000, 2),
        "gain_m": None if gain is None else round(gain),
        "loss_m": None if loss is None else round(loss),
        "min_ele": None if low is None else round(low),
        "max_ele": None if high is None else round(high),
        "trimmed_m": round(trimmed),
        "gpx": f"gpx/{route_id}.gpx",
        "bounds": [[min(lats), min(lons)], [max(lats), max(lons)]],
        "coords": coords,
        "profile": elevation_profile(points, ele, cum, lo, hi),
    }


def load_config() -> dict:
    config = {"title": "Dirt Happens", "tagline": "Trail running routes", "repo_url": "", "privacy_trim_m": 200}
    path = ROOT / "config.json"
    if path.is_file():
        try:
            user = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            sys.exit(f"config.json is not valid JSON: {exc}")
        if isinstance(user, dict):
            config.update({k: v for k, v in user.items() if k in config})
    config["title"] = clean_text(config["title"], 60) or "Dirt Happens"
    config["tagline"] = clean_text(config["tagline"], 120)
    url = config["repo_url"] if isinstance(config["repo_url"], str) else ""
    config["repo_url"] = url if re.fullmatch(r"https://[^\s\"'<>]+", url) else ""
    trim = config["privacy_trim_m"]
    config["privacy_trim_m"] = trim if isinstance(trim, (int, float)) and trim >= 0 else 200
    return config


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def prepare_output(out: Path, routes_dir: Path) -> None:
    out = out.resolve()
    if out == ROOT or out in ROOT.parents or out == Path(out.anchor) or is_inside(routes_dir.resolve(), out):
        sys.exit(f"Refusing to use {out} as the output folder.")
    if out.exists():
        shutil.rmtree(out)
    (out / "gpx").mkdir(parents=True)


def find_gpx_files(routes_dir: Path, problems: list):
    base = routes_dir.resolve()
    files = []
    for p in sorted(routes_dir.rglob("*")):
        if p.suffix.lower() != ".gpx":
            continue
        if p.is_symlink() or not is_inside(p.resolve(), base):
            problems.append((p, "symbolic links are not allowed"))
        elif p.is_file():
            files.append(p)
    return files


def render_page(config: dict, routes: list) -> str:
    template = (ROOT / "template.html").read_text(encoding="utf-8")
    values = {
        "TITLE": html.escape(config["title"]),
        "TAGLINE": html.escape(config["tagline"]),
        "DATA": safe_json({"repoUrl": config["repo_url"], "routes": routes}),
    }
    return re.sub(r"__(TITLE|TAGLINE|DATA)__", lambda m: values[m.group(1)], template)


def serve(out: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"\nPreview: http://127.0.0.1:{port}   (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Dirt Happens route map from a folder of GPX files.")
    ap.add_argument("--routes", default="routes", help="folder with GPX files (default: routes)")
    ap.add_argument("--out", default="site", help="output folder (default: site)")
    ap.add_argument("--trim", type=float, default=None, help="metres to cut from each end of published tracks")
    ap.add_argument("--strict", action="store_true", help="exit with an error if any route file is invalid")
    ap.add_argument("--serve", action="store_true", help="preview the result on http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    config = load_config()
    trim_m = args.trim if args.trim is not None else config["privacy_trim_m"]
    routes_dir = (ROOT / args.routes).resolve()
    out_dir = (ROOT / args.out).resolve()
    if not routes_dir.is_dir():
        sys.exit(f"Routes folder not found: {routes_dir}")

    prepare_output(out_dir, routes_dir)
    problems: list[tuple[Path, str]] = []
    routes, used_ids = [], set()
    for path in find_gpx_files(routes_dir, problems):
        try:
            routes.append(process_route(path, routes_dir, trim_m, used_ids, out_dir))
        except GpxError as exc:
            problems.append((path, str(exc)))
    routes.sort(key=lambda r: r["name"].casefold())

    (out_dir / "index.html").write_text(render_page(config, routes), encoding="utf-8")

    for path, reason in problems:
        print(f"  SKIPPED {path.relative_to(routes_dir)}: {reason}", file=sys.stderr)
    print(f"Built {len(routes)} route(s) into {out_dir.relative_to(ROOT) if is_inside(out_dir, ROOT) else out_dir}")
    for r in routes:
        climb = "n/a" if r["gain_m"] is None else f"+{r['gain_m']} m"
        print(f"  {r['name'][:38]:<38} {r['distance_km']:>6.1f} km {climb:>9}  {', '.join(r['tags'])}")
    if not routes:
        print("No routes yet. Add .gpx files to the routes/ folder (or try: python build_map.py --routes sample_routes).")
    if problems and args.strict:
        sys.exit(f"{len(problems)} route file(s) could not be used.")
    if args.serve:
        serve(out_dir, args.port)


if __name__ == "__main__":
    main()
