"""SVG charts rendered from recorded data, with a small script for hover readouts and toggles.

Every builder returns an HTML string for ``st.html(..., unsafe_allow_javascript=True)``. Text
is escaped and embedded JSON cannot close its script tag.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from html import escape

from wsn.models import PlainData

CHART_WIDTH = 960
MARGIN_LEFT = 56
MARGIN_RIGHT = 16
MARGIN_TOP = 28
MARGIN_BOTTOM = 40
MAX_POINTS = 480


class ValueFormat(StrEnum):
    INTEGER = "integer"
    DECIMAL = "decimal"
    PERCENT = "percent"
    JOULES = "joules"
    PERCENT_SHARE = "share"
    """A 0-1 share shown as a percentage."""


@dataclass(frozen=True, slots=True)
class Series:
    key: str
    name: str
    color: str
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    emphasis: bool = True


def element_id(prefix: str, *parts: str) -> str:
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=6).hexdigest()
    return f"{prefix}-{digest}"


def json_for_script(data: PlainData) -> str:
    """JSON that is safe inside a ``<script>`` element.

    Every ``<`` becomes ``\\u003c``: the script can never be closed early, and the sanitizer
    (which drops elements whose text looks like markup) leaves the script in place.
    """
    return json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")


def nice_ceiling(value: float) -> float:
    """Round up to 1, 2, 2.5 or 5 times a power of ten."""
    if value <= 0:
        return 1.0
    power = math.pow(10.0, math.floor(math.log10(value)))
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        if value <= step * power:
            return step * power
    return 10.0 * power


def format_number(value: float, style: ValueFormat) -> str:
    if style is ValueFormat.PERCENT_SHARE:
        return f"{value * 100:.0f}%"
    if style is ValueFormat.PERCENT:
        return f"{value:.1f}%"
    if style is ValueFormat.JOULES:
        return f"{value:.3f} J"
    if style is ValueFormat.DECIMAL:
        return f"{value:,.2f}"
    return f"{value:,.0f}"


def _downsample(xs: Sequence[float], ys: Sequence[float]) -> tuple[list[float], list[float]]:
    if len(xs) <= MAX_POINTS:
        return list(xs), list(ys)
    step = math.ceil(len(xs) / MAX_POINTS)
    keep = list(range(0, len(xs), step))
    if keep[-1] != len(xs) - 1:
        keep.append(len(xs) - 1)
    return [xs[i] for i in keep], [ys[i] for i in keep]


@dataclass(frozen=True, slots=True)
class _Frame:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    height: int

    def x(self, value: float) -> float:
        span = self.x_max - self.x_min or 1.0
        inner = CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
        return MARGIN_LEFT + (value - self.x_min) / span * inner

    def y(self, value: float) -> float:
        inner = self.height - MARGIN_TOP - MARGIN_BOTTOM
        span = self.y_max - self.y_min or 1.0
        return MARGIN_TOP + (1.0 - (value - self.y_min) / span) * inner

    def y_ticks(self) -> list[float]:
        return round_ticks(self.y_min, self.y_max, target=5)


def round_ticks(low: float, high: float, target: int = 6) -> list[float]:
    """Round-number ticks (1, 2, 2.5 or 5 times a power of ten) inside ``[low, high]``."""
    step = nice_ceiling((high - low) / target) if high > low else 1.0
    first = math.ceil(low / step) * step
    count = int((high - first) / step) + 1
    return [first + i * step for i in range(max(count, 1))]


def _path(frame: _Frame, xs: Sequence[float], ys: Sequence[float]) -> str:
    return "M" + "L".join(f"{frame.x(x):.1f},{frame.y(y):.1f}" for x, y in zip(xs, ys, strict=True))


def _axes(frame: _Frame, style: ValueFormat, y_label: str, x_label: str) -> str:
    parts = ['<g class="rl-grid">']
    y_values = frame.y_ticks()
    for value in y_values:
        y = frame.y(value)
        parts.append(
            f'<line x1="{MARGIN_LEFT}" x2="{CHART_WIDTH - MARGIN_RIGHT}" '
            f'y1="{y:.1f}" y2="{y:.1f}"/>'
        )
    parts.append('</g><g class="rl-axis">')
    step = y_values[1] - y_values[0] if len(y_values) > 1 else 1.0
    whole = style not in {ValueFormat.JOULES, ValueFormat.PERCENT_SHARE} and step >= 1
    for value in y_values:
        text = f"{value:,.0f}" if whole else format_number(value, style).replace(" J", "")
        label = escape(text)
        parts.append(
            f'<text x="{MARGIN_LEFT - 10}" y="{frame.y(value) + 4:.1f}" '
            f'text-anchor="end">{label}</text>'
        )
    for value in round_ticks(frame.x_min, frame.x_max):
        parts.append(
            f'<text x="{frame.x(value):.1f}" y="{frame.height - MARGIN_BOTTOM + 18}" '
            f'text-anchor="middle">{value:,.0f}</text>'
        )
    parts.append(
        f'<text class="rl-axis-title" x="{CHART_WIDTH - MARGIN_RIGHT}" y="{frame.height - 4}" '
        f'text-anchor="end">{escape(x_label)}</text>'
        f'<text class="rl-axis-title" x="{MARGIN_LEFT}" y="{MARGIN_TOP - 14}" '
        f'text-anchor="start">{escape(y_label)}</text></g>'
    )
    return "".join(parts)


LINE_SCRIPT = """
(() => {
  const fig = document.getElementById(%(id)s);
  if (!fig || fig.dataset.ready) return;
  fig.dataset.ready = "1";
  const data = %(data)s;
  const svg = fig.querySelector("svg");
  const tip = fig.querySelector(".rl-tip");
  const cursor = fig.querySelector(".rl-cursor");
  const fmt = (v) => {
    if (data.format === "percent") return v.toFixed(1) + "%%";
    if (data.format === "share") return Math.round(v * 100) + "%%";
    if (data.format === "joules") return v.toFixed(3) + " J";
    if (data.format === "decimal") return v.toLocaleString(undefined, {maximumFractionDigits: 2});
    return Math.round(v).toLocaleString();
  };
  const nearest = (xs, x) => {
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
    return Math.abs(xs[hi] - x) < Math.abs(xs[lo] - x) ? hi : lo;
  };
  const toX = (px) =>
    data.xMin + (px - data.left) / (data.right - data.left) * (data.xMax - data.xMin);
  svg.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const px = (event.clientX - box.left) / box.width * data.width;
    if (px < data.left || px > data.right) { tip.hidden = true; cursor.style.opacity = 0; return; }
    const x = toX(px);
    const rows = data.series
      .filter((s, i) => fig.querySelector('[data-series="' + i + '"]').dataset.off !== "true")
      .map((s) => ({ s, v: s.ys[nearest(s.xs, x)] }))
      .sort((a, b) => b.v - a.v);
    const title = data.xLabel + " " + Math.round(x).toLocaleString();
    tip.replaceChildren();
    const head = document.createElement("div");
    head.className = "rl-tip-title";
    head.textContent = title;
    tip.append(head);
    for (const r of rows) {
      const row = document.createElement("div"); row.className = "rl-tip-row";
      const name = document.createElement("span");
      const dot = document.createElement("i");
      dot.className = "rl-dot";
      dot.style.setProperty("--c", r.s.color);
      name.append(dot, document.createTextNode(r.s.name));
      const val = document.createElement("b"); val.textContent = fmt(r.v);
      row.append(name, val); tip.append(row);
    }
    cursor.setAttribute("x1", px); cursor.setAttribute("x2", px); cursor.style.opacity = 1;
    tip.hidden = false;
    const fx = event.clientX - fig.getBoundingClientRect().left;
    const left = fx + 16 + tip.offsetWidth > fig.clientWidth ? fx - tip.offsetWidth - 16 : fx + 16;
    tip.style.left = Math.max(0, left) + "px";
    tip.style.top = (event.clientY - fig.getBoundingClientRect().top - 20) + "px";
  });
  svg.addEventListener("pointerleave", () => { tip.hidden = true; cursor.style.opacity = 0; });
  fig.querySelectorAll(".rl-chip").forEach((chip) => {
    const target = fig.querySelector('[data-series="' + chip.dataset.target + '"]');
    chip.addEventListener("click", () => {
      const on = chip.getAttribute("aria-pressed") !== "true";
      chip.setAttribute("aria-pressed", String(on));
      target.dataset.off = String(!on);
    });
    const focus = (state) => {
      fig.classList.toggle("rl-focus", state);
      target.classList.toggle("rl-focused", state);
    };
    chip.addEventListener("pointerenter", () => focus(true));
    chip.addEventListener("pointerleave", () => focus(false));
    chip.addEventListener("focus", () => focus(true));
    chip.addEventListener("blur", () => focus(false));
  });
})();
"""


def line_chart(
    series: Sequence[Series],
    *,
    y_label: str,
    value_format: ValueFormat,
    x_label: str = "Round",
    height: int = 340,
    caption: str = "",
    y_max: float | None = None,
    animate: bool = True,
) -> str:
    """Multi-series line chart with a legend that toggles series and a hover readout."""
    if not series:
        return ""
    sampled = [(s, *_downsample(s.xs, s.ys)) for s in series]
    x_min = min(min(xs) for _, xs, _ in sampled)
    x_max = max(max(xs) for _, xs, _ in sampled)
    top = y_max if y_max is not None else nice_ceiling(max(max(ys) for _, _, ys in sampled) * 1.04)
    lowest = min(min(ys) for _, _, ys in sampled)
    bottom = -nice_ceiling(-lowest * 1.04) if lowest < 0 else 0.0
    frame = _Frame(x_min, x_max, bottom, top, height)
    chart = element_id("rl-line", y_label, *(s.key for s in series), str(len(series[0].xs)))
    chips = "".join(
        f'<button type="button" class="rl-chip" aria-pressed="true" data-target="{i}">'
        f'<span class="rl-dot" style="--c:{s.color}"></span>{escape(s.name)}</button>'
        for i, (s, _, _) in enumerate(sampled)
    )
    ordered = sorted(enumerate(sampled), key=lambda item: item[1][0].emphasis)
    paths = "".join(
        f'<g class="rl-series{"" if s.emphasis else " rl-muted"}" data-series="{i}">'
        f'<path pathLength="1" stroke="{s.color}" d="{_path(frame, xs, ys)}"/></g>'
        for i, (s, xs, ys) in ordered
    )
    data: dict[str, PlainData] = {
        "format": value_format.value,
        "xLabel": x_label,
        "width": CHART_WIDTH,
        "left": MARGIN_LEFT,
        "right": CHART_WIDTH - MARGIN_RIGHT,
        "xMin": x_min,
        "xMax": x_max,
        "series": [
            {
                "name": s.name,
                "color": s.color,
                "xs": [round(x, 3) for x in xs],
                "ys": [round(y, 4) for y in ys],
            }
            for s, xs, ys in sampled
        ],
    }
    label = f"{y_label} by {x_label.lower()} for {', '.join(s.name for s in series)}"
    caption_html = (
        f'<figcaption class="rl-caption">{escape(caption)}</figcaption>' if caption else ""
    )
    script = LINE_SCRIPT % {"id": json.dumps(chart), "data": json_for_script(data)}
    return (
        f'<figure class="rl-chart{" rl-draw" if animate else ""}" id="{chart}">'
        f'<div class="rl-legend" role="group" aria-label="Show or hide series">{chips}</div>'
        f'<svg viewBox="0 0 {CHART_WIDTH} {height}" role="img" aria-label="{escape(label)}">'
        f"{_axes(frame, value_format, y_label, x_label)}"
        f'<line class="rl-cursor" x1="0" x2="0" y1="{MARGIN_TOP}" '
        f'y2="{height - MARGIN_BOTTOM}" style="opacity:0"/>{paths}</svg>'
        f'<div class="rl-tip" hidden></div>{caption_html}</figure>'
        f"<script>{script}</script>"
    )


# --------------------------------------------------------------------------- ranked list


@dataclass(frozen=True, slots=True)
class RankRow:
    name: str
    color: str
    value: float
    label: str


def ranking(rows: Sequence[RankRow], *, higher_is_better: bool) -> str:
    """Ranked rows with thin proportional bars; the leader gets the raised surface."""
    if not rows:
        return ""
    ordered = sorted(rows, key=lambda r: r.value, reverse=higher_is_better)
    peak = max(abs(r.value) for r in ordered) or 1.0
    items = "".join(
        f'<li class="{"rl-lead" if position == 1 else ""}">'
        f'<span class="rl-rank-pos">{position}</span>'
        f'<span class="rl-rank-name"><i class="rl-dot" style="--c:{row.color}"></i>'
        f"<span>{escape(row.name)}</span></span>"
        f'<span class="rl-rank-track" aria-hidden="true"><span class="rl-rank-bar" '
        f'style="--c:{row.color};width:{max(abs(row.value) / peak, 0.02) * 100:.1f}%">'
        "</span></span>"
        f'<span class="rl-rank-value">{escape(row.label)}</span></li>'
        for position, row in enumerate(ordered, start=1)
    )
    return f'<ol class="rl-rank">{items}</ol>'


# --------------------------------------------------------------------------- intervals


@dataclass(frozen=True, slots=True)
class IntervalRow:
    name: str
    color: str
    mean: float
    low: float
    high: float
    label: str


def interval_plot(rows: Sequence[IntervalRow], *, left_note: str, right_note: str) -> str:
    """Mean difference with its 95% interval on a shared scale centred on zero."""
    if not rows:
        return ""
    reach = max(max(abs(r.low), abs(r.high)) for r in rows) or 1.0

    def at(value: float) -> float:
        return 50.0 + value / reach * 46.0

    body = "".join(
        f'<div class="rl-interval-row"><span class="rl-rank-name"><i class="rl-dot" '
        f'style="--c:{r.color}"></i><span>{escape(r.name)}</span></span>'
        f'<div class="rl-interval-track" aria-hidden="true">'
        f'<span class="rl-interval-range" style="--c:{r.color};left:{at(r.low):.2f}%;'
        f'width:{max(at(r.high) - at(r.low), 0.4):.2f}%"></span>'
        f'<span class="rl-interval-dot" style="--c:{r.color};left:{at(r.mean):.2f}%"></span>'
        f"</div>"
        f'<span class="rl-interval-value">{escape(r.label)}</span></div>'
        for r in rows
    )
    scale = (
        f'<div class="rl-interval-row" aria-hidden="true"><span></span>'
        f'<div class="rl-interval-scale"><span>{escape(left_note)}</span><span>No difference</span>'
        f"<span>{escape(right_note)}</span></div><span></span></div>"
    )
    return f'<div class="rl-interval">{body}{scale}</div>'


# --------------------------------------------------------------------------- stacked bars


@dataclass(frozen=True, slots=True)
class Segment:
    name: str
    color: str
    value: float


@dataclass(frozen=True, slots=True)
class StackRow:
    name: str
    color: str
    segments: tuple[Segment, ...]


def stacked_bars(rows: Sequence[StackRow]) -> str:
    """One proportional bar per row, scaled to the largest total, with a shared key."""
    if not rows:
        return ""
    peak = max(sum(s.value for s in r.segments) for r in rows) or 1.0
    body = []
    for row in rows:
        total = sum(s.value for s in row.segments)
        segments = "".join(
            f'<span title="{escape(s.name)}: {s.value:,.0f}" '
            f'style="width:{s.value / peak * 100:.2f}%;background:{s.color}"></span>'
            for s in row.segments
            if s.value > 0
        )
        body.append(
            f'<div class="rl-drops-row"><span class="rl-rank-name"><i class="rl-dot" '
            f'style="--c:{row.color}"></i><span>{escape(row.name)}</span></span>'
            f'<div class="rl-drops-bar">{segments}</div>'
            f'<span class="rl-drops-total">{total:,.0f}</span></div>'
        )
    names = dict.fromkeys((s.name, s.color) for r in rows for s in r.segments if s.value > 0)
    key = "".join(
        f'<span><i class="rl-swatch" style="--c:{color}"></i>{escape(name)}</span>'
        for name, color in names
    )
    return f'<div class="rl-drops">{"".join(body)}</div><div class="rl-key">{key}</div>'
