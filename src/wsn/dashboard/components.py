"""Small HTML building blocks shared by the pages (all text escaped)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

import streamlit as st

from wsn.dashboard.charts import element_id, json_for_script
from wsn.dashboard.theme import HEALTHY, LOW

SVG_PATTERN = re.compile(r"<svg[\s>].*?</svg>", re.DOTALL)
SVG_MOUNT = (
    "<script>(() => {{ const svgs = {svgs}; svgs.forEach((markup, i) => {{"
    " const slot = document.getElementById({prefix} + i);"
    " if (slot) slot.outerHTML = markup; }}); }})();</script>"
)


def mount_svgs(markup: str) -> str:
    """Move inline SVG out of reach of Streamlit's HTML sanitizer.

    ``st.html`` sanitises with an HTML-only profile that strips ``<svg>``. Each SVG becomes an
    empty slot plus a script (scripts run after insertion) that swaps the slot for the SVG.
    The SVG strings are our own escaped output, never user input.
    """
    svgs = SVG_PATTERN.findall(markup)
    if not svgs:
        return markup
    prefix = element_id("rl-svg", markup) + "-"
    counter = iter(range(len(svgs)))
    slotted = SVG_PATTERN.sub(
        lambda _: f'<span class="rl-svg-slot" id="{prefix}{next(counter)}"></span>', markup
    )
    mount = SVG_MOUNT.format(svgs=json_for_script(list(svgs)), prefix=json.dumps(prefix))
    # A leading <script> is parsed into <head> and dropped by the sanitizer, so the mount
    # goes right before the first existing script (keeping it ahead of chart code) or last.
    first_script = slotted.find("<script>")
    if first_script <= 0:
        return slotted + mount
    return slotted[:first_script] + mount + slotted[first_script:]


def show(markup: str) -> None:
    """Render trusted, escaped markup (scripts allowed for charts and the replay)."""
    st.html(mount_svgs(markup), unsafe_allow_javascript=True)


def page_head(title: str, lead: str, *, answer_color: str | None = None) -> None:
    dot = (
        f'<span class="rl-answer-dot" style="background:{answer_color}" aria-hidden="true"></span>'
        if answer_color
        else ""
    )
    lead_class = "rl-answer" if answer_color else ""
    show(
        f'<header class="rl-head"><h1>{escape(title)}</h1>'
        f'<p class="{lead_class}">{dot}{escape(lead)}</p></header>'
    )


def battery(fraction: float) -> str:
    """The signature glyph: a battery cell filled to ``fraction`` of its charge."""
    level = min(max(fraction, 0.0), 1.0)
    color = HEALTHY if level >= 0.6 else LOW
    return (
        f'<span class="rl-battery" role="img" aria-label="{level:.0%} battery left">'
        f'<span class="rl-battery-cell"><span class="rl-battery-fill" '
        f'style="width:calc({level * 100:.0f}% - 2px);--rl-battery-color:{color}"></span></span>'
        '<span class="rl-battery-tip"></span></span>'
    )


@dataclass(frozen=True, slots=True)
class Stat:
    label: str
    figure: str
    unit: str = ""
    change: str = ""
    change_strong: str = ""
    worse: bool = False
    glyph: str = ""


def stats(items: Sequence[Stat]) -> str:
    tiles = []
    for item in items:
        unit = f'<span class="rl-stat-unit">{escape(item.unit)}</span>' if item.unit else ""
        change = ""
        if item.change or item.change_strong:
            strong = (
                f'<b class="{"rl-worse" if item.worse else ""}">{escape(item.change_strong)}</b> '
                if item.change_strong
                else ""
            )
            change = f'<p class="rl-stat-change">{strong}{escape(item.change)}</p>'
        tiles.append(
            f'<div class="rl-stat"><p class="rl-stat-label">{item.glyph}{escape(item.label)}</p>'
            f'<p class="rl-stat-figure">{escape(item.figure)}{unit}</p>{change}</div>'
        )
    return f'<div class="rl-stats">{"".join(tiles)}</div>'


def panel(title: str, body: str, *, note: str = "", aside: str = "") -> str:
    note_html = f'<p class="rl-panel-note">{escape(note)}</p>' if note else ""
    return (
        f'<section class="rl-panel"><div class="rl-panel-head"><div>'
        f'<h2 class="rl-panel-title">{escape(title)}</h2>{note_html}</div>{aside}</div>'
        f"{body}</section>"
    )


def empty_state(title: str, message: str, command: str = "") -> None:
    code = f"<code>{escape(command)}</code>" if command else ""
    show(
        f'<div class="rl-empty"><strong>{escape(title)}</strong>'
        f"<p>{escape(message)}</p>{code}</div>"
    )


BRAND = """
<div class="rl-brand">
  <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
    <rect x="0.5" y="0.5" width="27" height="27" rx="8" fill="#1C2226"
      stroke="rgba(220,232,240,.14)"/>
    <circle cx="9" cy="18" r="2.2" fill="#4FC79A"/>
    <circle cx="19" cy="9" r="2.2" fill="#E3B341"/>
    <circle cx="20" cy="19" r="2.2" fill="none" stroke="#5A646B" stroke-width="1.4"/>
    <path d="M9 18 L14 14 L19 9 M14 14 L20 19" stroke="rgba(220,232,240,.35)" stroke-width="1"/>
    <rect x="12" y="12" width="4" height="4" rx="0.8" fill="#E7ECEF" transform="rotate(45 14 14)"/>
  </svg>
  <div><div class="rl-brand-name">Routing Lab</div>
  <div class="rl-brand-sub">Sensor network lifetime</div></div>
</div>
"""
