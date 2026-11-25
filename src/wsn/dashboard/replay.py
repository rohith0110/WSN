"""Synchronised replay of recorded runs: node batteries, live links and sampled packet paths.

Everything drawn comes from saved artifacts. Node energy between two recorded snapshots is
interpolated linearly for display only; packets shown are the sampled, really traced packets.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from wsn.artifacts import StoredRun
from wsn.dashboard.charts import element_id, json_for_script
from wsn.dashboard.theme import DEAD, HEALTHY, LOW
from wsn.models import PlainData

FIELD_PADDING = 4.0
NODE_RADIUS_FRACTION = 0.016
TIMELINE_WIDTH = 960
TIMELINE_HEIGHT = 88


@dataclass(frozen=True, slots=True)
class ReplayRun:
    label: str
    color: str
    run: StoredRun


def _series(run: StoredRun, column: str) -> list[float]:
    simulated = run.rounds[run.rounds["simulated"]]
    return [float(v) for v in simulated[column]]


def _run_data(entry: ReplayRun) -> dict[str, PlainData]:
    run = entry.run
    topology = run.topology
    sink = topology.sink_id
    snapshot_rounds = run.snapshot_rounds()
    initial = run.initial_energy_j or 1.0
    energy: list[PlainData] = [
        [round(e / initial, 3) for e in run.energy_at(r)] for r in snapshot_rounds
    ]
    links: list[PlainData] = [[a, b] for (a, b) in topology.edge_lengths_m()]
    traces: list[PlainData] = [
        [trace.round_number, list(trace.path), trace.outcome == "delivered"]
        for trace in run.traces
        if len(trace.path) > 1
    ]
    alive: list[PlainData] = [
        run.summary.node_count,
        *(int(v) for v in _series(run, "alive_nodes")),
    ]
    delivered: list[PlainData] = [
        0,
        *(int(v) for v in _series(run, "delivered_packets_cumulative")),
    ]
    remaining: list[PlainData] = [
        1.0,
        *(
            round(v / (initial * run.summary.node_count), 4)
            for v in _series(run, "total_remaining_energy_j")
        ),
    ]
    nodes: list[PlainData] = [[round(x, 2), round(y, 2)] for x, y in topology.positions[:sink]]
    return {
        "label": entry.label,
        "color": entry.color,
        "nodes": nodes,
        "sink": [round(topology.positions[sink][0], 2), round(topology.positions[sink][1], 2)],
        "range": topology.communication_range_m,
        "links": links,
        "snapRounds": list(snapshot_rounds),
        "energy": energy,
        "traces": traces,
        "alive": alive,
        "delivered": delivered,
        "remaining": remaining,
        "lastRound": run.summary.rounds_executed,
    }


def _field_svg(entry: ReplayRun, index: int, width: float, height: float) -> str:
    topology = entry.run.topology
    sink_x, sink_y = topology.positions[topology.sink_id]
    radius = max(width, height) * NODE_RADIUS_FRACTION
    nodes = "".join(
        f'<circle class="rl-node" cx="{x:.2f}" cy="{height - y:.2f}" r="{radius:.2f}" '
        f'fill="{HEALTHY}" stroke="#0B0E10"/>'
        for x, y in topology.positions[: topology.sink_id]
    )
    size = radius * 2.2
    pad = FIELD_PADDING
    label = (
        f"Sensor field for {entry.label}: {topology.node_count} sensors around a sink. "
        "Colors show remaining battery."
    )
    return (
        f'<svg viewBox="{-pad} {-pad} {width + 2 * pad} {height + 2 * pad}" role="img" '
        f'aria-label="{escape(label)}" data-field="{index}">'
        f'<circle class="rl-range" cx="{sink_x:.2f}" cy="{height - sink_y:.2f}" '
        f'r="{topology.communication_range_m:.2f}"/>'
        f'<path class="rl-link" d=""/><g class="rl-nodes">{nodes}</g>'
        f'<rect class="rl-sink" x="{sink_x - size / 2:.2f}" y="{height - sink_y - size / 2:.2f}" '
        f'width="{size:.2f}" height="{size:.2f}" rx="{size * 0.2:.2f}" '
        f'transform="rotate(45 {sink_x:.2f} {height - sink_y:.2f})"/>'
        f'<g class="rl-packets"></g></svg>'
    )


def _timeline_svg(entries: Sequence[ReplayRun], max_round: int, node_count: int) -> str:
    def path(entry: ReplayRun) -> str:
        alive = [node_count, *_series(entry.run, "alive_nodes")]
        points = [
            f"{i / max(max_round, 1) * TIMELINE_WIDTH:.1f},"
            f"{(1 - a / max(node_count, 1)) * (TIMELINE_HEIGHT - 8) + 4:.1f}"
            for i, a in enumerate(alive[: max_round + 1])
        ]
        return "M" + "L".join(points)

    lines = "".join(
        f'<path d="{path(e)}" fill="none" stroke="{e.color}" stroke-width="1.6"/>' for e in entries
    )
    return (
        f'<svg class="rl-timeline" viewBox="0 0 {TIMELINE_WIDTH} {TIMELINE_HEIGHT}" '
        f'preserveAspectRatio="none" aria-hidden="true" '
        f'style="width:100%;height:{TIMELINE_HEIGHT}px;'
        f'display:block;margin-top:16px">'
        f'<line x1="0" x2="{TIMELINE_WIDTH}" y1="{TIMELINE_HEIGHT - 4}" y2="{TIMELINE_HEIGHT - 4}" '
        f'stroke="rgba(220,232,240,.08)"/>{lines}'
        f'<line class="rl-playhead" x1="0" x2="0" y1="0" y2="{TIMELINE_HEIGHT}" '
        f'stroke="#E7ECEF" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>'
    )


REPLAY_SCRIPT = """
(() => {
  const root = document.getElementById(%(id)s);
  if (!root || root.dataset.ready) return;
  root.dataset.ready = "1";
  const D = %(data)s;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const stops = [[0, [122, 82, 48]], [0.3, %(low)s], [0.6, %(healthy)s], [1, %(healthy)s]];
  const mix = (e) => {
    for (let i = 1; i < stops.length; i++) {
      if (e <= stops[i][0]) {
        const [a, ca] = stops[i - 1], [b, cb] = stops[i];
        const t = (e - a) / (b - a || 1);
        return "rgb(" + ca.map((c, k) => Math.round(c + (cb[k] - c) * t)).join(",") + ")";
      }
    }
    return "rgb(" + stops[stops.length - 1][1].join(",") + ")";
  };
  const fields = D.runs.map((run, i) => {
    const svg = root.querySelector('[data-field="' + i + '"]');
    return {
      run, svg,
      nodes: [...svg.querySelectorAll(".rl-node")],
      links: svg.querySelector(".rl-link"),
      packets: svg.querySelector(".rl-packets"),
      live: root.querySelector('[data-live="' + i + '"]'),
      aliveKey: "",
    };
  });
  const H = D.height;
  const pos = (run, n) => n === run.nodes.length ? run.sink : run.nodes[n];
  const energyAt = (run, r) => {
    const rs = run.snapRounds;
    if (r >= rs[rs.length - 1]) return run.energy[rs.length - 1];
    let k = 0;
    while (k < rs.length - 2 && rs[k + 1] <= r) k++;
    const t = (r - rs[k]) / ((rs[k + 1] - rs[k]) || 1);
    return run.energy[k].map((e, n) => {
      const next = run.energy[k + 1][n];
      return next <= 0 && r >= rs[k + 1] ? 0 : e + (next - e) * Math.min(Math.max(t, 0), 1);
    });
  };
  const deadAt = (run, r) => {
    const rs = run.snapRounds;
    let k = 0;
    while (k < rs.length - 1 && rs[k + 1] <= r) k++;
    return run.energy[k].map((e) => e <= 0);
  };
  const fmtPct = (v) => Math.round(v * 100) + "%%";
  const pointOnPath = (run, path, p) => {
    const pts = path.map((n) => pos(run, n));
    const seg = Math.min(Math.floor(p * (pts.length - 1)), pts.length - 2);
    const t = p * (pts.length - 1) - seg;
    const [x1, y1] = pts[seg], [x2, y2] = pts[seg + 1];
    return [x1 + (x2 - x1) * t, y1 + (y2 - y1) * t];
  };
  let round = 0, playing = false, speed = 24, last = 0;
  const range = root.querySelector("input[type=range]");
  const label = root.querySelector(".rl-scrub label");
  const playBtn = root.querySelector(".rl-play");
  const playhead = root.querySelector(".rl-playhead");
  const draw = () => {
    const whole = Math.min(Math.floor(round), D.maxRound);
    for (const f of fields) {
      const run = f.run;
      const e = energyAt(run, round);
      const dead = deadAt(run, round);
      f.nodes.forEach((c, n) => {
        c.setAttribute("fill", mix(Math.max(e[n], 0)));
        c.dataset.dead = String(dead[n]);
      });
      const key = dead.map((d) => d ? 0 : 1).join("");
      if (key !== f.aliveKey) {
        f.aliveKey = key;
        const sink = run.nodes.length;
        f.links.setAttribute("d", run.links
          .filter(([a, b]) => (a === sink || !dead[a]) && (b === sink || !dead[b]))
          .map(([a, b]) => { const [x1, y1] = pos(run, a), [x2, y2] = pos(run, b);
            return "M" + x1 + "," + (H - y1) + "L" + x2 + "," + (H - y2); }).join(""));
      }
      f.packets.replaceChildren();
      if (!reduce && round > 0 && whole < run.lastRound) {
        const current = whole + 1, p = round - whole;
        for (const [r, path, ok] of run.traces) {
          if (r !== current) continue;
          const [x, y] = pointOnPath(run, path, p);
          const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          dot.setAttribute("class", "rl-packet"); dot.setAttribute("r", "1.1");
          dot.setAttribute("cx", x); dot.setAttribute("cy", H - y);
          dot.dataset.failed = String(!ok && p > 0.9);
          f.packets.append(dot);
        }
      }
      const i = Math.min(whole, run.alive.length - 1);
      f.live.querySelector("[data-k=alive]").textContent = run.alive[i] + " of " + run.nodes.length;
      f.live.querySelector("[data-k=energy]").textContent = fmtPct(run.remaining[i]);
      f.live.querySelector("[data-k=delivered]").textContent = run.delivered[i].toLocaleString();
    }
    range.value = String(whole);
    label.textContent = "Round " + whole.toLocaleString() + " of " + D.maxRound.toLocaleString();
    playhead.setAttribute("x1", whole / D.maxRound * %(tw)s);
    playhead.setAttribute("x2", whole / D.maxRound * %(tw)s);
  };
  const setPlaying = (state) => {
    playing = state;
    playBtn.setAttribute("aria-label", state ? "Pause replay" : "Play replay");
    const idle = round >= D.maxRound ? "Replay" : "Play";
    playBtn.querySelector(".rl-play-text").textContent = state ? "Pause" : idle;
    playBtn.querySelector(".rl-icon-play").style.display = state ? "none" : "";
    playBtn.querySelector(".rl-icon-pause").style.display = state ? "" : "none";
    if (state) { last = performance.now(); requestAnimationFrame(tick); }
  };
  const tick = (now) => {
    if (!playing) return;
    round = Math.min(round + (now - last) / 1000 * speed, D.maxRound);
    last = now;
    draw();
    if (round >= D.maxRound) { setPlaying(false); return; }
    requestAnimationFrame(tick);
  };
  playBtn.addEventListener("click", () => {
    if (!playing && round >= D.maxRound) round = 0;
    setPlaying(!playing);
  });
  range.addEventListener("input", () => { round = Number(range.value); draw(); });
  root.querySelectorAll(".rl-speed button").forEach((b) => b.addEventListener("click", () => {
    speed = Number(b.dataset.speed);
    root.querySelectorAll(".rl-speed button")
      .forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
  }));
  draw();
})();
"""


def _rgb(color: str) -> str:
    value = color.lstrip("#")
    return json.dumps([int(value[i : i + 2], 16) for i in (0, 2, 4)])


def replay(entries: Sequence[ReplayRun], *, width: float, height: float) -> str:
    """Side-by-side (or single) field replay sharing one timeline and one set of controls."""
    if not entries:
        return ""
    max_round = max(e.run.summary.rounds_executed for e in entries)
    node_count = entries[0].run.summary.node_count
    replay_id = element_id("rl-replay", *(str(e.run.directory) for e in entries), str(max_round))
    fields = "".join(
        f'<div class="rl-field"><div class="rl-field-head">'
        f'<span class="rl-field-name"><i class="rl-dot" style="--c:{e.color}"></i>'
        f"{escape(e.label)}</span>"
        f'<span class="rl-field-live" data-live="{i}" aria-live="off">'
        f'<span>Alive <b data-k="alive"></b></span>'
        f'<span>Battery left <b data-k="energy"></b></span>'
        f'<span>Delivered <b data-k="delivered"></b></span></span></div>'
        f"{_field_svg(e, i, width, height)}</div>"
        for i, e in enumerate(entries)
    )
    data: dict[str, PlainData] = {
        "maxRound": max_round,
        "height": height,
        "runs": [_run_data(e) for e in entries],
    }
    script = REPLAY_SCRIPT % {
        "id": json.dumps(replay_id),
        "data": json_for_script(data),
        "low": _rgb(LOW),
        "healthy": _rgb(HEALTHY),
        "tw": TIMELINE_WIDTH,
    }
    play_icon = (
        '<svg class="rl-icon-play" viewBox="0 0 16 16" aria-hidden="true">'
        '<path d="M5 3.2v9.6L12.6 8z" fill="currentColor"/></svg>'
    )
    pause_icon = (
        '<svg class="rl-icon-pause" viewBox="0 0 16 16" aria-hidden="true" style="display:none">'
        '<rect x="4" y="3" width="3" height="10" rx="1" fill="currentColor"/>'
        '<rect x="9" y="3" width="3" height="10" rx="1" fill="currentColor"/></svg>'
    )
    speeds = "".join(
        f'<button type="button" data-speed="{rate}" aria-pressed="{str(rate == 24).lower()}">'
        f"{label}</button>"
        for rate, label in ((12, "0.5\u00d7"), (24, "1\u00d7"), (60, "2.5\u00d7"), (120, "5\u00d7"))
    )
    key = (
        f'<div class="rl-map-key">'
        f'<span><i style="background:{HEALTHY}"></i>Battery above 60%</span>'
        f'<span><i style="background:{LOW}"></i>Battery running low</span>'
        f'<span><i style="box-shadow:inset 0 0 0 1.5px {DEAD}"></i>Battery empty</span>'
        '<span><i style="background:#E7ECEF;border-radius:2px;transform:rotate(45deg)">'
        "</i>Sink</span>"
        '<span><i style="background:#E7ECEF;width:6px;height:6px"></i>Packet in flight</span></div>'
    )
    controls = (
        f'<div class="rl-controls"><button type="button" class="rl-play" aria-label="Play replay">'
        f'{play_icon}{pause_icon}<span class="rl-play-text">Play</span></button>'
        f'<div class="rl-scrub"><label for="{replay_id}-r"></label>'
        f'<input id="{replay_id}-r" type="range" min="0" max="{max_round}" step="1" '
        'value="0"></div>'
        f'<div class="rl-speed" role="group" aria-label="Replay speed">{speeds}</div></div>'
    )
    return (
        f'<section class="rl-replay" id="{replay_id}" aria-label="Network replay">'
        f'{controls}<div class="rl-fields">{fields}</div>'
        f"{_timeline_svg(entries, max_round, node_count)}"
        f"{key}</section><script>{script}</script>"
    )
