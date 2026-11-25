"""Design tokens and the stylesheet shared by every dashboard page.

Direction: a calm field instrument at night. Graphite neutrals tinted slightly blue-green,
one radio-blue accent, and battery-charge colors that only ever describe energy.
"""

from __future__ import annotations

HEALTHY = "#4FC79A"
LOW = "#E3B341"
DEAD = "#5A646B"
ACCENT = "#6CB4F0"
TEXT = "#E7ECEF"
TEXT_2 = "#A9B3BA"
TEXT_3 = "#808B93"
SURFACE = "#151A1D"
RAISED = "#1C2226"
LINE = "rgba(220, 232, 240, 0.08)"
FAILURE = "#E7718A"

STYLESHEET = """
<style>
:root {
  --rl-bg: #0E1113;
  --rl-surface: #151A1D;
  --rl-raised: #1C2226;
  --rl-line: rgba(220, 232, 240, 0.08);
  --rl-line-strong: rgba(220, 232, 240, 0.14);
  --rl-text: #E7ECEF;
  --rl-text-2: #A9B3BA;
  --rl-text-3: #808B93;
  --rl-accent: #6CB4F0;
  --rl-healthy: #4FC79A;
  --rl-low: #E3B341;
  --rl-dead: #5A646B;
  --rl-fail: #E7718A;
  --rl-ease: cubic-bezier(0.23, 1, 0.32, 1);
  --rl-panel-radius: 16px;
  --rl-control-radius: 10px;
}

/* ------------------------------------------------------------------ shell */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] {
  max-width: 1320px;
  padding-inline: 32px;
  padding-bottom: 96px;
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-inline: 16px; }
}
[data-testid="stSidebarHeader"] { padding-bottom: 8px; }
[data-testid="stSidebarNav"] a { border-radius: var(--rl-control-radius); min-height: 36px; }
[data-testid="stSidebarNav"] a:focus-visible {
  outline: 2px solid var(--rl-accent);
  outline-offset: 2px;
}
h1, h2, h3 { text-wrap: balance; letter-spacing: -0.01em; }
p, li { text-wrap: pretty; }
[data-testid="stMetricValue"], .rl-num { font-variant-numeric: tabular-nums; }

/* ------------------------------------------------------------------ widgets */
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] * {
  white-space: nowrap; width: max-content; font-variant-numeric: tabular-nums;
}
[data-testid^="stBaseButton-primary"] {
  background: var(--rl-text); border-color: var(--rl-text); color: #0E1113;
}
[data-testid^="stBaseButton-primary"]:hover {
  background: #FFFFFF; border-color: #FFFFFF; color: #0E1113;
}
[data-testid^="stBaseButton-primary"]:active { transform: scale(0.98); }
[data-testid^="stBaseButton-primary"] p { color: #0E1113; font-weight: 500; }

/* ------------------------------------------------------------------ brand */
.rl-brand { display: flex; align-items: center; gap: 10px; padding: 4px 4px 0; }
.rl-brand svg { flex: none; }
.rl-brand-name { font-weight: 600; font-size: 15px; color: var(--rl-text); }
.rl-brand-sub { font-size: 12px; color: var(--rl-text-3); }

/* ------------------------------------------------------------------ page head */
.rl-head { margin: 8px 0 28px; max-width: 760px; }
.rl-head h1 {
  font-size: clamp(1.75rem, 1.1rem + 1.8vw, 2.5rem);
  line-height: 1.12;
  font-weight: 600;
  margin: 0 0 12px;
  color: var(--rl-text);
}
.rl-head p { font-size: 1.0625rem; line-height: 1.55; color: var(--rl-text-2); margin: 0; }
.rl-head .rl-answer { color: var(--rl-text); }
.rl-answer-dot {
  display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin-inline-end: 8px; vertical-align: 1px;
}

/* ------------------------------------------------------------------ panels */
.rl-panel {
  background: var(--rl-surface);
  border: 1px solid var(--rl-line);
  border-radius: var(--rl-panel-radius);
  padding: 20px 24px 24px;
}
.rl-panel + .rl-panel { margin-top: 16px; }
.rl-panel-head { display: flex; justify-content: space-between; align-items: baseline;
  gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
.rl-panel-title { font-size: 1rem; font-weight: 500; color: var(--rl-text); margin: 0; }
.rl-panel-note { font-size: 13px; color: var(--rl-text-3); margin: 4px 0 0; max-width: 64ch; }

/* ------------------------------------------------------------------ stats */
.rl-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.rl-stat {
  background: var(--rl-surface);
  border: 1px solid var(--rl-line);
  border-radius: 14px;
  padding: 16px 20px 18px;
  min-width: 0;
}
.rl-stat-label { font-size: 13px; color: var(--rl-text-2); margin: 0 0 10px;
  display: flex; align-items: center; gap: 8px; }
.rl-stat-figure { font-size: 1.875rem; line-height: 1.1; font-weight: 500;
  color: var(--rl-text); font-variant-numeric: tabular-nums; margin: 0; }
.rl-stat-unit {
  font-size: 0.9375rem;
  color: var(--rl-text-3);
  font-weight: 400;
  margin-inline-start: 4px;
}
.rl-stat-change { font-size: 13px; color: var(--rl-text-3); margin: 8px 0 0; }
.rl-stat-change b { font-weight: 500; color: var(--rl-healthy); }
.rl-stat-change b.rl-worse { color: var(--rl-fail); }

/* ------------------------------------------------------------------ battery signature */
.rl-battery { display: inline-flex; align-items: center; gap: 2px; vertical-align: -2px; }
.rl-battery-cell {
  position: relative; width: 22px; height: 11px; border-radius: 3px;
  border: 1.5px solid currentColor; color: var(--rl-text-3); overflow: hidden;
}
.rl-battery-fill { position: absolute; inset: 1px auto 1px 1px; border-radius: 1px;
  background: var(--rl-battery-color, var(--rl-healthy)); }
.rl-battery-tip { width: 2px; height: 5px; border-radius: 0 1px 1px 0; background: currentColor;
  color: var(--rl-text-3); }

/* ------------------------------------------------------------------ ranking */
.rl-rank { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; }
.rl-rank li {
  display: grid; grid-template-columns: 20px minmax(120px, 190px) minmax(0, 1fr) 92px;
  align-items: center; gap: 12px; min-height: 40px; padding: 0 8px; border-radius: 10px;
}
.rl-rank li.rl-lead { background: var(--rl-raised); }
.rl-rank-pos { font-size: 13px; color: var(--rl-text-3); font-variant-numeric: tabular-nums; }
.rl-rank-name { display: flex; align-items: center; gap: 8px; min-width: 0;
  font-size: 14px; color: var(--rl-text); }
.rl-rank-name span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rl-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; background: var(--c); }
.rl-rank-track { height: 6px; border-radius: 3px; background: var(--rl-line); overflow: hidden; }
.rl-rank-bar { display: block; height: 100%; border-radius: 3px; background: var(--c);
  transform-origin: left center; }
.rl-rank-value { font-size: 14px; text-align: end; color: var(--rl-text);
  font-variant-numeric: tabular-nums; }
@media (max-width: 640px) {
  .rl-rank li { grid-template-columns: 20px minmax(0, 1fr) 80px; }
  .rl-rank-track { grid-column: 2 / 4; grid-row: 2; margin-bottom: 8px; }
}

/* ------------------------------------------------------------------ condition winners */
.rl-cond { list-style: none; margin: 0; padding: 0; }
.rl-cond li { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) 110px;
  gap: 12px; align-items: center; min-height: 40px; font-size: 14px; color: var(--rl-text-2);
  border-top: 1px solid var(--rl-line); }
.rl-cond li:first-child { border-top: 0; }
.rl-cond-gain { text-align: end; color: var(--rl-text); font-variant-numeric: tabular-nums; }

/* ------------------------------------------------------------------ next steps */
.rl-next { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.rl-next a {
  display: block; text-decoration: none; color: inherit;
  background: var(--rl-surface); border: 1px solid var(--rl-line); border-radius: 14px;
  padding: 16px 20px;
  transition: background-color 150ms var(--rl-ease), border-color 150ms var(--rl-ease);
}
.rl-next a:hover { background: var(--rl-raised); border-color: var(--rl-line-strong); }
.rl-next a:focus-visible { outline: 2px solid var(--rl-accent); outline-offset: 2px; }
.rl-next strong {
  display: block;
  font-weight: 500;
  font-size: 15px;
  color: var(--rl-text);
  margin-bottom: 4px;
}
.rl-next span { font-size: 13px; color: var(--rl-text-2); line-height: 1.5; }

/* ------------------------------------------------------------------ charts */
.rl-chart { margin: 0; position: relative; }
.rl-chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.rl-chart .rl-grid line { stroke: var(--rl-line); stroke-width: 1; }
.rl-chart .rl-axis text { fill: var(--rl-text-3); font-size: 11px;
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
.rl-chart .rl-axis-title { fill: var(--rl-text-3); font-size: 12px; }
.rl-series path { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round;
  transition: opacity 150ms var(--rl-ease); }
.rl-series.rl-muted path { stroke-width: 1.5; }
.rl-series[data-off="true"] path { opacity: 0; }
.rl-chart.rl-focus .rl-series:not(.rl-focused) path { opacity: 0.18; }
.rl-cursor { stroke: var(--rl-line-strong); stroke-width: 1; }
.rl-legend { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 12px; }
.rl-chip {
  display: inline-flex; align-items: center; gap: 8px; min-height: 30px; padding: 0 12px;
  border-radius: 999px; border: 1px solid var(--rl-line-strong); background: transparent;
  color: var(--rl-text); font: inherit; font-size: 13px; cursor: pointer;
  transition: background-color 150ms var(--rl-ease), color 150ms var(--rl-ease);
}
.rl-chip:hover { background: var(--rl-raised); }
.rl-chip:active { transform: scale(0.97); }
.rl-chip:focus-visible { outline: 2px solid var(--rl-accent); outline-offset: 2px; }
.rl-chip[aria-pressed="false"] { color: var(--rl-text-3); }
.rl-chip[aria-pressed="false"] .rl-dot {
  background: transparent;
  box-shadow: inset 0 0 0 1.5px var(--c);
}
.rl-tip {
  position: absolute; pointer-events: none; z-index: 5; min-width: 180px;
  background: var(--rl-raised); border: 1px solid var(--rl-line-strong); border-radius: 10px;
  padding: 10px 12px; font-size: 12px; color: var(--rl-text-2);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.rl-tip-title { color: var(--rl-text); font-weight: 500; margin-bottom: 6px; }
.rl-tip-row {
  display: flex;
  align-items: center;
  gap: 8px;
  justify-content: space-between;
  line-height: 1.7;
}
.rl-tip-row span:first-child { display: flex; align-items: center; gap: 8px; }
.rl-tip-row b { font-weight: 500; color: var(--rl-text); font-variant-numeric: tabular-nums; }
.rl-caption { font-size: 12px; color: var(--rl-text-3); margin: 10px 0 0; }

/* ------------------------------------------------------------------ interval plot */
.rl-interval { display: grid; gap: 2px; }
.rl-interval-row { display: grid; grid-template-columns: minmax(130px, 200px) minmax(0, 1fr) 190px;
  align-items: center; gap: 12px; min-height: 36px; }
.rl-interval-track { position: relative; height: 24px; }
.rl-interval-track::before {
  content: ""; position: absolute; left: 50%; top: 0; bottom: 0; width: 1px;
  background: var(--rl-line-strong);
}
.rl-interval-range {
  position: absolute; top: 50%; height: 3px; margin-top: -1.5px; border-radius: 2px;
  background: var(--c); opacity: 0.55;
}
.rl-interval-dot {
  position: absolute; top: 50%; width: 11px; height: 11px; margin: -5.5px 0 0 -5.5px;
  border-radius: 50%; background: var(--c); box-shadow: 0 0 0 2px var(--rl-surface);
}
.rl-interval-value { text-align: end; font-size: 13px; color: var(--rl-text-2);
  font-variant-numeric: tabular-nums; }
.rl-interval-value b { color: var(--rl-text); font-weight: 500; }
.rl-interval-scale { display: flex; justify-content: space-between; font-size: 11px;
  color: var(--rl-text-3); padding-inline: 0; }

/* ------------------------------------------------------------------ stacked drops */
.rl-drops { display: grid; gap: 10px; }
.rl-drops-row { display: grid; grid-template-columns: minmax(130px, 190px) minmax(0, 1fr) 70px;
  gap: 12px; align-items: center; font-size: 13px; color: var(--rl-text); }
.rl-drops-bar {
  display: flex;
  height: 12px;
  border-radius: 4px;
  overflow: hidden;
  background: var(--rl-line);
}
.rl-drops-bar span { height: 100%; }
.rl-drops-total { text-align: end; color: var(--rl-text-2); font-variant-numeric: tabular-nums; }
.rl-key { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--rl-text-2);
  margin-top: 14px; }
.rl-key span { display: inline-flex; align-items: center; gap: 6px; }
.rl-swatch { width: 10px; height: 10px; border-radius: 3px; background: var(--c); }

/* ------------------------------------------------------------------ replay */
.rl-replay { background: var(--rl-surface); border: 1px solid var(--rl-line);
  border-radius: var(--rl-panel-radius); padding: 20px 24px 24px; }
.rl-fields {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}
.rl-field-head { display: flex; justify-content: space-between; align-items: center; gap: 12px;
  margin-bottom: 10px; }
.rl-field-name { display: flex; align-items: center; gap: 8px; font-weight: 500; font-size: 15px; }
.rl-field-live { display: flex; gap: 16px; font-size: 13px; color: var(--rl-text-2);
  font-variant-numeric: tabular-nums; }
.rl-field-live b { color: var(--rl-text); font-weight: 500; }
.rl-field svg { display: block; width: 100%; height: auto; border-radius: 12px;
  background: #0B0E10; border: 1px solid var(--rl-line); }
.rl-field .rl-link { stroke: rgba(220, 232, 240, 0.09); stroke-width: 0.22; }
.rl-field .rl-node { stroke-width: 1.4; transition: r 200ms var(--rl-ease); }
.rl-field .rl-node[data-dead="true"] { fill: transparent !important; stroke: var(--rl-dead); }
.rl-field .rl-sink { fill: var(--rl-text); }
.rl-field .rl-range { fill: none; stroke: rgba(108, 180, 240, 0.28); stroke-width: 0.25;
  stroke-dasharray: 1.2 1.6; }
.rl-packet { fill: var(--rl-text); }
.rl-packet[data-failed="true"] { fill: var(--rl-fail); }
.rl-controls {
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 20px;
}
.rl-play {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  height: 40px; padding: 0 18px 0 14px; border-radius: var(--rl-control-radius); border: 0;
  background: var(--rl-text); color: #0E1113; font: inherit; font-weight: 500; font-size: 14px;
  cursor: pointer;
  transition: transform 120ms var(--rl-ease), background-color 150ms var(--rl-ease);
}
.rl-play:hover { background: #FFFFFF; }
.rl-play:active { transform: scale(0.97); }
.rl-play:focus-visible { outline: 2px solid var(--rl-accent); outline-offset: 3px; }
.rl-play svg { width: 16px; height: 16px; }
.rl-scrub { flex: 1 1 240px; display: grid; gap: 4px; }
.rl-scrub input { width: 100%; accent-color: var(--rl-accent); }
.rl-scrub label { font-size: 13px; color: var(--rl-text-2); font-variant-numeric: tabular-nums; }
.rl-speed {
  display: inline-flex;
  border: 1px solid var(--rl-line-strong);
  border-radius: var(--rl-control-radius);
  padding: 2px; gap: 2px; }
.rl-speed button {
  height: 32px;
  min-width: 44px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--rl-text-2); font: inherit; font-size: 13px; cursor: pointer; }
.rl-speed button[aria-pressed="true"] { background: var(--rl-raised); color: var(--rl-text); }
.rl-speed button:focus-visible { outline: 2px solid var(--rl-accent); outline-offset: 1px; }
.rl-map-key {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--rl-text-2);
  margin-top: 14px;
}
.rl-map-key span { display: inline-flex; align-items: center; gap: 6px; }
.rl-map-key i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

.rl-field-head { flex-wrap: wrap; }
.rl-field-name { white-space: nowrap; }
@media (max-width: 640px) {
  .rl-replay, .rl-panel { padding: 16px; }
  .rl-field-live { gap: 12px; font-size: 12px; }
}

/* ------------------------------------------------------------------ empty state */
.rl-empty { border: 1px dashed var(--rl-line-strong); border-radius: var(--rl-panel-radius);
  padding: 32px; max-width: 640px; }
.rl-empty strong { display: block; font-weight: 500; font-size: 16px; margin-bottom: 6px; }
.rl-empty p { color: var(--rl-text-2); margin: 0 0 12px; font-size: 14px; }
.rl-empty code { font-size: 13px; }

/* ------------------------------------------------------------------ the one load moment */
@media (prefers-reduced-motion: no-preference) {
  .rl-draw .rl-series path {
    stroke-dasharray: 1; stroke-dashoffset: 1;
    animation: rl-draw 1100ms var(--rl-ease) forwards;
  }
  .rl-draw .rl-series:nth-child(2) path { animation-delay: 60ms; }
  .rl-draw .rl-series:nth-child(3) path { animation-delay: 120ms; }
  .rl-draw .rl-series:nth-child(4) path { animation-delay: 180ms; }
  .rl-draw .rl-series:nth-child(5) path { animation-delay: 240ms; }
  .rl-draw .rl-series:nth-child(6) path { animation-delay: 300ms; }
  .rl-rank-bar { animation: rl-grow 700ms var(--rl-ease) both; }
  .rl-rank li:nth-child(2) .rl-rank-bar { animation-delay: 30ms; }
  .rl-rank li:nth-child(3) .rl-rank-bar { animation-delay: 60ms; }
  .rl-rank li:nth-child(4) .rl-rank-bar { animation-delay: 90ms; }
  .rl-rank li:nth-child(5) .rl-rank-bar { animation-delay: 120ms; }
  .rl-rank li:nth-child(6) .rl-rank-bar { animation-delay: 150ms; }
}
@keyframes rl-draw { to { stroke-dashoffset: 0; } }
@keyframes rl-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
</style>
"""
