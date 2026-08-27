/* Revenue Recovery dashboard — vanilla JS, no framework, no build step.
 * Fetches from the FastAPI backend (same origin) and renders 5 panels with
 * hand-rolled SVG charts. Colors are resolved as literal hex here (not
 * css var() inside dynamically-built SVG attributes) since var() support
 * in plain SVG presentation attributes is inconsistent across browsers.
 */

const ACCENT = "#6366F1";
const ACCENT_TEXT = "#8285F4";
const WARNING = "#F59E0B";
const TEXT_SECONDARY = "#94A3B8";
const TEXT_MUTED = "#5B6472";
const SURFACE_DIM = [30, 35, 45]; // base of the sequential heatmap ramp
const ACCENT_RGB = [99, 102, 241];

const SVG_NS = "http://www.w3.org/2000/svg";

// Fixed per-tactic identity for the reward-over-time chart, assigned once
// in registry order (mirrors tactics/registry.py) so a tactic's look never
// drifts across reloads. 4 shades x 3 dash patterns (coprime periods) give
// 12 unique combinations before any repeat -- comfortably covers 10 tactics.
const TACTIC_ORDER = [
  "immediate_retry", "payday_retry", "alternate_gateway_retry", "soft_nudge_email",
  "discount_email", "sms_reminder", "whatsapp_nudge", "payment_plan_offer",
  "account_manager_outreach", "no_action",
];
const SHADES = ["#FFFFFF", TEXT_SECONDARY, ACCENT, TEXT_MUTED];
const DASHES = ["none", "6 3", "1 3"];
const TACTIC_STYLE = {};
TACTIC_ORDER.forEach((name, i) => {
  TACTIC_STYLE[name] = { color: SHADES[i % SHADES.length], dash: DASHES[i % DASHES.length] };
});
const CSS_DASH_STYLE = { none: "solid", "6 3": "dashed", "1 3": "dotted" };

// ---- formatting -----------------------------------------------------------

function fmtCurrency(v) {
  if (v === null || v === undefined) return "—";
  return `₹${v.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}
function fmtSignedCurrency(v) {
  if (v === null || v === undefined) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${fmtCurrency(v)}`;
}
function fmtPercent(v) {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(1)}%`;
}
function fmtSignedPercent(v) {
  if (v === null || v === undefined) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(1)}pp`;
}
function fmtCompact(v) {
  return Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0);
}

// ---- svg helpers ------------------------------------------------------------

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  return el;
}

function mixAccent(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const rgb = SURFACE_DIM.map((base, i) => Math.round(base + (ACCENT_RGB[i] - base) * clamped));
  return `rgb(${rgb.join(",")})`;
}

// ---- tooltip ----------------------------------------------------------------

const tooltipEl = document.getElementById("tooltip");

function attachTooltip(el, contentFn) {
  el.addEventListener("mouseenter", (e) => {
    tooltipEl.innerHTML = contentFn();
    tooltipEl.hidden = false;
    positionTooltip(e);
  });
  el.addEventListener("mousemove", positionTooltip);
  el.addEventListener("mouseleave", () => {
    tooltipEl.hidden = true;
  });
}
function positionTooltip(e) {
  tooltipEl.style.left = `${e.clientX}px`;
  tooltipEl.style.top = `${e.clientY}px`;
}

// ---- panel: tactic performance ------------------------------------------------

function renderTacticTable(rows) {
  const tbody = document.querySelector("#tactic-table tbody");
  tbody.innerHTML = "";
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const netClass = r.net_value < 0 ? "negative" : "";
    tr.innerHTML = `
      <td>${r.tactic_name}</td>
      <td class="num">${r.deployments}</td>
      <td class="num">${fmtPercent(r.recovery_rate)}</td>
      <td class="num">${fmtSignedCurrency(r.avg_reward)}</td>
      <td class="num">${fmtCurrency(r.avg_cost)}</td>
      <td class="num ${netClass}">${fmtSignedCurrency(r.net_value)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ---- panel: bandit heatmap ----------------------------------------------------

function renderHeatmap(states) {
  const svg = document.getElementById("heatmap");
  const emptyEl = document.getElementById("heatmap-empty");
  svg.innerHTML = "";

  if (!states.length) {
    emptyEl.hidden = false;
    svg.setAttribute("height", 0);
    return;
  }
  emptyEl.hidden = true;

  const segments = [...new Set(states.map((s) => s.segment_key))].sort();
  const tactics = [...new Set(states.map((s) => s.tactic_name))].sort();
  const cellW = 92;
  const cellH = 24;
  const labelW = 200;
  const topH = 90;
  const width = labelW + tactics.length * cellW + 10;
  const height = topH + segments.length * cellH + 10;

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);

  const byKey = new Map(states.map((s) => [`${s.segment_key}|${s.tactic_name}`, s]));

  tactics.forEach((t, ci) => {
    const x = labelW + ci * cellW + cellW / 2;
    const text = svgEl("text", { x, y: topH - 12, "text-anchor": "start", transform: `rotate(-40 ${x} ${topH - 12})` });
    text.textContent = t;
    svg.appendChild(text);
  });

  segments.forEach((seg, ri) => {
    const y = topH + ri * cellH;
    const label = svgEl("text", { x: labelW - 10, y: y + cellH / 2 + 4, "text-anchor": "end" });
    label.textContent = seg;
    svg.appendChild(label);

    tactics.forEach((t, ci) => {
      const x = labelW + ci * cellW;
      const state = byKey.get(`${seg}|${t}`);
      const rect = svgEl("rect", {
        x: x + 1,
        y: y + 1,
        width: cellW - 2,
        height: cellH - 2,
        rx: 3,
        class: "heat-cell",
        fill: state ? mixAccent(state.estimated_win_rate) : "transparent",
      });
      rect.style.animation = "fadeIn 0.4s ease both";
      rect.style.animationDelay = `${(ri * tactics.length + ci) * 3}ms`;
      if (state) {
        attachTooltip(
          rect,
          () =>
            `<strong>${t}</strong><br>${seg}<br>Est. win rate: ${(state.estimated_win_rate * 100).toFixed(1)}%` +
            `<span class="muted">alpha=${state.alpha.toFixed(2)}  beta=${state.beta.toFixed(2)}</span>`
        );
      }
      svg.appendChild(rect);
    });
  });
}

// ---- panel: attribution -------------------------------------------------------

function renderAttribution(rows) {
  const tbody = document.querySelector("#attribution-table tbody");
  const emptyEl = document.getElementById("attribution-empty");
  const banner = document.getElementById("low-data-banner");
  tbody.innerHTML = "";

  if (!rows.length) {
    emptyEl.hidden = false;
    banner.hidden = true;
    return;
  }
  emptyEl.hidden = true;

  let lowDataCount = 0;
  rows.forEach((r) => {
    const reliable = r.confidence === "reliable";
    if (!reliable) lowDataCount += 1;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.segment_key}</td>
      <td class="num">${r.control_count}</td>
      <td class="num">${fmtPercent(r.control_recovery_rate)}</td>
      <td class="num">${r.treated_count}</td>
      <td class="num">${fmtPercent(r.treated_recovery_rate)}</td>
      <td class="num">${fmtSignedPercent(r.uplift_pct)}</td>
      <td><span class="badge ${reliable ? "badge-reliable" : "badge-low-data"}">${r.confidence}</span></td>
    `;
    tbody.appendChild(tr);
  });

  if (lowDataCount > 0) {
    banner.hidden = false;
    banner.textContent = `${lowDataCount} segment(s) have fewer than 5 control accounts — their baseline (and uplift) isn't reliable yet.`;
  } else {
    banner.hidden = true;
  }
}

// ---- panel: recovery funnel ----------------------------------------------------

function renderFunnel(stages) {
  const svg = document.getElementById("funnel");
  svg.innerHTML = "";
  if (!stages.length) return;

  const width = 720;
  const height = 260;
  const padTop = 34;
  const padBottom = 34;
  const padSide = 30;
  const gap = 28;
  const maxCount = Math.max(...stages.map((s) => s.count), 1);
  const barW = (width - padSide * 2 - gap * (stages.length - 1)) / stages.length;

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  stages.forEach((s, i) => {
    const barH = (s.count / maxCount) * (height - padTop - padBottom);
    const x = padSide + i * (barW + gap);
    const y = height - padBottom - barH;
    const opacity = (1 - i * 0.2).toFixed(2);

    const rect = svgEl("rect", {
      x,
      y,
      width: barW,
      height: Math.max(barH, 1),
      rx: 4,
      class: "funnel-bar",
      fill: ACCENT,
      opacity,
    });
    rect.style.transformBox = "fill-box";
    rect.style.transformOrigin = "bottom";
    rect.style.animation = "growUp 0.5s cubic-bezier(0.4,0,0.2,1) both";
    rect.style.animationDelay = `${i * 80}ms`;
    attachTooltip(rect, () => `<strong>${s.stage}</strong><br>${s.count} accounts`);
    svg.appendChild(rect);

    const valueLabel = svgEl("text", { x: x + barW / 2, y: y - 8, "text-anchor": "middle", class: "funnel-label" });
    valueLabel.textContent = s.count;
    svg.appendChild(valueLabel);

    const stageLabel = svgEl("text", { x: x + barW / 2, y: height - padBottom + 18, "text-anchor": "middle" });
    stageLabel.textContent = s.stage;
    svg.appendChild(stageLabel);
  });
}

// ---- panel: reward over time ---------------------------------------------------

function renderRewardChart(points) {
  const svg = document.getElementById("reward-chart");
  const emptyEl = document.getElementById("reward-empty");
  const legendEl = document.getElementById("reward-legend");
  svg.innerHTML = "";
  legendEl.innerHTML = "";

  if (!points.length) {
    emptyEl.hidden = false;
    return;
  }
  emptyEl.hidden = true;

  const dates = [...new Set(points.map((p) => p.date))].sort();
  const tacticsPresent = [...new Set(points.map((p) => p.tactic_name))];

  const width = 900;
  const height = 340;
  const pad = { top: 16, right: 16, bottom: 34, left: 64 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const xScale = (d) => pad.left + (dates.length > 1 ? dates.indexOf(d) / (dates.length - 1) : 0.5) * innerW;
  const values = points.map((p) => p.avg_reward);
  const minV = Math.min(0, ...values);
  const maxV = Math.max(...values, 1);
  const yScale = (v) => pad.top + innerH - ((v - minV) / (maxV - minV)) * innerH;

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = minV + ((maxV - minV) * i) / ticks;
    const y = yScale(v);
    svg.appendChild(svgEl("line", { x1: pad.left, x2: width - pad.right, y1: y, y2: y, class: "grid-line" }));
    const label = svgEl("text", { x: pad.left - 10, y: y + 4, "text-anchor": "end" });
    label.textContent = fmtCompact(v);
    svg.appendChild(label);
  }

  // x-axis date labels (start, middle, end -- avoid crowding)
  [0, Math.floor((dates.length - 1) / 2), dates.length - 1].forEach((idx) => {
    if (idx < 0 || idx >= dates.length) return;
    const d = dates[idx];
    const label = svgEl("text", { x: xScale(d), y: height - pad.bottom + 20, "text-anchor": "middle" });
    label.textContent = d;
    svg.appendChild(label);
  });

  // reveal clip: the whole chart draws in left-to-right, independent of
  // each line's own dash pattern (stroke-dasharray can't serve both the
  // draw-in animation and the per-tactic identity pattern at once).
  const clipId = "reward-reveal-clip";
  const defs = svgEl("defs", {});
  const clipRect = svgEl("rect", { x: 0, y: 0, width: 0, height });
  const clipPath = svgEl("clipPath", { id: clipId });
  clipPath.appendChild(clipRect);
  defs.appendChild(clipPath);
  svg.appendChild(defs);

  const dataGroup = svgEl("g", { "clip-path": `url(#${clipId})` });
  svg.appendChild(dataGroup);

  tacticsPresent.forEach((tactic) => {
    const style = TACTIC_STYLE[tactic] || { color: TEXT_SECONDARY, dash: "none" };
    const series = points
      .filter((p) => p.tactic_name === tactic)
      .sort((a, b) => dates.indexOf(a.date) - dates.indexOf(b.date));

    const d = series
      .map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.date).toFixed(1)} ${yScale(p.avg_reward).toFixed(1)}`)
      .join(" ");

    const path = svgEl("path", { d, class: "reward-line", stroke: style.color });
    if (style.dash !== "none") path.setAttribute("stroke-dasharray", style.dash);
    dataGroup.appendChild(path);

    series.forEach((p) => {
      const dot = svgEl("circle", { cx: xScale(p.date), cy: yScale(p.avg_reward), r: 3, fill: style.color, class: "reward-dot" });
      attachTooltip(dot, () => `<strong>${tactic}</strong><br>${p.date}<br>Avg reward: ${fmtSignedCurrency(p.avg_reward)}`);
      dataGroup.appendChild(dot);
    });

    const legendItem = document.createElement("div");
    legendItem.className = "legend-item";
    legendItem.innerHTML = `<span class="legend-swatch" style="border-top-color:${style.color}; border-top-style:${CSS_DASH_STYLE[style.dash]}"></span>${tactic}`;
    legendEl.appendChild(legendItem);
  });

  clipRect.animate([{ width: "0px" }, { width: `${width}px` }], { duration: 700, easing: "cubic-bezier(0.4,0,0.2,1)", fill: "forwards" });
}

// ---- orchestration ----------------------------------------------------------------

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`);
  return res.json();
}

async function loadAll() {
  const refreshBtn = document.getElementById("refresh-btn");
  refreshBtn.classList.add("spinning");

  const [tactic, bandit, attribution, funnel, reward] = await Promise.allSettled([
    fetchJSON("/stats/tactic-performance"),
    fetchJSON("/bandit/state"),
    fetchJSON("/stats/attribution"),
    fetchJSON("/stats/funnel"),
    fetchJSON("/stats/reward-over-time"),
  ]);

  if (tactic.status === "fulfilled") renderTacticTable(tactic.value.tactics);
  else console.error(tactic.reason);

  if (bandit.status === "fulfilled") renderHeatmap(bandit.value.arms);
  else console.error(bandit.reason);

  if (attribution.status === "fulfilled") renderAttribution(attribution.value.rows);
  else console.error(attribution.reason);

  if (funnel.status === "fulfilled") renderFunnel(funnel.value.stages);
  else console.error(funnel.reason);

  if (reward.status === "fulfilled") renderRewardChart(reward.value.points);
  else console.error(reward.reason);

  document.getElementById("last-updated").textContent = `Last loaded: ${new Date().toLocaleTimeString()}`;
  setTimeout(() => refreshBtn.classList.remove("spinning"), 400);
}

document.getElementById("refresh-btn").addEventListener("click", loadAll);
loadAll();
setInterval(loadAll, 30000);
