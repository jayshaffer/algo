"use strict";

// === Helpers ===

function escapeHtml(s) {
  if (!s) return "";
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;")
          .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function formatCurrency(n) {
  if (n == null) return "—";
  return "$" + Number(n).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatPct(n) {
  if (n == null) return "—";
  var val = Number(n);
  var sign = val >= 0 ? "+" : "";
  return sign + val.toFixed(2) + "%";
}

function pnlClass(n) {
  if (n == null || Number(n) === 0) return "";
  return Number(n) >= 0 ? "gain" : "loss";
}

function shortOrderId(id) {
  if (!id) return "—";
  return id.length > 12 ? id.slice(0, 8) + "..." : id;
}

function tickerHref(ticker) {
  return "/ticker/" + encodeURIComponent(String(ticker || "").toUpperCase()) + "/";
}

function computeTWR(snapshots) {
  if (!snapshots || snapshots.length === 0) return [];

  var returns = [0];
  var cumulativeGrowth = 1.0;

  for (var i = 1; i < snapshots.length; i++) {
    var prevValue = snapshots[i - 1].portfolio_value;
    var prevDep = snapshots[i - 1].cumulative_deposits || snapshots[i - 1].portfolio_value;
    var currDep = snapshots[i].cumulative_deposits || prevDep;
    var deposit = currDep - prevDep;

    var startCapital = prevValue + deposit;
    if (startCapital > 0) {
      cumulativeGrowth *= snapshots[i].portfolio_value / startCapital;
    }

    returns.push((cumulativeGrowth - 1) * 100);
  }

  return returns;
}

async function fetchJSON(file) {
  var resp = await fetch("/data/" + file);
  if (!resp.ok) return null;
  return resp.json();
}

// === Chart range filter state ===

var DEFAULT_RANGE_DAYS = 7;

var perfData = {
  snapshots: null,
  benchmark: null,
  decisions: null,
  rangeStart: null,  // YYYY-MM-DD — inclusive lower bound
  rangeEnd: null,    // YYYY-MM-DD — inclusive upper bound
};
var chartInstances = {};  // canvasId -> Chart instance

function destroyCharts() {
  Object.keys(chartInstances).forEach(function (id) {
    try { chartInstances[id].destroy(); } catch (e) { /* already gone */ }
  });
  chartInstances = {};
}

function latestSnapshotDate(snapshots) {
  if (!snapshots || snapshots.length === 0) return null;
  // snapshots.json is date-ordered ascending; take the last row's date.
  return normalizeDate(snapshots[snapshots.length - 1].date);
}

function firstSnapshotDate(snapshots) {
  if (!snapshots || snapshots.length === 0) return null;
  return normalizeDate(snapshots[0].date);
}

function cutoffDate(anchorDate, days) {
  // anchorDate is a YYYY-MM-DD string. Subtract `days` calendar days
  // and return another YYYY-MM-DD string.
  //
  // All UTC (the "T00:00:00Z" suffix and the getUTCDate/setUTCDate pair)
  // to keep date-only arithmetic timezone-free — never simplify to
  // new Date(anchorDate) or get/setDate(), which would shift by a day
  // for users west of UTC.
  var d = new Date(anchorDate + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() - days);
  return d.toISOString().slice(0, 10);
}

function filterByDateWindow(rows, dateKey, start, end) {
  // start/end inclusive; both required. Used by applyRange.
  if (!rows) return rows;
  if (!start || !end) return rows;
  return rows.filter(function (r) {
    var d = normalizeDate(r[dateKey]);
    return d >= start && d <= end;
  });
}

function presetToWindow(preset) {
  // preset: "7" | "30" | "365" | "all". Returns { start, end, preset }.
  var anchor = latestSnapshotDate(perfData.snapshots);
  if (!anchor) return { start: null, end: null, preset: preset };
  if (preset === "all") {
    return { start: firstSnapshotDate(perfData.snapshots), end: anchor, preset: "all" };
  }
  var days = parseInt(preset, 10);
  return { start: cutoffDate(anchor, days), end: anchor, preset: preset };
}

function updatePresetActiveState(preset) {
  // preset === null when the user dragged the brush — clear all highlights.
  document.querySelectorAll(".range-btn").forEach(function (btn) {
    var isActive = preset != null && btn.getAttribute("data-range") === String(preset);
    btn.classList.toggle("is-active", isActive);
    btn.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

function applyRange(spec) {
  // spec: { start: YYYY-MM-DD, end: YYYY-MM-DD, preset: "7"|"30"|"365"|"all"|null }
  perfData.rangeStart = spec.start;
  perfData.rangeEnd = spec.end;
  destroyCharts();
  var filteredSnapshots = filterByDateWindow(perfData.snapshots, "date", spec.start, spec.end);
  var filteredBenchmark = filterByDateWindow(perfData.benchmark, "date", spec.start, spec.end);
  var filteredDecisions = filterByDateWindow(perfData.decisions, "date", spec.start, spec.end);
  renderEquityCurve(filteredSnapshots, filteredDecisions);
  renderPnlChart(filteredSnapshots, filteredDecisions);
  renderBenchmark(filteredSnapshots, filteredBenchmark, filteredDecisions);
  updatePresetActiveState(spec.preset);
  updateBrushPosition(spec.start, spec.end);
}

// === Brush strip ===

var brush = {
  el: null,           // .range-brush element
  svg: null,          // child <svg>
  overviewPath: null, // <path> showing the full timeline polyline
  bar: null,          // <rect> highlighting the selected window
  leftHandle: null,   // <rect> at the left edge
  rightHandle: null,  // <rect> at the right edge
  leftHit: null,      // wider invisible <rect> for easier grabbing
  rightHit: null,
  width: 0,           // measured pixel width of the svg
  domain: null,       // { first: "YYYY-MM-DD", last: "YYYY-MM-DD" }
};

var SVG_NS = "http://www.w3.org/2000/svg";
var BRUSH_HANDLE_WIDTH = 6;
var BRUSH_HANDLE_HIT_WIDTH = 18;

function dateToFraction(date) {
  // Linear interpolation between brush.domain.first (0) and .last (1).
  if (!brush.domain) return 0;
  var first = new Date(brush.domain.first + "T00:00:00Z").getTime();
  var last = new Date(brush.domain.last + "T00:00:00Z").getTime();
  var d = new Date(date + "T00:00:00Z").getTime();
  if (last === first) return 0;
  var f = (d - first) / (last - first);
  return Math.max(0, Math.min(1, f));
}

function fractionToDate(fraction) {
  if (!brush.domain) return null;
  var first = new Date(brush.domain.first + "T00:00:00Z").getTime();
  var last = new Date(brush.domain.last + "T00:00:00Z").getTime();
  var t = first + (last - first) * Math.max(0, Math.min(1, fraction));
  return new Date(t).toISOString().slice(0, 10);
}

function renderBrushOverview() {
  // One polyline of portfolio_value across all snapshots, normalized
  // into the SVG's 0..100 y-coordinate space (preserveAspectRatio:none
  // stretches to the strip height).
  var snapshots = perfData.snapshots;
  if (!snapshots || snapshots.length === 0) return;
  var values = snapshots.map(function (s) { return s.portfolio_value; });
  var min = Math.min.apply(null, values);
  var max = Math.max.apply(null, values);
  var range = max - min || 1;
  var pts = snapshots.map(function (s) {
    var x = dateToFraction(normalizeDate(s.date)) * 100;
    var y = 100 - ((s.portfolio_value - min) / range) * 80 - 10;  // 10% padding top/bottom
    return x.toFixed(2) + "," + y.toFixed(2);
  });
  brush.overviewPath.setAttribute("d", "M " + pts.join(" L "));
}

function initBrush() {
  brush.el = document.querySelector(".range-brush");
  if (!brush.el) return;  // no brush container (e.g. activity page)
  if (!perfData.snapshots || perfData.snapshots.length === 0) return;

  brush.svg = brush.el.querySelector(".range-brush-svg");
  brush.svg.setAttribute("viewBox", "0 0 100 100");
  brush.domain = {
    first: firstSnapshotDate(perfData.snapshots),
    last: latestSnapshotDate(perfData.snapshots),
  };

  // Build SVG children in z-order (back to front).
  brush.overviewPath = document.createElementNS(SVG_NS, "path");
  brush.overviewPath.setAttribute("class", "range-brush-overview");
  brush.overviewPath.setAttribute("vector-effect", "non-scaling-stroke");
  brush.svg.appendChild(brush.overviewPath);

  brush.bar = document.createElementNS(SVG_NS, "rect");
  brush.bar.setAttribute("class", "range-brush-bar");
  brush.bar.setAttribute("y", "0");
  brush.bar.setAttribute("height", "100");
  brush.bar.setAttribute("data-role", "bar");
  brush.svg.appendChild(brush.bar);

  function makeHandle(role, visibleClass, hitClass) {
    var visible = document.createElementNS(SVG_NS, "rect");
    visible.setAttribute("class", visibleClass);
    visible.setAttribute("y", "0");
    visible.setAttribute("height", "100");
    visible.setAttribute("width", String(BRUSH_HANDLE_WIDTH / 2));
    visible.setAttribute("data-role", role + "-handle");
    brush.svg.appendChild(visible);

    var hit = document.createElementNS(SVG_NS, "rect");
    hit.setAttribute("class", hitClass);
    hit.setAttribute("y", "0");
    hit.setAttribute("height", "100");
    hit.setAttribute("width", String(BRUSH_HANDLE_HIT_WIDTH / 2));
    hit.setAttribute("data-role", role + "-hit");
    brush.svg.appendChild(hit);
    return { visible: visible, hit: hit };
  }

  var left = makeHandle("left", "range-brush-handle", "range-brush-handle-hit");
  brush.leftHandle = left.visible;
  brush.leftHit = left.hit;
  var right = makeHandle("right", "range-brush-handle", "range-brush-handle-hit");
  brush.rightHandle = right.visible;
  brush.rightHit = right.hit;

  renderBrushOverview();
  window.addEventListener("resize", function () {
    // No DOM measurement needed — viewBox is 0..100 so the SVG scales.
    // Kept as a hook in case future code needs pixel measurements.
  });
}

function updateBrushPosition(start, end) {
  // Move handles and bar to reflect the current window. No-op if the
  // brush hasn't been initialized (e.g. on the activity page).
  if (!brush.svg || !brush.domain || !start || !end) return;
  var leftFrac = dateToFraction(start) * 100;
  var rightFrac = dateToFraction(end) * 100;
  brush.bar.setAttribute("x", String(leftFrac));
  brush.bar.setAttribute("width", String(Math.max(0, rightFrac - leftFrac)));
  var halfVis = BRUSH_HANDLE_WIDTH / 2 / 2;
  var halfHit = BRUSH_HANDLE_HIT_WIDTH / 2 / 2;
  brush.leftHandle.setAttribute("x", String(leftFrac - halfVis));
  brush.leftHit.setAttribute("x", String(leftFrac - halfHit));
  brush.rightHandle.setAttribute("x", String(rightFrac - halfVis));
  brush.rightHit.setAttribute("x", String(rightFrac - halfHit));
}

function setupRangeControl() {
  var control = document.querySelector(".range-control");
  if (!control) return;
  control.addEventListener("click", function (e) {
    var btn = e.target.closest(".range-btn");
    if (!btn) return;
    var raw = btn.getAttribute("data-range");
    applyRange(presetToWindow(raw));
  });
}

// === Hamburger toggle (all pages) ===
function setupHamburger() {
  var btn = document.querySelector(".site-nav .hamburger");
  var links = document.querySelector(".site-nav .links");
  if (!btn || !links) return;
  btn.addEventListener("click", function () {
    links.classList.toggle("open");
  });
}

function setupBackButtons() {
  document.querySelectorAll("[data-back-fallback]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var fallback = btn.getAttribute("data-back-fallback") || "/";
      if (window.history.length > 1) {
        window.history.back();
      } else {
        window.location.href = fallback;
      }
    });
  });
}

// === Equity / benchmark charts ===

function normalizeDate(s) {
  if (!s) return "";
  return String(s).slice(0, 10);
}

function buildDecisionMarkers(decisions, valueByDate, label) {
  if (!decisions || !valueByDate) return [];

  var buys = [];
  var sells = [];
  decisions.forEach(function (d) {
    var action = String(d.action || "").toLowerCase();
    if (action !== "buy" && action !== "sell") return;
    var date = normalizeDate(d.date);
    if (valueByDate[date] == null) return;
    var point = {
      x: date,
      y: valueByDate[date],
      ticker: d.ticker || "",
      action: action,
      quantity: d.quantity,
    };
    if (action === "buy") buys.push(point);
    if (action === "sell") sells.push(point);
  });

  function dataset(name, points, color, pointStyle) {
    return {
      type: "scatter",
      label: name,
      data: points,
      borderColor: color,
      backgroundColor: color,
      pointStyle: pointStyle,
      pointRadius: 5,
      pointHoverRadius: 7,
      showLine: false,
      valueLabel: label,
    };
  }

  return [
    dataset("Buys", buys, "#3fb950", "triangle"),
    dataset("Sells", sells, "#f85149", "rectRot"),
  ].filter(function (d) { return d.data.length > 0; });
}

function renderEquityCurve(snapshots, decisions) {
  // Plots actual account equity (portfolio_value) in dollars over time.
  // Deposits stair-step the line up; trading P&L is the wiggle on top.
  // SPY line shows the deposit-matched shadow portfolio (spy_value_if_deposited)
  // computed server-side — what the same cash flows would be worth had they
  // gone into SPY instead. Apples-to-apples vs an account that gets new
  // deposits at irregular intervals.
  var canvas = document.getElementById("equity-chart");
  var emptyMsg = document.getElementById("chart-empty");
  if (!canvas) return;

  if (!snapshots || snapshots.length === 0) {
    canvas.style.display = "none";
    if (emptyMsg) emptyMsg.style.display = "block";
    return;
  }

  canvas.style.display = "";
  if (emptyMsg) emptyMsg.style.display = "none";

  var labels = snapshots.map(function (s) { return s.date; });
  var equityValues = snapshots.map(function (s) { return s.portfolio_value; });
  var equityByDate = {};
  snapshots.forEach(function (s) { equityByDate[normalizeDate(s.date)] = s.portfolio_value; });
  var spyValues = snapshots.map(function (s) {
    return s.spy_value_if_deposited != null ? s.spy_value_if_deposited : null;
  });
  var hasSpy = spyValues.some(function (v) { return v != null; });

  var datasets = [{
    label: "Portfolio",
    data: equityValues,
    borderColor: "#58a6ff",
    backgroundColor: "rgba(88, 166, 255, 0.08)",
    fill: true,
    tension: 0.3,
    pointRadius: 0,
    pointHitRadius: 8,
    borderWidth: 2,
  }];

  if (hasSpy) {
    datasets.push({
      label: "S&P 500 (deposit-matched)",
      data: spyValues,
      borderColor: "#8b949e",
      borderDash: [6, 3],
      backgroundColor: "transparent",
      fill: false,
      tension: 0.3,
      pointRadius: 0,
      pointHitRadius: 8,
      borderWidth: 2,
    });
  }

  datasets = datasets.concat(buildDecisionMarkers(decisions, equityByDate, "Portfolio"));

  chartInstances["equity-chart"] = new Chart(canvas, {
    type: "line",
    data: { labels: labels, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: datasets.length > 1, labels: { color: "#8b949e" } },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              if (ctx.raw && ctx.raw.action) {
                return ctx.raw.action.toUpperCase() + " " + ctx.raw.ticker + ": " +
                  formatCurrency(ctx.raw.y);
              }
              return ctx.dataset.label + ": " + formatCurrency(ctx.parsed.y);
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8b949e", maxTicksLimit: 8 },
          grid: { color: "rgba(48, 54, 61, 0.5)" },
        },
        y: {
          ticks: {
            color: "#8b949e",
            callback: function (v) { return formatCurrency(v); },
          },
          grid: { color: "rgba(48, 54, 61, 0.5)" },
        },
      },
    },
  });
}

function renderPnlChart(snapshots, decisions) {
  // Cumulative P&L in dollars: portfolio_value - cumulative_deposits.
  // Deposits add to both terms simultaneously, so the line tracks pure trading
  // gain/loss rather than the gross equity number.
  var canvas = document.getElementById("pnl-chart");
  var emptyMsg = document.getElementById("pnl-empty");
  if (!canvas) return;

  if (!snapshots || snapshots.length === 0) {
    canvas.style.display = "none";
    if (emptyMsg) emptyMsg.style.display = "block";
    return;
  }

  canvas.style.display = "";
  if (emptyMsg) emptyMsg.style.display = "none";

  var labels = snapshots.map(function (s) { return s.date; });
  var pnlValues = snapshots.map(function (s) {
    var pv = s.portfolio_value;
    var dep = s.cumulative_deposits;
    if (pv == null || dep == null) return null;
    return pv - dep;
  });
  var pnlByDate = {};
  snapshots.forEach(function (s, i) { pnlByDate[normalizeDate(s.date)] = pnlValues[i]; });
  var datasets = [{
    label: "Cumulative P&L",
    data: pnlValues,
    borderColor: "#58a6ff",
    backgroundColor: "rgba(88, 166, 255, 0.08)",
    fill: true,
    tension: 0.3,
    pointRadius: 0,
    pointHitRadius: 8,
    borderWidth: 2,
  }].concat(buildDecisionMarkers(decisions, pnlByDate, "P&L"));

  chartInstances["pnl-chart"] = new Chart(canvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              if (ctx.raw && ctx.raw.action) {
                var markerSign = ctx.raw.y >= 0 ? "+" : "−";
                return ctx.raw.action.toUpperCase() + " " + ctx.raw.ticker + ": " +
                  markerSign + formatCurrency(Math.abs(ctx.raw.y));
              }
              var v = ctx.parsed.y;
              var sign = v >= 0 ? "+" : "−";
              return sign + formatCurrency(Math.abs(v));
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8b949e", maxTicksLimit: 8 },
          grid: { color: "rgba(48, 54, 61, 0.5)" },
        },
        y: {
          ticks: {
            color: "#8b949e",
            callback: function (v) {
              var sign = v >= 0 ? "+" : "−";
              return sign + formatCurrency(Math.abs(v));
            },
          },
          grid: {
            color: function (ctx) {
              return ctx.tick.value === 0 ? "#8b949e" : "rgba(48, 54, 61, 0.5)";
            },
          },
        },
      },
    },
  });
}

function renderBenchmark(snapshots, benchmark, decisions) {
  var canvas = document.getElementById("benchmark-chart");
  var emptyMsg = document.getElementById("benchmark-empty");
  if (!canvas) return;

  if (!snapshots || snapshots.length === 0 || !benchmark || benchmark.length === 0) {
    canvas.style.display = "none";
    if (emptyMsg) emptyMsg.style.display = "block";
    return;
  }

  var labels = snapshots.map(function (s) { return s.date; });
  var portfolioReturns = computeTWR(snapshots);

  var spyMap = {};
  benchmark.forEach(function (b) { spyMap[b.date] = b.close; });

  var spyBase = null;
  for (var i = 0; i < labels.length; i++) {
    if (spyMap[labels[i]] != null) { spyBase = spyMap[labels[i]]; break; }
  }
  if (!spyBase) {
    canvas.style.display = "none";
    if (emptyMsg) emptyMsg.style.display = "block";
    return;
  }

  // Recovery: data IS renderable, so undo any prior empty-state hiding.
  canvas.style.display = "";
  if (emptyMsg) emptyMsg.style.display = "none";

  var spyReturns = labels.map(function (date) {
    var close = spyMap[date];
    if (close == null) return null;
    return ((close - spyBase) / spyBase) * 100;
  });
  var returnByDate = {};
  labels.forEach(function (date, i) { returnByDate[normalizeDate(date)] = portfolioReturns[i]; });
  var datasets = [
    {
      label: "Portfolio",
      data: portfolioReturns,
      borderColor: "#58a6ff",
      backgroundColor: "rgba(88, 166, 255, 0.08)",
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      pointHitRadius: 8,
      borderWidth: 2,
    },
    {
      label: "S&P 500",
      data: spyReturns,
      borderColor: "#8b949e",
      borderDash: [6, 3],
      backgroundColor: "transparent",
      fill: false,
      tension: 0.3,
      pointRadius: 0,
      pointHitRadius: 8,
      borderWidth: 2,
    },
  ].concat(buildDecisionMarkers(decisions, returnByDate, "Return"));

  chartInstances["benchmark-chart"] = new Chart(canvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, labels: { color: "#8b949e" } },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              if (ctx.raw && ctx.raw.action) {
                return ctx.raw.action.toUpperCase() + " " + ctx.raw.ticker + ": " +
                  formatPct(ctx.raw.y);
              }
              return ctx.dataset.label + ": " + formatPct(ctx.parsed.y);
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8b949e", maxTicksLimit: 8 },
          grid: { color: "rgba(48, 54, 61, 0.5)" },
        },
        y: {
          ticks: {
            color: "#8b949e",
            callback: function (v) { return (v >= 0 ? "+" : "") + v.toFixed(1) + "%"; },
          },
          grid: { color: "rgba(48, 54, 61, 0.5)" },
        },
      },
    },
  });
}

// === Activity-page renderers ===

function renderPositions(positions) {
  var tbody = document.querySelector("#positions-table tbody");
  var emptyMsg = document.getElementById("positions-empty");
  if (!tbody) return;

  if (!positions || positions.length === 0) {
    var t = document.getElementById("positions-table");
    if (t) t.style.display = "none";
    if (emptyMsg) emptyMsg.style.display = "block";
    return;
  }

  positions.forEach(function (p) {
    var tr = document.createElement("tr");
    tr.innerHTML =
      "<td><strong>" + escapeHtml(p.ticker) + "</strong></td>" +
      '<td class="num">' + p.shares + "</td>" +
      '<td class="num">' + formatCurrency(p.avg_cost) + "</td>";
    tbody.appendChild(tr);
  });
}

function renderDecisions(decisions) {
  var tbody = document.querySelector("#decisions-table tbody");
  var emptyMsg = document.getElementById("decisions-empty");
  if (!tbody) return;

  if (!decisions || decisions.length === 0) {
    var t = document.getElementById("decisions-table");
    if (t) t.style.display = "none";
    if (emptyMsg) emptyMsg.style.display = "block";
    return;
  }

  decisions.forEach(function (d) {
    var badgeClass = "badge badge-" + escapeHtml(d.action || "hold");
    var ticker = d.ticker || "—";
    var tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" + escapeHtml(d.date || "—") + "</td>" +
      '<td><a href="' + tickerHref(ticker) + '"><strong>' + escapeHtml(ticker) + "</strong></a></td>" +
      '<td><span class="' + badgeClass + '">' + escapeHtml(d.action || "—") + "</span></td>" +
      '<td class="num">' + (d.quantity || "—") + "</td>" +
      '<td class="reasoning-cell">' + escapeHtml(d.reasoning || "—") + "</td>" +
      '<td class="num"><span class="order-id">' + escapeHtml(shortOrderId(d.order_id)) + "</span></td>";
    tbody.appendChild(tr);
  });
}

function renderTheses(theses) {
  var container = document.getElementById("theses-list");
  var emptyMsg = document.getElementById("theses-empty");
  if (!container) return;

  if (!theses || theses.length === 0) {
    if (emptyMsg) emptyMsg.style.display = "block";
    return;
  }

  theses.forEach(function (t) {
    var card = document.createElement("div");
    card.className = "thesis-card";
    var direction = t.direction || "";
    card.innerHTML =
      '<div class="head">' +
        '<span class="ticker">' + escapeHtml(t.ticker) + "</span>" +
        '<span class="direction ' + escapeHtml(direction) + '">' + escapeHtml(direction) + "</span>" +
      "</div>" +
      '<p class="body">' + escapeHtml(t.thesis || "") + "</p>" +
      '<div class="triggers">' +
        "Entry: " + escapeHtml(t.entry_trigger || "—") + " &nbsp;|&nbsp; Exit: " + escapeHtml(t.exit_trigger || "—") +
      "</div>";
    container.appendChild(card);
  });
}

// === Per-page initializers ===

function initPerformancePage() {
  Promise.all([
    fetchJSON("snapshots.json"),
    fetchJSON("benchmark.json"),
    fetchJSON("decisions.json"),
  ]).then(function (parts) {
    perfData.snapshots = parts[0];
    perfData.benchmark = parts[1];
    perfData.decisions = parts[2];
    setupRangeControl();
    initBrush();
    applyRange(presetToWindow(String(DEFAULT_RANGE_DAYS)));
  }).catch(function (err) {
    console.error("Failed to load performance data:", err);
  });
}

function initActivityPage() {
  Promise.all([
    fetchJSON("positions.json"),
    fetchJSON("theses.json"),
    fetchJSON("decisions.json"),
  ]).then(function (parts) {
    renderPositions(parts[0]);
    renderTheses(parts[1]);
    renderDecisions(parts[2]);
  }).catch(function (err) {
    console.error("Failed to load activity data:", err);
  });
}

// === Dispatcher ===

document.addEventListener("DOMContentLoaded", function () {
  setupHamburger();
  setupBackButtons();
  var page = document.body.dataset.page;
  switch (page) {
    case "performance":
    case "home":
      initPerformancePage();
      break;
    case "activity":
      initActivityPage();
      break;
    case "strategy":
    case "learning":
    case "how-it-works":
    case "mistakes":
    case "attribution":
      // Fully server-rendered.
      break;
    default:
      break;
  }
});
