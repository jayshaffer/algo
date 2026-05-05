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

// === Hamburger toggle (all pages) ===
function setupHamburger() {
  var btn = document.querySelector(".site-nav .hamburger");
  var links = document.querySelector(".site-nav .links");
  if (!btn || !links) return;
  btn.addEventListener("click", function () {
    links.classList.toggle("open");
  });
}

// === Equity / benchmark charts (Performance page) ===

function renderEquityCurve(snapshots) {
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

  var labels = snapshots.map(function (s) { return s.date; });
  var equityValues = snapshots.map(function (s) { return s.portfolio_value; });
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

  new Chart(canvas, {
    type: "line",
    data: { labels: labels, datasets: datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { display: datasets.length > 1, labels: { color: "#8b949e" } },
        tooltip: {
          callbacks: {
            label: function (ctx) {
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

function renderPnlChart(snapshots) {
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

  var labels = snapshots.map(function (s) { return s.date; });
  var pnlValues = snapshots.map(function (s) {
    var pv = s.portfolio_value;
    var dep = s.cumulative_deposits;
    if (pv == null || dep == null) return null;
    return pv - dep;
  });

  new Chart(canvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: [{
        label: "Cumulative P&L",
        data: pnlValues,
        borderColor: "#58a6ff",
        backgroundColor: "rgba(88, 166, 255, 0.08)",
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        pointHitRadius: 8,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function (ctx) {
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

function renderBenchmark(snapshots, benchmark) {
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

  var spyReturns = labels.map(function (date) {
    var close = spyMap[date];
    if (close == null) return null;
    return ((close - spyBase) / spyBase) * 100;
  });

  new Chart(canvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
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
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true, labels: { color: "#8b949e" } },
        tooltip: {
          callbacks: {
            label: function (ctx) {
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
    var tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" + escapeHtml(d.date || "—") + "</td>" +
      "<td><strong>" + escapeHtml(d.ticker || "—") + "</strong></td>" +
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
  ]).then(function (parts) {
    var snapshots = parts[0];
    var benchmark = parts[1];
    renderEquityCurve(snapshots);
    renderPnlChart(snapshots);
    renderBenchmark(snapshots, benchmark);
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
  var page = document.body.dataset.page;
  switch (page) {
    case "performance":
      initPerformancePage();
      break;
    case "activity":
      initActivityPage();
      break;
    case "home":
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
