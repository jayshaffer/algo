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

function truncate(s, max) {
  if (!s) return "—";
  return s.length > max ? s.slice(0, max) + "..." : s;
}

function shortOrderId(id) {
  if (!id) return "—";
  return id.length > 12 ? id.slice(0, 8) + "..." : id;
}

// === Data Fetching ===

async function fetchJSON(file) {
  var resp = await fetch("data/" + file);
  if (!resp.ok) return null;
  return resp.json();
}

async function fetchAllData() {
  var [summary, snapshots, positions, decisions, theses, benchmark] = await Promise.all([
    fetchJSON("summary.json"),
    fetchJSON("snapshots.json"),
    fetchJSON("positions.json"),
    fetchJSON("decisions.json"),
    fetchJSON("theses.json"),
    fetchJSON("benchmark.json"),
  ]);
  return { summary: summary, snapshots: snapshots, positions: positions,
           decisions: decisions, theses: theses, benchmark: benchmark };
}

// === Renderers ===

function renderSummary(s, snapshots, benchmark) {
  if (!s) return;

  document.getElementById("last-updated").textContent =
    s.last_updated ? "Last updated " + s.last_updated : "";

  document.getElementById("portfolio-value").textContent = formatCurrency(s.portfolio_value);

  var dailyEl = document.getElementById("daily-pnl");
  if (s.daily_pnl != null) {
    dailyEl.textContent = formatCurrency(s.daily_pnl) + " (" + formatPct(s.daily_pnl_pct) + ")";
    dailyEl.className = "card-value " + pnlClass(s.daily_pnl);
  }

  var totalEl = document.getElementById("total-return");
  if (s.total_pnl != null) {
    totalEl.textContent = formatCurrency(s.total_pnl) + " (" + formatPct(s.total_pnl_pct) + ")";
    totalEl.className = "card-value " + pnlClass(s.total_pnl);
  }

  document.getElementById("positions-count").textContent =
    s.positions_count != null ? s.positions_count : "—";

  document.getElementById("cash-value").textContent = formatCurrency(s.cash);

  // vs S&P: portfolio return minus SPY return
  var vsSp = document.getElementById("vs-sp");
  if (snapshots && snapshots.length > 1 && benchmark && benchmark.length > 0) {
    var baseDeposits = snapshots[0].cumulative_deposits || snapshots[0].portfolio_value;
    var nowDeposits = snapshots[snapshots.length - 1].cumulative_deposits || baseDeposits;
    var portfolioNow = snapshots[snapshots.length - 1].portfolio_value;
    var portfolioReturn = ((portfolioNow - nowDeposits) / baseDeposits) * 100;

    var spyMap = {};
    benchmark.forEach(function (b) { spyMap[b.date] = b.close; });
    // Find first/last available benchmark close (dates may not align on weekends)
    var spyStart = null;
    for (var i = 0; i < snapshots.length; i++) {
      if (spyMap[snapshots[i].date] != null) { spyStart = spyMap[snapshots[i].date]; break; }
    }
    var spyEnd = null;
    for (var j = snapshots.length - 1; j >= 0; j--) {
      if (spyMap[snapshots[j].date] != null) { spyEnd = spyMap[snapshots[j].date]; break; }
    }

    if (spyStart && spyEnd) {
      var spyReturn = ((spyEnd - spyStart) / spyStart) * 100;
      var alpha = portfolioReturn - spyReturn;
      vsSp.textContent = formatPct(alpha);
      vsSp.className = "card-value " + pnlClass(alpha);
    }
  }
}

function renderEquityCurve(snapshots, benchmark) {
  var canvas = document.getElementById("equity-chart");
  var emptyMsg = document.getElementById("chart-empty");

  if (!snapshots || snapshots.length === 0) {
    canvas.style.display = "none";
    emptyMsg.style.display = "block";
    return;
  }

  var labels = snapshots.map(function (s) { return s.date; });

  // Normalize portfolio to % return
  var baseValue = snapshots[0].portfolio_value;
  var portfolioReturns = snapshots.map(function (s) {
    return ((s.portfolio_value - baseValue) / baseValue) * 100;
  });

  var datasets = [{
    label: "Portfolio",
    data: portfolioReturns,
    borderColor: "#00d4aa",
    backgroundColor: "rgba(0, 212, 170, 0.08)",
    fill: true,
    tension: 0.3,
    pointRadius: 0,
    pointHitRadius: 8,
    borderWidth: 2,
  }];

  // Add SPY benchmark if available
  if (benchmark && benchmark.length > 0) {
    // Build a date->close map for SPY
    var spyMap = {};
    benchmark.forEach(function (b) { spyMap[b.date] = b.close; });

    // Find SPY base value matching the first snapshot date
    var spyBase = spyMap[labels[0]];

    if (spyBase) {
      var spyReturns = labels.map(function (date) {
        var close = spyMap[date];
        if (close == null) return null;
        return ((close - spyBase) / spyBase) * 100;
      });

      datasets.push({
        label: "S&P 500",
        data: spyReturns,
        borderColor: "#5a6a7a",
        borderDash: [6, 3],
        backgroundColor: "transparent",
        fill: false,
        tension: 0.3,
        pointRadius: 0,
        pointHitRadius: 8,
        borderWidth: 2,
      });
    }
  }

  new Chart(canvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: datasets,
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: datasets.length > 1, labels: { color: "#8892a4" } },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              return "Portfolio: " + formatCurrency(ctx.parsed.y);
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#8892a4", maxTicksLimit: 8 },
          grid: { color: "rgba(30, 58, 95, 0.4)" },
        },
        y: {
          ticks: {
            color: "#8892a4",
            callback: function (v) { return formatCurrency(v); },
          },
          grid: { color: "rgba(30, 58, 95, 0.4)" },
        },
      },
    },
  });
}

function renderBenchmark(snapshots, benchmark) {
  var canvas = document.getElementById("benchmark-chart");
  var emptyMsg = document.getElementById("benchmark-empty");

  if (!snapshots || snapshots.length === 0 || !benchmark || benchmark.length === 0) {
    canvas.style.display = "none";
    emptyMsg.style.display = "block";
    return;
  }

  var labels = snapshots.map(function (s) { return s.date; });

  // Normalize portfolio to % return, adjusted for cash deposits/withdrawals
  // Return = (portfolio_value - cumulative_deposits) / first_deposits * 100
  var baseDeposits = snapshots[0].cumulative_deposits || snapshots[0].portfolio_value;
  var portfolioReturns = snapshots.map(function (s) {
    var deposits = s.cumulative_deposits || baseDeposits;
    return ((s.portfolio_value - deposits) / baseDeposits) * 100;
  });

  // Build a date->close map for SPY
  var spyMap = {};
  benchmark.forEach(function (b) { spyMap[b.date] = b.close; });

  // Find first available benchmark close (dates may not align on weekends)
  var spyBase = null;
  for (var i = 0; i < labels.length; i++) {
    if (spyMap[labels[i]] != null) { spyBase = spyMap[labels[i]]; break; }
  }
  if (!spyBase) {
    canvas.style.display = "none";
    emptyMsg.style.display = "block";
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
          borderColor: "#00d4aa",
          backgroundColor: "rgba(0, 212, 170, 0.08)",
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHitRadius: 8,
          borderWidth: 2,
        },
        {
          label: "S&P 500",
          data: spyReturns,
          borderColor: "#5a6a7a",
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
        legend: { display: true, labels: { color: "#8892a4" } },
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
          ticks: { color: "#8892a4", maxTicksLimit: 8 },
          grid: { color: "rgba(30, 58, 95, 0.4)" },
        },
        y: {
          ticks: {
            color: "#8892a4",
            callback: function (v) { return (v >= 0 ? "+" : "") + v.toFixed(1) + "%"; },
          },
          grid: { color: "rgba(30, 58, 95, 0.4)" },
        },
      },
    },
  });
}

function renderPositions(positions) {
  var tbody = document.querySelector("#positions-table tbody");
  var emptyMsg = document.getElementById("positions-empty");

  if (!positions || positions.length === 0) {
    document.getElementById("positions-table").style.display = "none";
    emptyMsg.style.display = "block";
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

  if (!decisions || decisions.length === 0) {
    document.getElementById("decisions-table").style.display = "none";
    emptyMsg.style.display = "block";
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

  if (!theses || theses.length === 0) {
    emptyMsg.style.display = "block";
    return;
  }

  theses.forEach(function (t) {
    var card = document.createElement("div");
    card.className = "thesis-card";
    card.innerHTML =
      '<div class="thesis-header">' +
        '<span class="thesis-ticker">' + escapeHtml(t.ticker) + "</span>" +
        '<span class="thesis-direction ' + escapeHtml(t.direction || "") + '">' + escapeHtml(t.direction || "") + "</span>" +
        '<span class="thesis-confidence">' + escapeHtml(t.confidence || "") + "</span>" +
      "</div>" +
      '<p class="thesis-body">' + escapeHtml(t.thesis || "") + "</p>" +
      '<div class="thesis-triggers">' +
        "Entry: " + escapeHtml(t.entry_trigger || "—") + " &nbsp;|&nbsp; Exit: " + escapeHtml(t.exit_trigger || "—") +
      "</div>";
    container.appendChild(card);
  });
}

// === Init ===

document.addEventListener("DOMContentLoaded", function () {
  fetchAllData().then(function (data) {
    renderSummary(data.summary, data.snapshots, data.benchmark);
    renderEquityCurve(data.snapshots);
    renderBenchmark(data.snapshots, data.benchmark);
    renderPositions(data.positions);
    renderDecisions(data.decisions);
    renderTheses(data.theses);
  }).catch(function (err) {
    console.error("Failed to load dashboard data:", err);
    document.getElementById("last-updated").textContent = "Failed to load data";
  });
});
