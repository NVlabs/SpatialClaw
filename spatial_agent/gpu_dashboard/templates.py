"""Inline Jinja2 templates for the GPU dashboard.

Bootstrap 5.3.3 + Chart.js 4.4.4, matching the style of the main
visualization_server so both dashboards feel like part of one product.
"""

BASE_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{% block title %}Spatial Agent — GPU Dashboard{% endblock %}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  :root {
    --bg: #0f1420;
    --panel: #151b2b;
    --panel-2: #1b2236;
    --line: #2a3149;
    --ink: #e8ecf5;
    --muted: #96a0bd;
    --accent: #6c8cff;
    --ok: #25c786;
    --warn: #f5b841;
    --bad: #ef4f6a;
    --purple: #a07bff;
  }
  html, body { background: var(--bg); color: var(--ink); }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                      "Helvetica Neue", Arial, sans-serif; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { color: #9ab3ff; text-decoration: underline; }
  .navbar { background: #0b0f1a !important; border-bottom: 1px solid var(--line); }
  .navbar-brand { font-weight: 700; letter-spacing: 0.5px; color: #fff !important; }
  .navbar .nav-link { color: var(--muted) !important; }
  .navbar .nav-link.active, .navbar .nav-link:hover { color: #fff !important; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
           padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.25); }
  .panel h6 { color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em;
              font-size: 0.72em; margin: 0 0 10px 0; font-weight: 600; }
  .kpi { font-size: 1.8rem; font-weight: 700; letter-spacing: -0.01em; }
  .kpi-sub { color: var(--muted); font-size: 0.85em; }
  .node-card { background: var(--panel); border: 1px solid var(--line);
               border-radius: 12px; padding: 16px; height: 100%; }
  .node-card h5 { margin: 0 0 4px 0; font-weight: 600; font-size: 1.02rem; }
  .node-card .meta { color: var(--muted); font-size: 0.8em; margin-bottom: 10px; }
  .gpu-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
  .gpu-label { width: 52px; color: var(--muted); font-variant-numeric: tabular-nums;
               font-size: 0.85em; }
  .gpu-bar-wrap { flex: 1; height: 10px; background: var(--panel-2);
                  border-radius: 999px; overflow: hidden; position: relative; }
  .gpu-bar { height: 100%; border-radius: 999px;
             transition: width .3s ease, background-color .3s ease; }
  .gpu-util { width: 48px; text-align: right; font-variant-numeric: tabular-nums;
              font-weight: 600; }
  .gpu-sub { width: 150px; text-align: right; color: var(--muted); font-size: 0.78em;
             font-variant-numeric: tabular-nums; }
  .badge-soft { background: var(--panel-2); color: var(--muted); font-weight: 500;
                border: 1px solid var(--line); }
  .badge-vllm { background: rgba(108,140,255,.15); color: #9ab3ff;
                border: 1px solid rgba(108,140,255,.4); }
  .badge-gpu-tool { background: rgba(160,123,255,.15); color: #c6b3ff;
                    border: 1px solid rgba(160,123,255,.4); }
  .badge-both { background: rgba(37,199,134,.12); color: #8cf0c0;
                border: 1px solid rgba(37,199,134,.4); }
  .hl-good { color: var(--ok); }
  .hl-warn { color: var(--warn); }
  .hl-bad  { color: var(--bad); }
  .chart-wrap { background: var(--panel); border: 1px solid var(--line);
                border-radius: 12px; padding: 18px; }
  .form-select, .form-control { background: var(--panel-2); color: var(--ink);
                                border-color: var(--line); }
  .form-select:focus, .form-control:focus { background: var(--panel-2); color: var(--ink);
                                            border-color: var(--accent);
                                            box-shadow: 0 0 0 .15rem rgba(108,140,255,.25); }
  .btn-soft { background: var(--panel-2); color: var(--ink); border: 1px solid var(--line); }
  .btn-soft:hover { background: #222a42; color: #fff; }
  .btn-soft.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .table { color: var(--ink); }
  .table thead th { color: var(--muted); border-bottom-color: var(--line);
                    font-weight: 500; text-transform: uppercase; font-size: 0.72em;
                    letter-spacing: 0.08em; }
  .table td, .table th { border-top-color: var(--line); }
  hr { border-color: var(--line); opacity: 1; }
  .dim { color: var(--muted); }
  .stale { opacity: 0.45; }
  {% block extra_css %}{% endblock %}
</style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-dark">
  <div class="container-fluid px-4">
    <a class="navbar-brand" href="/">Spatial Agent — GPU Dashboard</a>
    <div class="collapse navbar-collapse">
      <ul class="navbar-nav me-auto">
        <li class="nav-item"><a class="nav-link {% if page=='overview' %}active{% endif %}" href="/">Overview</a></li>
        <li class="nav-item"><a class="nav-link {% if page=='history' %}active{% endif %}" href="/history">History</a></li>
        <li class="nav-item"><a class="nav-link {% if page=='agents' %}active{% endif %}" href="/agents">Agents</a></li>
      </ul>
      <div class="d-flex align-items-center gap-3">
        <span class="dim small" id="last-sample"></span>
        <span class="dim small" id="live-indicator"></span>
        <button class="btn btn-soft btn-sm" onclick="location.reload()">Reload</button>
      </div>
    </div>
  </div>
</nav>
<div class="container-fluid px-4 py-3">
  {% block content %}{% endblock %}
</div>
<script>
function utilColor(v) {
  if (v == null || isNaN(v)) return '#6c757d';
  if (v >= 80) return '#ef4f6a';
  if (v >= 50) return '#f5b841';
  if (v >= 15) return '#25c786';
  return '#6c8cff';
}
function fmtPct(v) { return v == null || isNaN(v) ? '—' : v.toFixed(0) + '%'; }
function fmtMem(used, total) {
  if (used == null || total == null) return '—';
  const g = 1024;
  return (used/g).toFixed(1) + ' / ' + (total/g).toFixed(0) + ' GiB';
}
function fmtPower(v) { return v == null || isNaN(v) ? '—' : v.toFixed(0) + ' W'; }
function fmtTemp(v)  { return v == null || isNaN(v) ? '—' : v.toFixed(0) + '°C'; }
function fmtTimeAgo(ts) {
  if (!ts) return '—';
  const delta = Math.floor(Date.now()/1000 - ts);
  if (delta < 60) return delta + 's ago';
  if (delta < 3600) return Math.floor(delta/60) + 'm ago';
  if (delta < 86400) return Math.floor(delta/3600) + 'h ago';
  return Math.floor(delta/86400) + 'd ago';
}
function utilClass(v) {
  if (v == null || isNaN(v)) return 'warn';
  if (v >= 80) return 'bad';
  if (v >= 50) return 'warn';
  return 'good';
}
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
{% block extra_js %}{% endblock %}
</body>
</html>
"""


OVERVIEW_PAGE = """
{% extends "base.html" %}
{% block title %}Overview — GPU Dashboard{% endblock %}
{% block content %}
<div class="row g-3 mb-4" id="kpi-row">
  <div class="col-lg-3 col-6">
    <div class="panel"><h6>Servers tracked</h6>
      <div class="kpi" id="kpi-server-count">—</div>
      <div class="kpi-sub" id="kpi-server-sub">&nbsp;</div></div>
  </div>
  <div class="col-lg-3 col-6">
    <div class="panel"><h6>Running agents</h6>
      <div class="kpi" id="kpi-agents">—</div>
      <div class="kpi-sub" id="kpi-agents-sub">&nbsp;</div></div>
  </div>
  <div class="col-lg-3 col-6">
    <div class="panel"><h6>Mean GPU util</h6>
      <div class="kpi" id="kpi-mean">—</div>
      <div class="kpi-sub">across owned GPUs</div></div>
  </div>
  <div class="col-lg-3 col-6">
    <div class="panel"><h6>Peak GPU util</h6>
      <div class="kpi" id="kpi-peak">—</div>
      <div class="kpi-sub" id="kpi-peak-sub">&nbsp;</div></div>
  </div>
</div>

<div id="cards-root" class="row g-3"></div>
<div id="empty-state" class="panel text-center py-5" style="display:none">
  <div class="dim">No servers registered yet. The sampler may still be
    starting up, or no vLLM / GPU-tool servers exist in serve.json /
    gpu_server.json.</div>
</div>

<script>
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function renderCard(c) {
  const typeLabel = c.service_type === 'vllm' ? 'vLLM' : 'gpu-tool';
  const toolsLine = (c.service_type === 'gpu_tool' && c.tools && c.tools.length)
    ? `<div class="dim small mt-1">${esc(c.tools.join(', '))}</div>` : '';
  let gpuRows = '';
  if (c.gpus && c.gpus.length) {
    gpuRows = c.gpus.map(g => {
      const u = g.util_pct == null ? 0 : g.util_pct;
      return `
        <div class="gpu-row">
          <div class="gpu-label">GPU ${g.gpu_index}</div>
          <div class="gpu-bar-wrap">
            <div class="gpu-bar" style="width:${u}%;background:${utilColor(u)};"></div>
          </div>
          <div class="gpu-util">${fmtPct(u)}</div>
          <div class="gpu-sub">${esc(g.mem_label)} · ${esc(g.power_label)} · ${esc(g.temp_label)}</div>
        </div>`;
    }).join('');
  } else {
    const wait = c.gpus_expected && c.gpus_expected.length
      ? `expected GPUs ${c.gpus_expected.join(', ')} — no sample yet`
      : `no sample yet (waiting for sampler to resolve pid ${esc(c.pid || '?')} → GPUs)`;
    gpuRows = `<div class="dim small py-2">${wait}</div>`;
  }
  const agoEl = c.last_ts
    ? `<span data-ago="${c.last_ts}">${fmtTimeAgo(c.last_ts)}</span>`
    : `<span class="dim">—</span>`;
  const serverUrl = '/servers/' + encodeURIComponent(c.server_id);
  return `
    <div class="col-xl-6" data-server="${esc(c.server_id)}">
      <div class="node-card${c.stale ? ' stale' : ''}">
        <div class="d-flex justify-content-between align-items-start">
          <div>
            <h5><a href="${serverUrl}">${esc(c.display_label)}</a></h5>
            <div class="meta">
              ${esc(c.node)} (${esc(c.ip)}) · job ${esc(c.slurm_job_id || '—')}
              ${c.pid ? '· pid ' + esc(c.pid) : ''} ·
              updated ${agoEl}
            </div>
          </div>
          <div class="text-end">
            <span class="badge badge-${esc(c.service_type)}">${typeLabel}</span>
            ${toolsLine}
          </div>
        </div>
        <div class="mt-2">${gpuRows}</div>
      </div>
    </div>`;
}

function renderSummary(s) {
  document.getElementById('kpi-server-count').textContent = s.server_count;
  document.getElementById('kpi-server-sub').textContent =
    `${s.gpu_count} GPUs on ${s.node_count} node(s)`;
  document.getElementById('kpi-agents').textContent = s.agent_total;
  const bk = s.agent_breakdown ? Object.keys(s.agent_breakdown).length : 0;
  document.getElementById('kpi-agents-sub').textContent = `across ${bk} benchmark(s)`;
  const meanEl = document.getElementById('kpi-mean');
  meanEl.textContent = s.mean_util + '%';
  meanEl.className = 'kpi hl-' + (s.mean_util_class || 'good');
  const peakEl = document.getElementById('kpi-peak');
  peakEl.textContent = s.peak_util + '%';
  peakEl.className = 'kpi hl-' + (s.peak_util_class || 'good');
  document.getElementById('kpi-peak-sub').textContent = s.peak_label || '—';
  const lastSampleEl = document.getElementById('last-sample');
  if (lastSampleEl && s.last_ts) {
    lastSampleEl.textContent = 'last sample ' + fmtTimeAgo(s.last_ts);
  }
}

const POLL_MS = 5000;
let _lastRefresh = 0;

async function refresh() {
  try {
    const r = await fetch('/api/overview', {cache: 'no-store'});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    renderSummary(data.summary);
    const root = document.getElementById('cards-root');
    const empty = document.getElementById('empty-state');
    if (!data.cards || !data.cards.length) {
      root.innerHTML = '';
      empty.style.display = '';
    } else {
      empty.style.display = 'none';
      root.innerHTML = data.cards.map(renderCard).join('');
    }
    _lastRefresh = Date.now();
    const ind = document.getElementById('live-indicator');
    if (ind) ind.textContent = '● live (every ' + (POLL_MS/1000) + 's)';
  } catch (e) {
    const ind = document.getElementById('live-indicator');
    if (ind) ind.textContent = '⚠ refresh failed — retrying';
  }
}

refresh();
setInterval(refresh, POLL_MS);
</script>
{% endblock %}
"""


HISTORY_PAGE = """
{% extends "base.html" %}
{% block title %}History — GPU Dashboard{% endblock %}
{% block content %}
<div class="d-flex flex-wrap align-items-center gap-2 mb-3">
  <div class="panel py-2 px-3" style="flex: 1; min-width: 320px;">
    <label class="dim small me-2">Server</label>
    <select class="form-select form-select-sm d-inline-block" style="width:auto"
            id="server-select" onchange="reloadChart()">
      <option value="">all servers (stacked)</option>
      {% for sid, label, stype in servers %}
        <option value="{{ sid }}">{{ stype }} · {{ label }} ({{ sid }})</option>
      {% endfor %}
    </select>
  </div>
  <div class="panel py-2 px-3">
    <label class="dim small me-2">Service</label>
    <select class="form-select form-select-sm d-inline-block" style="width:auto"
            id="service-select" onchange="reloadChart()">
      <option value="">any</option>
      <option value="vllm">vLLM only</option>
      <option value="gpu_tool">GPU tool only</option>
    </select>
  </div>
  <div class="panel py-2 px-3">
    <span class="dim small me-2">Range</span>
    <div class="btn-group btn-group-sm" role="group" id="range-group">
      <button type="button" class="btn btn-soft" data-hours="6">6h</button>
      <button type="button" class="btn btn-soft active" data-hours="24">24h</button>
      <button type="button" class="btn btn-soft" data-hours="72">3d</button>
      <button type="button" class="btn btn-soft" data-hours="168">7d</button>
      <button type="button" class="btn btn-soft" data-hours="720">30d</button>
    </div>
  </div>
</div>

<div class="chart-wrap mb-3">
  <canvas id="util-chart" height="110"></canvas>
</div>

<div class="row g-3">
  <div class="col-lg-6">
    <div class="chart-wrap"><canvas id="mem-chart" height="160"></canvas></div>
  </div>
  <div class="col-lg-6">
    <div class="chart-wrap"><canvas id="power-chart" height="160"></canvas></div>
  </div>
</div>

<script>
let currentHours = 24;

document.querySelectorAll('#range-group button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('#range-group button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    currentHours = parseInt(b.dataset.hours, 10);
    reloadChart();
  });
});

function pickBucket(hours) {
  if (hours <= 6)  return 60;
  if (hours <= 24) return 300;
  if (hours <= 72) return 900;
  if (hours <= 168) return 1800;
  return 3600;
}
function palette(i) {
  const colors = ['#6c8cff','#a07bff','#25c786','#f5b841','#ef4f6a',
                  '#38c6e0','#c9a14a','#e66ce0','#83d856','#ff8a5c'];
  return colors[i % colors.length];
}

async function reloadChart() {
  const hours = currentHours;
  const since = Math.floor(Date.now()/1000) - hours*3600;
  const bucket = pickBucket(hours);
  const serverId = document.getElementById('server-select').value;
  const service = document.getElementById('service-select').value;
  const params = new URLSearchParams({ since, bucket });
  if (serverId) params.set('server_id', serverId);
  if (service) params.set('service', service);

  const [gpuRes, agentRes] = await Promise.all([
    fetch('/api/history?' + params).then(r => r.json()),
    fetch('/api/agents/history?since=' + since + '&bucket=' + bucket).then(r => r.json()),
  ]);

  const bySrv = {};
  gpuRes.rows.forEach(r => {
    const key = r.service_id || r.node || 'unknown';
    (bySrv[key] = bySrv[key] || {util:[],mem:[],power:[]});
    bySrv[key].util.push({x: r.bucket_ts*1000, y: r.util_pct});
    bySrv[key].mem.push({x: r.bucket_ts*1000, y: (r.mem_used_mb||0)/1024});
    bySrv[key].power.push({x: r.bucket_ts*1000, y: r.power_w});
  });
  const keys = Object.keys(bySrv).sort();

  const utilDatasets = keys.map((k, i) => ({
    label: k, data: bySrv[k].util,
    borderColor: palette(i), backgroundColor: palette(i) + '22',
    borderWidth: 1.8, pointRadius: 0, tension: 0.25, fill: false, yAxisID: 'y',
  }));
  utilDatasets.push({
    label: 'running agents',
    data: agentRes.rows.map(r => ({x: r.bucket_ts*1000, y: r.total})),
    borderColor: '#ffffff', backgroundColor: 'rgba(255,255,255,.08)',
    borderWidth: 2, borderDash: [4, 4], pointRadius: 0, stepped: true,
    yAxisID: 'y2', fill: false,
  });
  renderChart('util-chart', utilDatasets, {y:'GPU util %', y2:'agents'});

  const memDatasets = keys.map((k, i) => ({
    label: k, data: bySrv[k].mem,
    borderColor: palette(i), borderWidth: 1.5, pointRadius: 0, tension: 0.25, fill: false,
  }));
  renderChart('mem-chart', memDatasets, {y: 'GPU memory used (GiB)'});

  const pwDatasets = keys.map((k, i) => ({
    label: k, data: bySrv[k].power,
    borderColor: palette(i), borderWidth: 1.5, pointRadius: 0, tension: 0.25, fill: false,
  }));
  renderChart('power-chart', pwDatasets, {y: 'Power (W)'});
}

function renderChart(canvasId, datasets, axisLabels) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  const existing = Chart.getChart(canvasId);
  if (existing) existing.destroy();
  const scales = {
    x: { type: 'time', ticks: { color: '#96a0bd' },
         grid: { color: 'rgba(255,255,255,0.04)' } },
    y: { title: { display: true, text: axisLabels.y, color: '#96a0bd' },
         ticks: { color: '#96a0bd' }, grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true },
  };
  if (axisLabels.y2) {
    scales.y2 = { position: 'right', title: { display: true, text: axisLabels.y2, color: '#96a0bd' },
                   ticks: { color: '#96a0bd' }, grid: { display: false }, beginAtZero: true };
  }
  new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {
        legend: { labels: { color: '#e8ecf5' } },
        tooltip: { mode: 'index', intersect: false },
      },
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      scales,
    },
  });
}

reloadChart();
</script>
{% endblock %}
"""


SERVER_DETAIL_PAGE = """
{% extends "base.html" %}
{% block title %}{{ server.display_label }} — GPU Dashboard{% endblock %}
{% block content %}
<div class="mb-3">
  <a href="/" class="dim small">← Overview</a>
  <h4 class="mt-1 mb-0">
    <span class="badge badge-{{ server.service_type }} me-2">
      {% if server.service_type == 'vllm' %}vLLM{% else %}gpu-tool{% endif %}
    </span>
    {{ server.display_label }}
  </h4>
  <div class="dim small">
    {{ server.server_id }} · {{ server.node }} ({{ server.ip }}) ·
    job {{ server.slurm_job_id or '—' }}{% if server.pid %} · pid {{ server.pid }}{% endif %}
    {% if server.gpus_expected %} · GPUs {{ server.gpus_expected|join(', ') }}{% endif %}
    {% if server.tools %} · tools: {{ server.tools|join(', ') }}{% endif %}
  </div>
</div>

<div class="chart-wrap mb-3">
  <canvas id="util-chart" height="110"></canvas>
</div>
<div class="row g-3">
  <div class="col-lg-6"><div class="chart-wrap"><canvas id="mem-chart" height="160"></canvas></div></div>
  <div class="col-lg-6"><div class="chart-wrap"><canvas id="power-chart" height="160"></canvas></div></div>
</div>

<script>
async function load() {
  const hours = 24;
  const since = Math.floor(Date.now()/1000) - hours*3600;
  const bucket = 300;
  const [res, agentRes] = await Promise.all([
    fetch('/api/servers/{{ server.server_id|urlencode }}/history?since=' + since + '&bucket=' + bucket).then(r => r.json()),
    fetch('/api/agents/history?since=' + since + '&bucket=' + bucket).then(r => r.json()),
  ]);
  const byGpu = {};
  res.rows.forEach(r => {
    (byGpu[r.gpu_index] = byGpu[r.gpu_index] || {util:[],mem:[],power:[]});
    byGpu[r.gpu_index].util.push({x: r.bucket_ts*1000, y: r.util_pct});
    byGpu[r.gpu_index].mem.push({x: r.bucket_ts*1000, y: (r.mem_used_mb||0)/1024});
    byGpu[r.gpu_index].power.push({x: r.bucket_ts*1000, y: r.power_w});
  });
  const colors = ['#6c8cff','#a07bff','#25c786','#f5b841','#ef4f6a',
                  '#38c6e0','#c9a14a','#e66ce0'];
  function ds(key) {
    return Object.keys(byGpu).sort((a,b)=>+a-+b).map((g, i) => ({
      label: 'GPU '+g, data: byGpu[g][key],
      borderColor: colors[i%colors.length], borderWidth: 1.6, pointRadius: 0, tension: .25,
      yAxisID: 'y',
    }));
  }
  const utilDatasets = ds('util');
  utilDatasets.push({
    label: 'running agents',
    data: agentRes.rows.map(r => ({x: r.bucket_ts*1000, y: r.total})),
    borderColor: '#ffffff', borderDash:[4,4], borderWidth: 2, pointRadius: 0,
    stepped: true, yAxisID: 'y2',
  });
  render('util-chart',  utilDatasets, {y:'GPU util %', y2:'agents'});
  render('mem-chart',   ds('mem'),   {y:'Memory (GiB)'});
  render('power-chart', ds('power'), {y:'Power (W)'});
}
function render(id, datasets, axes) {
  const old = Chart.getChart(id); if (old) old.destroy();
  const scales = {
    x: { type:'time', ticks:{color:'#96a0bd'}, grid:{color:'rgba(255,255,255,0.04)'} },
    y: { title:{display:true,text:axes.y,color:'#96a0bd'}, ticks:{color:'#96a0bd'},
         grid:{color:'rgba(255,255,255,0.04)'}, beginAtZero:true },
  };
  if (axes.y2) scales.y2 = { position:'right', title:{display:true,text:axes.y2,color:'#96a0bd'},
                              ticks:{color:'#96a0bd'}, grid:{display:false}, beginAtZero:true };
  new Chart(document.getElementById(id).getContext('2d'), {
    type:'line', data:{datasets},
    options:{ responsive:true, maintainAspectRatio:false, animation:false,
      plugins:{legend:{labels:{color:'#e8ecf5'}}, tooltip:{mode:'index',intersect:false}},
      interaction:{mode:'nearest',axis:'x',intersect:false}, scales,
    }});
}
load();
</script>
{% endblock %}
"""


AGENTS_PAGE = """
{% extends "base.html" %}
{% block title %}Agents — GPU Dashboard{% endblock %}
{% block content %}
<div class="row g-3 mb-3">
  <div class="col-lg-4 col-6">
    <div class="panel"><h6>Currently running</h6>
      <div class="kpi">{{ snap.total }}</div>
      <div class="kpi-sub">{{ snap.experiment_ids|length }} experiment process(es)</div></div>
  </div>
  <div class="col-lg-4 col-6">
    <div class="panel"><h6>Benchmarks</h6>
      {% if snap.by_benchmark %}
        {% for k, v in snap.by_benchmark.items() %}
          <div class="d-flex justify-content-between"><span>{{ k }}</span><span class="hl-good">{{ v }}</span></div>
        {% endfor %}
      {% else %}<div class="dim">—</div>{% endif %}
    </div>
  </div>
  <div class="col-lg-4 col-6">
    <div class="panel"><h6>Models</h6>
      {% if snap.by_model %}
        {% for k, v in snap.by_model.items() %}
          <div class="d-flex justify-content-between"><span>{{ k }}</span><span class="hl-good">{{ v }}</span></div>
        {% endfor %}
      {% else %}<div class="dim">—</div>{% endif %}
    </div>
  </div>
</div>

<div class="chart-wrap mb-3"><canvas id="agent-chart" height="110"></canvas></div>
<div class="chart-wrap"><canvas id="bench-chart" height="130"></canvas></div>

<script>
async function load() {
  const hours = 168;
  const since = Math.floor(Date.now()/1000) - hours*3600;
  const res = await fetch('/api/agents/breakdown?since=' + since).then(r => r.json());
  const tsList = res.rows.map(r => r.ts*1000);
  new Chart(document.getElementById('agent-chart').getContext('2d'), {
    type:'line',
    data:{datasets:[{label:'running agents',
                     data: res.rows.map(r=>({x:r.ts*1000,y:r.total})),
                     borderColor:'#25c786', backgroundColor:'rgba(37,199,134,.12)',
                     fill:true, borderWidth:2, stepped:true, pointRadius:0}]},
    options:{responsive:true, maintainAspectRatio:false, animation:false,
      scales:{
        x:{type:'time', ticks:{color:'#96a0bd'}, grid:{color:'rgba(255,255,255,0.04)'}},
        y:{beginAtZero:true, ticks:{color:'#96a0bd'}, grid:{color:'rgba(255,255,255,0.04)'}}},
      plugins:{legend:{labels:{color:'#e8ecf5'}}}}});

  // Per-benchmark stacked area
  const benches = new Set();
  res.rows.forEach(r => { if (r.by_benchmark) Object.keys(r.by_benchmark).forEach(b => benches.add(b)); });
  const bList = [...benches].sort();
  const colors = ['#6c8cff','#a07bff','#25c786','#f5b841','#ef4f6a','#38c6e0'];
  const datasets = bList.map((b, i) => ({
    label: b,
    data: res.rows.map(r => ({x: r.ts*1000, y: (r.by_benchmark || {})[b] || 0})),
    borderColor: colors[i%colors.length], backgroundColor: colors[i%colors.length] + '55',
    fill: true, stepped: true, borderWidth: 1.2, pointRadius: 0, stack: 'b',
  }));
  new Chart(document.getElementById('bench-chart').getContext('2d'), {
    type:'line', data:{datasets},
    options:{responsive:true, maintainAspectRatio:false, animation:false,
      scales:{
        x:{type:'time', ticks:{color:'#96a0bd'}, grid:{color:'rgba(255,255,255,0.04)'}},
        y:{stacked:true, beginAtZero:true, ticks:{color:'#96a0bd'}, grid:{color:'rgba(255,255,255,0.04)'}}},
      plugins:{legend:{labels:{color:'#e8ecf5'}}}}});
}
load();
</script>
{% endblock %}
"""
