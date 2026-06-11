"""Inline Jinja2 HTML templates for the visualization server."""

BASE_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{% block title %}Spatial Agent Results{% endblock %}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  body { background: #f8f9fa; }
  .navbar { background: #1a1a2e !important; }
  .navbar-brand { font-weight: 700; letter-spacing: 0.5px; }
  .card { border: none; box-shadow: 0 1px 3px rgba(0,0,0,.08); transition: box-shadow .15s; }
  .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.12); }
  .badge-method-spatial { background: #6f42c1; }
  .badge-method-cot { background: #0d6efd; }
  .badge-method-other { background: #6c757d; }
  .acc-high { color: #198754; font-weight: 700; }
  .acc-mid { color: #ffc107; font-weight: 700; }
  .acc-low { color: #dc3545; font-weight: 700; }
  .acc-bar { height: 6px; border-radius: 3px; background: #e9ecef; overflow: hidden; }
  .acc-bar-fill { height: 100%; border-radius: 3px; }
  .sample-correct { background: #d1e7dd; }
  .sample-incorrect { background: #f8d7da; }
  .sample-empty { background: #e2e3e5; }
  .search-box { max-width: 300px; }
  .breadcrumb { background: transparent; padding: 0; margin-bottom: 0.5rem; }
  .table-sortable th { cursor: pointer; user-select: none; }
  .table-sortable th:hover { background: #e9ecef; }
  .sort-arrow { font-size: 0.7em; margin-left: 4px; opacity: 0.3; }
  .sort-arrow.active { opacity: 1; }
  pre.code-block { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; font-size: 0.85em; max-height: 400px; overflow: auto; }
  .iframe-wrapper { border: 1px solid #dee2e6; border-radius: 6px; overflow: hidden; }
  .iframe-wrapper iframe { width: 100%; height: 80vh; border: none; }
  .filter-btn.active { font-weight: 700; }
  {% block extra_css %}{% endblock %}
</style>
</head>
<body>
<nav class="navbar navbar-dark mb-4">
  <div class="container-fluid">
    <a class="navbar-brand" href="/">Spatial Agent Results</a>
    <div class="d-flex align-items-center gap-3">
      <span class="text-light small" id="exp-count"></span>
      <span class="text-light small" id="auto-refresh-timer"></span>
      <button class="btn btn-outline-light btn-sm" onclick="refreshData()">Refresh</button>
    </div>
  </div>
</nav>
<div class="container-fluid px-4">
  {% block content %}{% endblock %}
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
function refreshData() {
  fetch('/api/refresh').then(r => r.json()).then(d => {
    document.getElementById('exp-count').textContent = d.count + ' experiments';
    location.reload();
  });
}
// Auto-refresh every 10 minutes
(function() {
  const INTERVAL = 10 * 60; // seconds
  let remaining = INTERVAL;
  const timerEl = document.getElementById('auto-refresh-timer');
  function updateTimer() {
    const m = Math.floor(remaining / 60);
    const s = remaining % 60;
    if (timerEl) timerEl.textContent = 'next refresh: ' + m + ':' + String(s).padStart(2, '0');
    if (remaining <= 0) {
      refreshData();
      remaining = INTERVAL;
    }
    remaining--;
  }
  updateTimer();
  setInterval(updateTimer, 1000);
})();
function accClass(v) {
  if (v >= 0.6) return 'acc-high';
  if (v >= 0.4) return 'acc-mid';
  return 'acc-low';
}
function accColor(v) {
  if (v >= 0.6) return '#198754';
  if (v >= 0.4) return '#ffc107';
  return '#dc3545';
}
function pct(v) { return (v * 100).toFixed(1) + '%'; }

// Generic table sorting
function makeSortable(tableId) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const headers = table.querySelectorAll('th[data-sort]');
  let currentSort = { col: null, asc: true };
  headers.forEach((th, i) => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      const type = th.dataset.sortType || 'string';
      const asc = currentSort.col === key ? !currentSort.asc : (type === 'number' ? false : true);
      currentSort = { col: key, asc };
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {
        let va = a.querySelector(`td[data-sort-key="${key}"]`)?.dataset.sortValue || a.cells[i]?.textContent || '';
        let vb = b.querySelector(`td[data-sort-key="${key}"]`)?.dataset.sortValue || b.cells[i]?.textContent || '';
        if (type === 'number') { va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }
        else { va = va.toLowerCase(); vb = vb.toLowerCase(); }
        if (va < vb) return asc ? -1 : 1;
        if (va > vb) return asc ? 1 : -1;
        return 0;
      });
      rows.forEach(r => tbody.appendChild(r));
      // Update arrows
      headers.forEach(h => h.querySelector('.sort-arrow')?.classList.remove('active'));
      th.querySelector('.sort-arrow')?.classList.add('active');
      th.querySelector('.sort-arrow').textContent = asc ? '▲' : '▼';
    });
  });
}

// Search filter
function setupSearch(inputId, tableId) {
  const input = document.getElementById(inputId);
  if (!input) return;
  input.addEventListener('input', () => {
    const q = input.value.toLowerCase();
    const rows = document.querySelectorAll(`#${tableId} tbody tr`);
    rows.forEach(r => {
      r.style.display = r.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
}
</script>
{% block extra_js %}{% endblock %}
</body>
</html>
"""

DASHBOARD_PAGE = """
{% extends "base.html" %}
{% block title %}Dashboard — Spatial Agent Results{% endblock %}
{% block content %}
<div class="row mb-4">
  <div class="col">
    <h4>Dashboard</h4>
    <p class="text-muted">{{ experiments | length }} experiments &middot; {{ models | length }} models &middot; {{ benchmarks | length }} benchmarks</p>
  </div>
</div>

<!-- Model cards -->
<h5 class="mb-3">Models</h5>
<div class="row g-3 mb-4">
{% for model in models %}
  <div class="col-md-4 col-lg-3">
    <div class="card h-100">
      <div class="card-body">
        <h6 class="card-title">
          <a href="/model/{{ url_encode(model.name) }}" class="text-decoration-none">{{ model.name }}</a>
        </h6>
        <div class="text-muted small mb-2">{{ model.count }} experiment{{ 's' if model.count != 1 }}</div>
        <div>
          {% for bname in model.benchmarks %}
          <span class="badge bg-secondary me-1 mb-1">{{ bname }}</span>
          {% endfor %}
        </div>
      </div>
    </div>
  </div>
{% endfor %}
</div>

<!-- Benchmark cards -->
<h5 class="mb-3">Benchmarks</h5>
<div class="row g-3 mb-4">
{% for bname, bdata in benchmarks.items() %}
  <div class="col-md-4 col-lg-3">
    <div class="card h-100">
      <div class="card-body">
        <h6 class="card-title">
          <a href="/benchmark/{{ bname }}" class="text-decoration-none">{{ bname }}</a>
        </h6>
        <div class="d-flex justify-content-between align-items-center mb-2">
          <span class="text-muted small">{{ bdata.count }} experiments</span>
          {% if bdata.best_acc is not none %}
          <span class="{{ 'acc-high' if bdata.best_acc >= 0.6 else ('acc-mid' if bdata.best_acc >= 0.4 else 'acc-low') }}">
            Best: {{ "%.1f" | format(bdata.best_acc * 100) }}%
          </span>
          {% endif %}
        </div>
        {% if bdata.best_acc is not none %}
        <div class="acc-bar">
          <div class="acc-bar-fill" style="width:{{ "%.1f" | format(bdata.best_acc * 100) }}%;background:{{ '#198754' if bdata.best_acc >= 0.6 else ('#ffc107' if bdata.best_acc >= 0.4 else '#dc3545') }}"></div>
        </div>
        <div class="text-muted small mt-1">{{ bdata.best_exp }}</div>
        {% endif %}
      </div>
    </div>
  </div>
{% endfor %}
</div>

<!-- Leaderboard table -->
<div class="card mb-4">
  <div class="card-body">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <h5 class="mb-0">All Experiments</h5>
      <input type="text" id="search-input" class="form-control form-control-sm search-box" placeholder="Search...">
    </div>
    <div class="table-responsive">
      <table class="table table-hover table-sortable" id="leaderboard">
        <thead>
          <tr>
            <th data-sort="name" data-sort-type="string">Experiment <span class="sort-arrow">▲</span></th>
            <th data-sort="benchmark" data-sort-type="string">Benchmark <span class="sort-arrow">▲</span></th>
            <th data-sort="method" data-sort-type="string">Method <span class="sort-arrow">▲</span></th>
            <th data-sort="model" data-sort-type="string">Model <span class="sort-arrow">▲</span></th>
            <th data-sort="accuracy" data-sort-type="number">Accuracy <span class="sort-arrow">▼</span></th>
            <th data-sort="samples" data-sort-type="number">Samples <span class="sort-arrow">▼</span></th>
            <th data-sort="tools" data-sort-type="string">Tools <span class="sort-arrow">▲</span></th>
            <th data-sort="date" data-sort-type="string">Date <span class="sort-arrow">▼</span></th>
          </tr>
        </thead>
        <tbody>
        {% for exp in experiments %}
          <tr>
            <td data-sort-key="name" data-sort-value="{{ exp.dir_name }}">
              <a href="/experiment/{{ exp.dir_name }}">{{ exp.dir_name }}</a>
            </td>
            <td data-sort-key="benchmark" data-sort-value="{{ exp.benchmark }}">{{ exp.benchmark }}</td>
            <td data-sort-key="method" data-sort-value="{{ exp.method }}">
              <span class="badge badge-method-{{ exp.method }}">{{ method_label(exp.method) }}</span>
            </td>
            <td data-sort-key="model" data-sort-value="{{ exp.model }}">{{ exp.model }}</td>
            <td data-sort-key="accuracy" data-sort-value="{{ exp.accuracy if exp.accuracy is not none else -1 }}">
              {% if exp.accuracy is not none %}
              <span class="{{ 'acc-high' if exp.accuracy >= 0.6 else ('acc-mid' if exp.accuracy >= 0.4 else 'acc-low') }}">
                {{ "%.1f" | format(exp.accuracy * 100) }}%
              </span>
              {% else %}
              <span class="text-muted">—</span>
              {% endif %}
            </td>
            <td data-sort-key="samples" data-sort-value="{{ exp.num_predictions }}">{{ exp.num_predictions }}</td>
            <td data-sort-key="tools" data-sort-value="{{ exp.tools | join(', ') }}">
              {% for t in exp.tools %}
              <span class="badge bg-secondary">{{ t }}</span>
              {% endfor %}
            </td>
            <td data-sort-key="date" data-sort-value="{{ exp.created_date }}">
              <span class="text-muted small">{{ exp.created_date }}</span>
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>
{% endblock %}
{% block extra_js %}
<script>
  makeSortable('leaderboard');
  setupSearch('search-input', 'leaderboard');
  document.getElementById('exp-count').textContent = '{{ experiments | length }} experiments';
</script>
{% endblock %}
"""

MODELS_PAGE = """
{% extends "base.html" %}
{% block title %}Models — Spatial Agent Results{% endblock %}
{% block content %}
<nav aria-label="breadcrumb">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="/">Dashboard</a></li>
    <li class="breadcrumb-item active">Models</li>
  </ol>
</nav>

<div class="row mb-4">
  <div class="col">
    <h4>Models</h4>
    <p class="text-muted">{{ models | length }} models across {{ total_experiments }} experiments</p>
  </div>
  <div class="col-auto">
    <input type="text" id="model-search" class="form-control form-control-sm search-box" placeholder="Search models...">
  </div>
</div>

<div class="row g-3 mb-4" id="model-grid">
{% for model in models %}
  <div class="col-md-4 col-lg-3 model-card-wrapper">
    <div class="card h-100">
      <div class="card-body">
        <h6 class="card-title">
          <a href="/model/{{ url_encode(model.name) }}" class="text-decoration-none">{{ model.name }}</a>
        </h6>
        <div class="text-muted small mb-2">{{ model.count }} experiment{{ 's' if model.count != 1 }}</div>
        <div>
          {% for bname in model.benchmarks %}
          <span class="badge bg-secondary me-1 mb-1">{{ bname }}</span>
          {% endfor %}
        </div>
      </div>
    </div>
  </div>
{% endfor %}
</div>
{% endblock %}
{% block extra_js %}
<script>
  document.getElementById('model-search')?.addEventListener('input', function() {
    const q = this.value.toLowerCase();
    document.querySelectorAll('.model-card-wrapper').forEach(c => {
      c.style.display = c.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
</script>
{% endblock %}
"""

MODEL_DETAIL_PAGE = """
{% extends "base.html" %}
{% block title %}{{ model_name }} — Spatial Agent Results{% endblock %}
{% block content %}
<nav aria-label="breadcrumb">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="/">Dashboard</a></li>
    <li class="breadcrumb-item"><a href="/models">Models</a></li>
    <li class="breadcrumb-item active">{{ model_name }}</li>
  </ol>
</nav>

<h4 class="mb-1">{{ model_name }}</h4>
<p class="text-muted">{{ experiments | length }} experiment{{ 's' if experiments | length != 1 }} across {{ benchmark_groups | length }} benchmark{{ 's' if benchmark_groups | length != 1 }}</p>

{% if chart_data | length > 0 %}
<div class="card mb-4">
  <div class="card-body">
    <h5 class="mb-3">Best Accuracy per Benchmark</h5>
    <canvas id="modelChart" height="80"></canvas>
  </div>
</div>
{% endif %}

{% for group in benchmark_groups %}
<div class="card mb-4">
  <div class="card-body">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <h5 class="mb-0">
        <a href="/benchmark/{{ group.name }}" class="text-decoration-none">{{ group.name }}</a>
        <span class="badge bg-secondary ms-2">{{ group.experiments | length }}</span>
      </h5>
    </div>
    <div class="table-responsive">
      <table class="table table-hover table-sortable" id="table-{{ loop.index }}">
        <thead>
          <tr>
            <th data-sort="name" data-sort-type="string">Experiment <span class="sort-arrow">▲</span></th>
            <th data-sort="method" data-sort-type="string">Method <span class="sort-arrow">▲</span></th>
            <th data-sort="accuracy" data-sort-type="number">Accuracy <span class="sort-arrow">▼</span></th>
            <th data-sort="correct" data-sort-type="number">Correct <span class="sort-arrow">▼</span></th>
            <th data-sort="total" data-sort-type="number">Total <span class="sort-arrow">▼</span></th>
            <th data-sort="avg_tokens" data-sort-type="number">Avg Tokens / Session <span class="sort-arrow">▼</span></th>
            <th data-sort="avg_calls" data-sort-type="number">Avg Calls / Session <span class="sort-arrow">▼</span></th>
            <th data-sort="tools" data-sort-type="string">Tools <span class="sort-arrow">▲</span></th>
            <th data-sort="date" data-sort-type="string">Date <span class="sort-arrow">▼</span></th>
          </tr>
        </thead>
        <tbody>
        {% for exp in group.experiments %}
          {% set tu = token_usage_by_exp.get(exp.dir_name) or {} %}
          <tr>
            <td data-sort-key="name" data-sort-value="{{ exp.dir_name }}">
              <a href="/experiment/{{ exp.dir_name }}">{{ exp.dir_name }}</a>
            </td>
            <td data-sort-key="method" data-sort-value="{{ exp.method }}">
              <span class="badge badge-method-{{ exp.method }}">{{ method_label(exp.method) }}</span>
            </td>
            <td data-sort-key="accuracy" data-sort-value="{{ exp.accuracy if exp.accuracy is not none else -1 }}">
              {% if exp.accuracy is not none %}
              <span class="{{ 'acc-high' if exp.accuracy >= 0.6 else ('acc-mid' if exp.accuracy >= 0.4 else 'acc-low') }}">
                {{ "%.1f" | format(exp.accuracy * 100) }}%
              </span>
              {% else %}<span class="text-muted">—</span>{% endif %}
            </td>
            <td data-sort-key="correct" data-sort-value="{{ exp.correct_samples or 0 }}">{{ exp.correct_samples or '—' }}</td>
            <td data-sort-key="total" data-sort-value="{{ exp.num_predictions }}">{{ exp.num_predictions }}</td>
            <td data-sort-key="avg_tokens" data-sort-value="{{ tu.avg_total_per_session if tu else -1 }}">
              {% if tu %}{{ tu.avg_total_per_session | fmt_int }}{% else %}<span class="text-muted">—</span>{% endif %}
            </td>
            <td data-sort-key="avg_calls" data-sort-value="{{ tu.avg_calls_per_session if tu else -1 }}">
              {% if tu %}{{ "%.1f" | format(tu.avg_calls_per_session) }}{% else %}<span class="text-muted">—</span>{% endif %}
            </td>
            <td data-sort-key="tools" data-sort-value="{{ exp.tools | join(', ') }}">
              {% for t in exp.tools %}<span class="badge bg-secondary">{{ t }}</span> {% endfor %}
            </td>
            <td data-sort-key="date" data-sort-value="{{ exp.created_date }}">
              <span class="text-muted small">{{ exp.created_date }}</span>
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>
{% endfor %}
{% endblock %}
{% block extra_js %}
<script>
  {% for group in benchmark_groups %}
  makeSortable('table-{{ loop.index }}');
  {% endfor %}

  {% if chart_data | length > 0 %}
  const chartData = {{ chart_data | tojson }};
  new Chart(document.getElementById('modelChart'), {
    type: 'bar',
    data: {
      labels: chartData.map(d => d.benchmark),
      datasets: [{
        label: 'Best Accuracy',
        data: chartData.map(d => d.accuracy * 100),
        backgroundColor: chartData.map(d => accColor(d.accuracy)),
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: chartData.length > 8 ? 'y' : 'x',
      scales: { [chartData.length > 8 ? 'x' : 'y']: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' } } },
      plugins: { legend: { display: false } }
    }
  });
  {% endif %}
</script>
{% endblock %}
"""

BENCHMARK_PAGE = """
{% extends "base.html" %}
{% block title %}{{ benchmark_name }} — Spatial Agent Results{% endblock %}
{% block content %}
<nav aria-label="breadcrumb">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="/">Dashboard</a></li>
    <li class="breadcrumb-item active">{{ benchmark_name }}</li>
  </ol>
</nav>

<h4 class="mb-3">{{ benchmark_name }}</h4>
<p class="text-muted">{{ experiments | length }} experiments</p>

<!-- Comparison chart -->
<div class="card mb-4">
  <div class="card-body">
    <canvas id="compChart" height="80"></canvas>
  </div>
</div>

<!-- Experiments table -->
<div class="card mb-4">
  <div class="card-body">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <h5 class="mb-0">Experiments</h5>
      <div class="d-flex gap-2">
        <input type="text" id="search-input" class="form-control form-control-sm search-box" placeholder="Search...">
        <button class="btn btn-sm btn-outline-primary" id="compare-btn" disabled onclick="goCompare()">Compare Selected</button>
      </div>
    </div>
    <div class="table-responsive">
      <table class="table table-hover table-sortable" id="exp-table">
        <thead>
          <tr>
            <th style="width:30px"><input type="checkbox" id="select-all"></th>
            <th data-sort="name" data-sort-type="string">Experiment <span class="sort-arrow">▲</span></th>
            <th data-sort="method" data-sort-type="string">Method <span class="sort-arrow">▲</span></th>
            <th data-sort="model" data-sort-type="string">Model <span class="sort-arrow">▲</span></th>
            <th data-sort="accuracy" data-sort-type="number">Accuracy <span class="sort-arrow">▼</span></th>
            <th data-sort="correct" data-sort-type="number">Correct <span class="sort-arrow">▼</span></th>
            <th data-sort="total" data-sort-type="number">Total <span class="sort-arrow">▼</span></th>
            <th data-sort="avg_tokens" data-sort-type="number">Avg Tokens / Session <span class="sort-arrow">▼</span></th>
            <th data-sort="avg_calls" data-sort-type="number">Avg Calls / Session <span class="sort-arrow">▼</span></th>
            <th data-sort="tools" data-sort-type="string">Tools <span class="sort-arrow">▲</span></th>
            <th data-sort="date" data-sort-type="string">Date <span class="sort-arrow">▼</span></th>
          </tr>
        </thead>
        <tbody>
        {% for exp in experiments %}
          {% set tu = token_usage_by_exp.get(exp.dir_name) or {} %}
          <tr>
            <td><input type="checkbox" class="exp-check" value="{{ exp.dir_name }}"></td>
            <td data-sort-key="name" data-sort-value="{{ exp.dir_name }}">
              <a href="/experiment/{{ exp.dir_name }}">{{ exp.dir_name }}</a>
            </td>
            <td data-sort-key="method" data-sort-value="{{ exp.method }}">
              <span class="badge badge-method-{{ exp.method }}">{{ method_label(exp.method) }}</span>
            </td>
            <td data-sort-key="model" data-sort-value="{{ exp.model }}">{{ exp.model }}</td>
            <td data-sort-key="accuracy" data-sort-value="{{ exp.accuracy if exp.accuracy is not none else -1 }}">
              {% if exp.accuracy is not none %}
              <span class="{{ 'acc-high' if exp.accuracy >= 0.6 else ('acc-mid' if exp.accuracy >= 0.4 else 'acc-low') }}">
                {{ "%.1f" | format(exp.accuracy * 100) }}%
              </span>
              {% else %}—{% endif %}
            </td>
            <td data-sort-key="correct" data-sort-value="{{ exp.correct_samples or 0 }}">{{ exp.correct_samples or '—' }}</td>
            <td data-sort-key="total" data-sort-value="{{ exp.num_predictions }}">{{ exp.num_predictions }}</td>
            <td data-sort-key="avg_tokens" data-sort-value="{{ tu.avg_total_per_session if tu else -1 }}">
              {% if tu %}{{ tu.avg_total_per_session | fmt_int }}{% else %}<span class="text-muted">—</span>{% endif %}
            </td>
            <td data-sort-key="avg_calls" data-sort-value="{{ tu.avg_calls_per_session if tu else -1 }}">
              {% if tu %}{{ "%.1f" | format(tu.avg_calls_per_session) }}{% else %}<span class="text-muted">—</span>{% endif %}
            </td>
            <td data-sort-key="tools" data-sort-value="{{ exp.tools | join(', ') }}">
              {% for t in exp.tools %}<span class="badge bg-secondary">{{ t }}</span> {% endfor %}
            </td>
            <td data-sort-key="date" data-sort-value="{{ exp.created_date }}">
              <span class="text-muted small">{{ exp.created_date }}</span>
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- Per-category breakdown (if results exist) -->
{% if category_data %}
<div class="card mb-4">
  <div class="card-body">
    <h5 class="mb-3">Per-Category Accuracy</h5>
    <canvas id="catChart" height="100"></canvas>
  </div>
</div>
{% endif %}
{% endblock %}

{% block extra_js %}
<script>
  makeSortable('exp-table');
  setupSearch('search-input', 'exp-table');

  // Comparison chart
  const expData = {{ chart_data | tojson }};
  if (expData.length > 0) {
    new Chart(document.getElementById('compChart'), {
      type: 'bar',
      data: {
        labels: expData.map(e => e.name),
        datasets: [{
          label: 'Accuracy',
          data: expData.map(e => e.accuracy * 100),
          backgroundColor: expData.map(e => accColor(e.accuracy)),
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: expData.length > 8 ? 'y' : 'x',
        scales: { [expData.length > 8 ? 'x' : 'y']: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' } } },
        plugins: { legend: { display: false } }
      }
    });
  }

  // Category chart
  {% if category_data %}
  const catData = {{ category_data | tojson }};
  const catLabels = Object.keys(catData[0].categories);
  new Chart(document.getElementById('catChart'), {
    type: 'bar',
    data: {
      labels: catLabels,
      datasets: catData.map((exp, i) => ({
        label: exp.name,
        data: catLabels.map(c => (exp.categories[c]?.accuracy || 0) * 100),
        borderRadius: 2,
      }))
    },
    options: {
      scales: { y: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' } } },
      plugins: { legend: { position: 'bottom' } }
    }
  });
  {% endif %}

  // Select-all and compare
  document.getElementById('select-all')?.addEventListener('change', e => {
    document.querySelectorAll('.exp-check').forEach(c => c.checked = e.target.checked);
    updateCompareBtn();
  });
  document.querySelectorAll('.exp-check').forEach(c => c.addEventListener('change', updateCompareBtn));
  function updateCompareBtn() {
    const checked = document.querySelectorAll('.exp-check:checked');
    document.getElementById('compare-btn').disabled = checked.length < 2;
  }
  function goCompare() {
    const exps = Array.from(document.querySelectorAll('.exp-check:checked')).map(c => c.value);
    window.location.href = '/compare?' + exps.map(e => 'exp=' + encodeURIComponent(e)).join('&');
  }
</script>
{% endblock %}
"""

EXPERIMENT_PAGE = """
{% extends "base.html" %}
{% block title %}{{ exp.dir_name }} — Spatial Agent Results{% endblock %}
{% block extra_css %}
.sample-row { cursor: pointer; }
.sample-row:hover { background: #e9ecef !important; }
.cot-detail-row td { padding: 0 !important; border-top: none !important; }
.cot-detail-row .cot-content { max-height: 400px; overflow-y: auto; padding: 12px 16px; background: #f8f9fa; border-top: 1px dashed #dee2e6; white-space: pre-wrap; font-size: 0.85em; font-family: monospace; line-height: 1.5; }
.cot-toggle { font-size: 0.75em; padding: 1px 5px; vertical-align: middle; }
.pred-cell { max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.badge-mra { font-weight: 600; font-size: 0.8em; }
.sample-numerical { background: #e8f0fe; }
.config-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
.config-item { padding: 8px 12px; background: #f1f3f5; border-radius: 6px; }
.config-item .label { font-size: 0.78em; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
.config-item .value { font-weight: 600; }
.pagination-sm .page-link { padding: 0.25rem 0.5rem; font-size: 0.85rem; }
{% endblock %}
{% block content %}
<nav aria-label="breadcrumb">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="/">Dashboard</a></li>
    <li class="breadcrumb-item"><a href="/benchmark/{{ exp.benchmark }}">{{ exp.benchmark }}</a></li>
    <li class="breadcrumb-item active">{{ exp.dir_name }}</li>
  </ol>
</nav>

<div class="d-flex justify-content-between align-items-start mb-3">
  <div>
    <h4>{{ exp.dir_name }}</h4>
    <span class="badge badge-method-{{ exp.method }} me-2">{{ exp.method }}</span>
    <span class="text-muted">{{ exp.model }} &middot; {{ exp.benchmark }}</span>
  </div>
  {% if exp.accuracy is not none %}
  <div class="text-end">
    <div class="fs-2 {{ 'acc-high' if exp.accuracy >= 0.6 else ('acc-mid' if exp.accuracy >= 0.4 else 'acc-low') }}">
      {{ "%.1f" | format(exp.accuracy * 100) }}%
    </div>
    <div class="text-muted small">{{ exp.correct_samples }}/{{ exp.total_samples }} correct</div>
  </div>
  {% endif %}
</div>

<!-- Config summary -->
<div class="card mb-4">
  <div class="card-body">
    <h6 class="mb-3">Configuration</h6>
    <div class="config-grid">
      <div class="config-item"><div class="label">Model</div><div class="value">{{ exp.model }}</div></div>
      <div class="config-item"><div class="label">Benchmark</div><div class="value">{{ exp.benchmark }}</div></div>
      <div class="config-item"><div class="label">Method</div><div class="value">{{ method_label(exp.method) }}</div></div>
      <div class="config-item"><div class="label">Tools</div><div class="value">{{ exp.tools | join(', ') or 'None' }}</div></div>
      <div class="config-item"><div class="label">Max Steps</div><div class="value">{{ exp.config.get('max_steps', '—') }}</div></div>
      <div class="config-item"><div class="label">Concurrency</div><div class="value">{{ exp.config.get('concurrency', '—') }}</div></div>
      <div class="config-item"><div class="label">Timeout</div><div class="value">{{ exp.config.get('timeout_sec', '—') }}s</div></div>
      <div class="config-item"><div class="label">Temperature</div><div class="value">{{ (exp.config.get('main_params') or {}).get('temperature', (exp.config.get('general_params') or {}).get('temperature', '—')) }}</div></div>
    </div>
    <details class="mt-3">
      <summary class="text-muted small">Full config JSON</summary>
      <pre class="code-block mt-2">{{ config_json }}</pre>
    </details>
  </div>
</div>

{% if token_usage %}
<!-- Token usage (aggregated across sessions with logged usage) -->
<div class="card mb-4">
  <div class="card-body">
    <h6 class="mb-3">Token Usage <span class="text-muted small">({{ token_usage.sessions }} session{{ '' if token_usage.sessions == 1 else 's' }} with logged usage)</span></h6>
    <div class="config-grid">
      <div class="config-item"><div class="label">Total Tokens</div><div class="value">{{ token_usage.total_tokens | fmt_int }}</div></div>
      <div class="config-item"><div class="label">Prompt</div><div class="value">{{ token_usage.total_prompt_tokens | fmt_int }}</div></div>
      <div class="config-item"><div class="label">Completion</div><div class="value">{{ token_usage.total_completion_tokens | fmt_int }}</div></div>
      {% if token_usage.total_reasoning_tokens %}
      <div class="config-item"><div class="label">Reasoning (of completion)</div><div class="value">{{ token_usage.total_reasoning_tokens | fmt_int }}</div></div>
      {% endif %}
      <div class="config-item"><div class="label">LLM Calls</div><div class="value">{{ token_usage.num_calls | fmt_int }}</div></div>
      <div class="config-item"><div class="label">Avg Tokens / Session</div><div class="value">{{ token_usage.avg_total_per_session | fmt_int }}</div></div>
      <div class="config-item"><div class="label">Avg Calls / Session</div><div class="value">{{ "%.1f" | format(token_usage.avg_calls_per_session) }}</div></div>
      <div class="config-item"><div class="label">Max Prompt (single call)</div><div class="value">{{ token_usage.max_prompt_tokens | fmt_int }}</div></div>
      <div class="config-item"><div class="label">Max Completion (single call)</div><div class="value">{{ token_usage.max_completion_tokens | fmt_int }}</div></div>
    </div>
  </div>
</div>
{% endif %}

<!-- Results breakdowns (per-category, per-dataset, per-type, etc.) -->
{% for bd in breakdowns %}
<div class="card mb-4">
  <div class="card-body">
    <h6 class="mb-3">{{ bd.title }}</h6>
    <div class="table-responsive">
      <table class="table table-sm">
        <thead>
          <tr>
            <th>Name</th>
            {% if bd.rows[0].correct is not none %}<th>Correct</th><th>Total</th>{% endif %}
            <th>Accuracy</th>
          </tr>
        </thead>
        <tbody>
        {% for row in bd.rows %}
          {% set raw_cat = row.name | replace(' (MRA)', '') | replace(' (unscored)', '') %}
        <tr>
          <td><a href="#category={{ raw_cat }}" class="breakdown-cat-link text-decoration-none" data-category="{{ raw_cat }}">{{ row.name }}</a></td>
          {% if row.correct is not none %}<td>{{ row.correct }}</td><td>{{ row.total }}</td>{% endif %}
          <td>
            <div class="d-flex align-items-center gap-2">
            {% if row.accuracy is not none %}
              <div class="acc-bar" style="width:100px">
                <div class="acc-bar-fill" style="width:{{ "%.1f" | format(row.accuracy * 100) }}%;background:{{ '#198754' if row.accuracy >= 0.6 else ('#ffc107' if row.accuracy >= 0.4 else '#dc3545') }}"></div>
              </div>
              <span class="{{ 'acc-high' if row.accuracy >= 0.6 else ('acc-mid' if row.accuracy >= 0.4 else 'acc-low') }}">
                {{ "%.1f" | format(row.accuracy * 100) }}%
              </span>
            {% else %}
              <span class="text-muted">N/A</span>
            {% endif %}
            </div>
          </td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>
{% endfor %}

<!-- Sample browser -->
<div class="card mb-4">
  <div class="card-body">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <h6 class="mb-0">Samples ({{ predictions | length }})</h6>
      <div class="d-flex gap-2 align-items-center flex-wrap">
        <div class="btn-group btn-group-sm" role="group">
          <button class="btn btn-outline-secondary filter-btn active" data-filter="all">All</button>
          <button class="btn btn-outline-success filter-btn" data-filter="correct">Correct</button>
          <button class="btn btn-outline-danger filter-btn" data-filter="incorrect">Incorrect</button>
          <button class="btn btn-outline-primary filter-btn" data-filter="numerical">Numerical</button>
          <button class="btn btn-outline-secondary filter-btn" data-filter="empty">Empty</button>
        </div>
        {% set sample_categories = predictions | map(attribute='category') | reject('equalto', '') | unique | sort | list %}
        {% if sample_categories %}
        <select id="category-filter" class="form-select form-select-sm" style="width:auto" title="Filter by category">
          <option value="all">All categories</option>
          {% for cat in sample_categories %}
            <option value="{{ cat }}">{{ cat }}</option>
          {% endfor %}
        </select>
        {% endif %}
        <input type="text" id="sample-search" class="form-control form-control-sm search-box" placeholder="Search sample ID...">
      </div>
    </div>
    <div class="table-responsive">
      <table class="table table-sm" id="samples-table">
        <thead>
          <tr>
            <th style="width:50px">#</th>
            <th>Sample ID</th>
            <th style="max-width:150px">Prediction</th>
            <th style="width:100px">Ground Truth</th>
            <th style="width:90px">Result</th>
            <th style="width:70px">Session</th>
          </tr>
        </thead>
        <tbody>
        {% for p in predictions %}
          {% if p.is_numerical %}
            {% set row_class = 'sample-numerical' %}
            {% set row_status = 'numerical' %}
          {% elif p.get('is_unscoreable', false) %}
            {% set row_class = 'sample-numerical' %}
            {% set row_status = 'unscored' %}
          {% elif p.is_correct %}
            {% set row_class = 'sample-correct' %}
            {% set row_status = 'correct' %}
          {% elif p.is_empty %}
            {% set row_class = 'sample-empty' %}
            {% set row_status = 'empty' %}
          {% else %}
            {% set row_class = 'sample-incorrect' %}
            {% set row_status = 'incorrect' %}
          {% endif %}
          <tr class="sample-row {{ row_class }}"
              data-status="{{ row_status }}"
              data-category="{{ p.category }}"
              data-idx="{{ loop.index0 }}"
              data-sample-id="{{ p.sample_id }}"
              data-has-cot="{{ 'true' if p.has_cot else 'false' }}">
            <td>{{ loop.index }}</td>
            <td><strong>{{ p.sample_id }}</strong></td>
            <td class="pred-cell" title="{{ p.extracted_pred or '' }}">
              {{ p.extracted_pred or '(empty)' }}
              {% if p.has_cot %}
                <button class="btn btn-outline-secondary cot-toggle" onclick="event.stopPropagation(); toggleCot(this.closest('tr'))" title="Show/hide full response">...</button>
              {% endif %}
            </td>
            <td>{{ p.extracted_gt }}</td>
            <td>
              {% if p.is_numerical %}
                {% set mra = p.mra_score %}
                <span class="badge badge-mra" style="background:{{ '#198754' if mra >= 0.6 else ('#ffc107' if mra >= 0.3 else '#dc3545') }};color:{{ '#000' if mra >= 0.3 and mra < 0.6 else '#fff' }}"
                      title="MRA ({{ p.category }})">
                  MRA {{ "%.0f" | format(mra * 100) }}%
                </span>
              {% elif p.get('is_unscoreable', false) %}
                <span class="badge bg-warning text-dark" title="{{ p.category }}">Unscored</span>
              {% elif p.is_correct %}
                <span class="badge bg-success">Correct</span>
              {% elif p.is_empty %}
                <span class="badge bg-secondary">Empty</span>
              {% else %}
                <span class="badge bg-danger">Incorrect</span>
              {% endif %}
            </td>
            <td>
              {% if p.has_session %}
                <a href="/experiment/{{ exp.dir_name }}/sample/{{ p.sample_id }}" class="badge bg-info text-decoration-none" onclick="event.stopPropagation()">View</a>
              {% else %}
                <span class="text-muted">—</span>
              {% endif %}
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
  const EXP_DIR = '{{ exp.dir_name }}';

  // Lazy-load CoT content and toggle the detail row
  function toggleCot(sampleRow) {
    const idx = sampleRow.dataset.idx;
    const sampleId = sampleRow.dataset.sampleId;
    let detailRow = document.getElementById('cot-row-' + idx);

    if (detailRow) {
      // Already created — just toggle visibility
      detailRow.style.display = detailRow.style.display === 'none' ? '' : 'none';
      return;
    }

    // Create the detail row and fetch content
    detailRow = document.createElement('tr');
    detailRow.className = 'cot-detail-row';
    detailRow.id = 'cot-row-' + idx;
    detailRow.dataset.status = sampleRow.dataset.status;
    detailRow.innerHTML = '<td colspan="6"><div class="cot-content" style="color:#6c757d;font-style:italic">Loading...</div></td>';
    sampleRow.after(detailRow);

    fetch('/api/experiment/' + encodeURIComponent(EXP_DIR) + '/prediction/' + encodeURIComponent(sampleId))
      .then(r => r.json())
      .then(data => {
        const div = detailRow.querySelector('.cot-content');
        div.style.color = '';
        div.style.fontStyle = '';
        div.textContent = data.content || '(empty)';
      })
      .catch(() => {
        detailRow.querySelector('.cot-content').textContent = '(failed to load)';
      });
  }

  // Click on sample row: toggle CoT if available
  document.querySelectorAll('.sample-row').forEach(row => {
    row.addEventListener('click', () => {
      if (row.dataset.hasCot === 'true') toggleCot(row);
    });
  });

  // Unified filter state: status (button group) + category (dropdown) + search
  let currentStatus = 'all';
  let currentCategory = 'all';
  let currentSearch = '';

  function applyFilters() {
    document.querySelectorAll('.cot-detail-row').forEach(r => r.style.display = 'none');
    document.querySelectorAll('#samples-table tbody tr.sample-row').forEach(row => {
      const statusOk = currentStatus === 'all' || row.dataset.status === currentStatus;
      const catOk = currentCategory === 'all' || row.dataset.category === currentCategory;
      const searchOk = !currentSearch || row.textContent.toLowerCase().includes(currentSearch);
      row.style.display = (statusOk && catOk && searchOk) ? '' : 'none';
    });
  }

  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentStatus = btn.dataset.filter;
      applyFilters();
    });
  });

  const categoryFilter = document.getElementById('category-filter');
  categoryFilter?.addEventListener('change', e => {
    currentCategory = e.target.value;
    applyFilters();
  });

  document.getElementById('sample-search')?.addEventListener('input', e => {
    currentSearch = e.target.value.toLowerCase();
    applyFilters();
  });

  function selectCategoryFilter(target) {
    if (!categoryFilter) return false;
    for (const opt of categoryFilter.options) {
      if (opt.value === target) {
        categoryFilter.value = target;
        currentCategory = target;
        applyFilters();
        return true;
      }
    }
    return false;
  }

  // Deep-link: #category=<name> pre-selects the category filter on load
  (function () {
    const m = /[#&]category=([^&]+)/.exec(location.hash);
    if (m) selectCategoryFilter(decodeURIComponent(m[1]));
  })();

  // Breakdown table category links -> filter sample browser and scroll to it
  document.querySelectorAll('.breakdown-cat-link').forEach(a => {
    a.addEventListener('click', e => {
      const target = a.dataset.category;
      if (selectCategoryFilter(target)) {
        e.preventDefault();
        document.getElementById('samples-table')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
</script>
{% endblock %}
"""

SAMPLE_DETAIL_PAGE = """
{% extends "base.html" %}
{% block title %}Sample {{ sample_id }} — {{ exp.dir_name }}{% endblock %}
{% block extra_css %}
.nav-sample { font-size: 0.9em; }
.token-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; }
.token-item { padding: 6px 10px; background: #f1f3f5; border-radius: 6px; }
.token-item .label { font-size: 0.72em; color: #6c757d; text-transform: uppercase; letter-spacing: 0.4px; }
.token-item .value { font-weight: 600; font-size: 0.95em; }
{% endblock %}
{% block content %}
<nav aria-label="breadcrumb">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="/">Dashboard</a></li>
    <li class="breadcrumb-item"><a href="/benchmark/{{ exp.benchmark }}">{{ exp.benchmark }}</a></li>
    <li class="breadcrumb-item"><a href="/experiment/{{ exp.dir_name }}">{{ exp.dir_name }}</a></li>
    <li class="breadcrumb-item active">Sample {{ sample_id }}</li>
  </ol>
</nav>

<div class="d-flex justify-content-between align-items-center mb-3">
  <h4>Sample {{ sample_id }}</h4>
  <div class="d-flex gap-2 nav-sample">
    {% if prev_id is not none %}
      <a href="/experiment/{{ exp.dir_name }}/sample/{{ prev_id }}" class="btn btn-outline-secondary btn-sm">&larr; {{ prev_id }}</a>
    {% endif %}
    {% if next_id is not none %}
      <a href="/experiment/{{ exp.dir_name }}/sample/{{ next_id }}" class="btn btn-outline-secondary btn-sm">{{ next_id }} &rarr;</a>
    {% endif %}
  </div>
</div>

<!-- Prediction info -->
{% if prediction %}
<div class="card mb-3">
  <div class="card-body d-flex justify-content-between align-items-center">
    <div>
      <strong>Prediction:</strong> {{ prediction.extracted_pred or '(empty)' }}
      &nbsp;&middot;&nbsp;
      <strong>Ground Truth:</strong> {{ prediction.extracted_gt }}
      {% if prediction.content and prediction.content != prediction.extracted_pred %}
        <br><span class="text-muted small">Raw: {{ prediction.content[:200] }}{{ '...' if prediction.content|length > 200 else '' }}</span>
      {% endif %}
    </div>
    {% if prediction.is_numerical and prediction.mra_score is not none %}
      <span class="badge fs-6" style="background:{{ '#198754' if prediction.mra_score >= 0.6 else ('#ffc107' if prediction.mra_score >= 0.3 else '#dc3545') }};color:{{ '#000' if prediction.mra_score >= 0.3 and prediction.mra_score < 0.6 else '#fff' }}">
        MRA {{ "%.0f" | format(prediction.mra_score * 100) }}%
      </span>
    {% elif prediction.is_correct %}
      <span class="badge bg-success fs-6">Correct</span>
    {% elif prediction.is_empty %}
      <span class="badge bg-secondary fs-6">Empty</span>
    {% else %}
      <span class="badge bg-danger fs-6">Incorrect</span>
    {% endif %}
  </div>
</div>
{% endif %}

{% if token_usage %}
<!-- Token usage for this session -->
<div class="card mb-3">
  <div class="card-body">
    <h6 class="mb-2">Token Usage</h6>
    <div class="token-grid">
      <div class="token-item"><div class="label">Total</div><div class="value">{{ ((token_usage.total_prompt_tokens or 0) + (token_usage.total_completion_tokens or 0)) | fmt_int }}</div></div>
      <div class="token-item"><div class="label">Prompt</div><div class="value">{{ token_usage.total_prompt_tokens | fmt_int }}</div></div>
      <div class="token-item"><div class="label">Completion</div><div class="value">{{ token_usage.total_completion_tokens | fmt_int }}</div></div>
      {% if token_usage.total_reasoning_tokens %}
      <div class="token-item"><div class="label">Reasoning</div><div class="value">{{ token_usage.total_reasoning_tokens | fmt_int }}</div></div>
      {% endif %}
      <div class="token-item"><div class="label">LLM Calls</div><div class="value">{{ token_usage.num_calls | fmt_int }}</div></div>
      <div class="token-item"><div class="label">Max Prompt</div><div class="value">{{ token_usage.max_prompt_tokens | fmt_int }}</div></div>
      <div class="token-item"><div class="label">Max Completion</div><div class="value">{{ token_usage.max_completion_tokens | fmt_int }}</div></div>
    </div>
  </div>
</div>
{% endif %}

<!-- Session report iframe -->
{% if has_report %}
<div class="card mb-3">
  <div class="card-body p-0 iframe-wrapper">
    <iframe src="/static/report/{{ exp.dir_name }}/{{ sample_id }}"></iframe>
  </div>
</div>
{% else %}
<div class="card mb-3">
  <div class="card-body text-center text-muted py-5">
    No session report available for this sample.
  </div>
</div>
{% endif %}
{% endblock %}
"""

COMPARE_PAGE = """
{% extends "base.html" %}
{% block title %}Compare Experiments — Spatial Agent Results{% endblock %}
{% block extra_css %}
.agree { background: #d1e7dd; }
.disagree { background: #fff3cd; }
.perf-table th, .perf-table td { text-align: center; }
.perf-table th:first-child, .perf-table td:first-child { text-align: left; }
.perf-full { background: #f8f9fa; }
.perf-common { background: #e8f4fd; }
{% endblock %}
{% block content %}
<nav aria-label="breadcrumb">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="/">Dashboard</a></li>
    <li class="breadcrumb-item active">Compare</li>
  </ol>
</nav>

<h4 class="mb-3">Experiment Comparison</h4>

<!-- Overall Performance: Full vs Common -->
<div class="card mb-4">
  <div class="card-body">
    <h6 class="mb-3">Overall Accuracy</h6>
    <div class="table-responsive">
      <table class="table table-sm perf-table">
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Method</th>
            <th>Model</th>
            <th class="perf-full">Full Accuracy</th>
            <th class="perf-full">Full Samples</th>
            <th class="perf-common">Common Accuracy</th>
            <th class="perf-common">Common Samples</th>
          </tr>
        </thead>
        <tbody>
        {% for pd in perf_data %}
          <tr>
            <td style="text-align:left"><a href="/experiment/{{ pd.dir_name }}">{{ pd.dir_name }}</a></td>
            <td><span class="badge badge-method-{{ pd.method }}">{{ method_label(pd.method) }}</span></td>
            <td>{{ pd.model }}</td>
            <td class="perf-full">
              {% if pd.full.accuracy is not none %}
              <span class="{{ 'acc-high' if pd.full.accuracy >= 0.6 else ('acc-mid' if pd.full.accuracy >= 0.4 else 'acc-low') }}">
                {{ "%.1f" | format(pd.full.accuracy * 100) }}%
              </span>
              <span class="text-muted small">({{ pd.full.correct }}/{{ pd.full.total }})</span>
              {% else %}—{% endif %}
            </td>
            <td class="perf-full">{{ pd.full.total }}</td>
            <td class="perf-common">
              {% if pd.common.accuracy is not none %}
              <span class="{{ 'acc-high' if pd.common.accuracy >= 0.6 else ('acc-mid' if pd.common.accuracy >= 0.4 else 'acc-low') }}">
                {{ "%.1f" | format(pd.common.accuracy * 100) }}%
              </span>
              <span class="text-muted small">({{ pd.common.correct }}/{{ pd.common.total }})</span>
              {% else %}—{% endif %}
            </td>
            <td class="perf-common">{{ pd.common.total }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- Per-Category Comparison: Full vs Common -->
{% if all_cat_names %}
<div class="card mb-4">
  <div class="card-body">
    <h6 class="mb-3">Per-Category Comparison</h6>
    <div class="table-responsive">
      <table class="table table-sm perf-table">
        <thead>
          <tr>
            <th rowspan="2" style="vertical-align:middle">Category</th>
            {% for pd in perf_data %}
            <th colspan="2" class="text-center" style="border-bottom:none">{{ pd.dir_name }}</th>
            {% endfor %}
          </tr>
          <tr>
            {% for pd in perf_data %}
            <th class="perf-full small">Full</th>
            <th class="perf-common small">Common</th>
            {% endfor %}
          </tr>
        </thead>
        <tbody>
        {% for cat_name in all_cat_names %}
          <tr>
            <td style="text-align:left">{{ cat_name }}</td>
            {% for pd in perf_data %}
              {% set ns = namespace(full_row=None, common_row=None) %}
              {% for r in pd.full_categories %}{% if r.name == cat_name %}{% set ns.full_row = r %}{% endif %}{% endfor %}
              {% for r in pd.common_categories %}{% if r.name == cat_name %}{% set ns.common_row = r %}{% endif %}{% endfor %}
              <td class="perf-full">
                {% if ns.full_row %}
                  <span class="{{ 'acc-high' if ns.full_row.accuracy >= 0.6 else ('acc-mid' if ns.full_row.accuracy >= 0.4 else 'acc-low') }}">
                    {{ "%.1f" | format(ns.full_row.accuracy * 100) }}%
                  </span>
                  <span class="text-muted small">({{ ns.full_row.total }})</span>
                {% else %}—{% endif %}
              </td>
              <td class="perf-common">
                {% if ns.common_row %}
                  <span class="{{ 'acc-high' if ns.common_row.accuracy >= 0.6 else ('acc-mid' if ns.common_row.accuracy >= 0.4 else 'acc-low') }}">
                    {{ "%.1f" | format(ns.common_row.accuracy * 100) }}%
                  </span>
                  <span class="text-muted small">({{ ns.common_row.total }})</span>
                {% else %}—{% endif %}
              </td>
            {% endfor %}
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>
{% endif %}

<!-- Agreement stats -->
{% if stats %}
<div class="card mb-4">
  <div class="card-body">
    <h6>Agreement Statistics (Common Samples)</h6>
    <div class="row g-3">
      <div class="col-md-3"><div class="config-item"><div class="label">Common Samples</div><div class="value">{{ stats.common }}</div></div></div>
      <div class="col-md-3"><div class="config-item"><div class="label">Both Correct</div><div class="value acc-high">{{ stats.both_correct }}</div></div></div>
      <div class="col-md-3"><div class="config-item"><div class="label">Both Wrong</div><div class="value acc-low">{{ stats.both_wrong }}</div></div></div>
      <div class="col-md-3"><div class="config-item"><div class="label">Disagree</div><div class="value acc-mid">{{ stats.disagree }}</div></div></div>
    </div>
  </div>
</div>
{% endif %}

<!-- Per-sample comparison -->
<div class="card mb-4">
  <div class="card-body">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <h6 class="mb-0">Per-Sample Comparison</h6>
      <div class="d-flex gap-2">
        <div class="btn-group btn-group-sm" role="group">
          <button class="btn btn-outline-secondary filter-btn active" data-filter="all">All</button>
          <button class="btn btn-outline-warning filter-btn" data-filter="disagree">Disagree Only</button>
        </div>
        <input type="text" id="comp-search" class="form-control form-control-sm search-box" placeholder="Search...">
      </div>
    </div>
    <div class="table-responsive">
      <table class="table table-sm" id="comp-table">
        <thead>
          <tr>
            <th>Sample ID</th>
            <th>Ground Truth</th>
            {% for exp in experiments %}
            <th>{{ exp.dir_name }}</th>
            {% endfor %}
            <th>Agreement</th>
            <th style="width:50px"></th>
          </tr>
        </thead>
        <tbody>
        {% for row in comparison_rows %}
          <tr class="{{ 'agree' if row.agreement else 'disagree' }}" data-agree="{{ 'agree' if row.agreement else 'disagree' }}">
            <td><strong>{{ row.sample_id }}</strong></td>
            <td>{{ row.ground_truth }}</td>
            {% for pred in row.predictions %}
            <td>
              {% if pred.is_numerical and pred.mra_score is not none %}
              <span class="{{ 'text-success fw-bold' if pred.mra_score >= 0.5 else 'text-danger' }}">{{ pred.display or '—' }}</span>
              <span class="text-muted small">(MRA {{ "%.0f" | format(pred.mra_score * 100) }}%)</span>
              {% else %}
              <span class="{{ 'text-success fw-bold' if pred.is_correct else 'text-danger' }}">{{ pred.display or '—' }}</span>
              {% endif %}
            </td>
            {% endfor %}
            <td>
              {% if row.agreement %}
                <span class="badge bg-success">Agree</span>
              {% else %}
                <span class="badge bg-warning text-dark">Disagree</span>
              {% endif %}
            </td>
            <td>
              {% if row.has_any_report %}
              <a href="/compare/sample/{{ row.sample_id }}?{{ exp_query | safe }}" class="badge bg-info text-decoration-none" title="Side-by-side reports">View</a>
              {% endif %}
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filter = btn.dataset.filter;
      document.querySelectorAll('#comp-table tbody tr').forEach(row => {
        if (filter === 'all') row.style.display = '';
        else row.style.display = row.dataset.agree === 'disagree' ? '' : 'none';
      });
    });
  });
  document.getElementById('comp-search')?.addEventListener('input', e => {
    const q = e.target.value.toLowerCase();
    document.querySelectorAll('#comp-table tbody tr').forEach(row => {
      row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
</script>
{% endblock %}
"""

COMPARE_SAMPLE_PAGE = """
{% extends "base.html" %}
{% block title %}Sample {{ sample_id }} — Side-by-Side Comparison{% endblock %}
{% block extra_css %}
.compare-grid { display: grid; grid-template-columns: repeat({{ experiments | length }}, 1fr); gap: 12px; }
.compare-panel { border: 1px solid #dee2e6; border-radius: 6px; overflow: hidden; display: flex; flex-direction: column; }
.compare-panel-header { padding: 8px 12px; background: #f1f3f5; font-weight: 600; font-size: 0.9em; display: flex; justify-content: space-between; align-items: center; }
.compare-panel iframe { flex: 1; width: 100%; border: none; min-height: 80vh; }
.compare-panel .no-report { flex: 1; display: flex; align-items: center; justify-content: center; color: #6c757d; min-height: 300px; }
.compare-panel .md-content { flex: 1; padding: 16px; overflow: auto; max-height: 80vh; font-size: 0.9em; line-height: 1.5; }
.compare-panel .md-content pre { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; overflow-x: auto; }
.compare-panel .md-content code { font-size: 0.85em; }
.compare-panel .md-content p { margin-bottom: 0.5em; }
.compare-panel .md-content h1, .compare-panel .md-content h2, .compare-panel .md-content h3 { margin-top: 1em; }
.nav-sample { font-size: 0.9em; }
{% endblock %}
{% block content %}
<nav aria-label="breadcrumb">
  <ol class="breadcrumb">
    <li class="breadcrumb-item"><a href="/">Dashboard</a></li>
    <li class="breadcrumb-item"><a href="/compare?{{ exp_query | safe }}">Compare</a></li>
    <li class="breadcrumb-item active">Sample {{ sample_id }}</li>
  </ol>
</nav>

<div class="d-flex justify-content-between align-items-center mb-3">
  <div>
    <h4 class="mb-1">Sample {{ sample_id }}</h4>
    {% if prediction %}
    <span class="text-muted">Ground Truth: <strong>{{ prediction.ground_truth }}</strong></span>
    {% endif %}
  </div>
  <div class="d-flex gap-2 nav-sample">
    {% if prev_id is not none %}
      <a href="/compare/sample/{{ prev_id }}?{{ exp_query | safe }}" class="btn btn-outline-secondary btn-sm">&larr; {{ prev_id }}</a>
    {% endif %}
    {% if next_id is not none %}
      <a href="/compare/sample/{{ next_id }}?{{ exp_query | safe }}" class="btn btn-outline-secondary btn-sm">{{ next_id }} &rarr;</a>
    {% endif %}
  </div>
</div>

<!-- Prediction summary -->
<div class="card mb-3">
  <div class="card-body">
    <div class="table-responsive">
      <table class="table table-sm mb-0">
        <thead>
          <tr>
            <th>Experiment</th>
            <th>Prediction</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
        {% for ep in exp_predictions %}
          <tr>
            <td><a href="/experiment/{{ ep.dir_name }}/sample/{{ sample_id }}">{{ ep.dir_name }}</a></td>
            <td>{{ ep.extracted_pred or '(empty)' }}</td>
            <td>
              {% if ep.is_numerical and ep.mra_score is not none %}
                <span class="badge" style="background:{{ '#198754' if ep.mra_score >= 0.6 else ('#ffc107' if ep.mra_score >= 0.3 else '#dc3545') }};color:{{ '#000' if ep.mra_score >= 0.3 and ep.mra_score < 0.6 else '#fff' }}">
                  MRA {{ "%.0f" | format(ep.mra_score * 100) }}%
                </span>
              {% elif ep.is_correct %}
                <span class="badge bg-success">Correct</span>
              {% elif ep.is_empty %}
                <span class="badge bg-secondary">Empty</span>
              {% else %}
                <span class="badge bg-danger">Incorrect</span>
              {% endif %}
            </td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- Side-by-side reports -->
<div class="compare-grid">
{% for ep in exp_predictions %}
  <div class="compare-panel">
    <div class="compare-panel-header">
      <span>{{ ep.dir_name }}</span>
      {% if ep.is_correct %}
        <span class="badge bg-success">Correct</span>
      {% elif ep.is_empty %}
        <span class="badge bg-secondary">Empty</span>
      {% else %}
        <span class="badge bg-danger">Incorrect</span>
      {% endif %}
    </div>
    {% if ep.has_report %}
      {% if ep.content %}
      <div class="md-content" id="md-{{ loop.index }}" style="max-height: 200px; border-bottom: 1px solid #dee2e6;"></div>
      {% endif %}
      <iframe src="/static/report/{{ ep.dir_name }}/{{ sample_id }}"></iframe>
    {% elif ep.content %}
      <div class="md-content" id="md-{{ loop.index }}"></div>
    {% else %}
      <div class="no-report">No response content</div>
    {% endif %}
  </div>
{% endfor %}
</div>

<script src="https://cdn.jsdelivr.net/npm/marked@15.0.0/marked.min.js"></script>
<script>
(function() {
  const contents = {
    {% for ep in exp_predictions %}
      {% if ep.content %}
        {{ loop.index }}: {{ ep.content | tojson }},
      {% endif %}
    {% endfor %}
  };
  for (const [idx, raw] of Object.entries(contents)) {
    const el = document.getElementById('md-' + idx);
    if (el) el.innerHTML = marked.parse(raw);
  }
})();
</script>
{% endblock %}
"""
