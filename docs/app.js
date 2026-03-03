const bodyEl = document.getElementById('results-body');
const stampEl = document.getElementById('stamp');
const tabs = Array.from(document.querySelectorAll('.tab'));
const thEls = Array.from(document.querySelectorAll('#results-table thead tr:first-child th'));

const modelSelect = document.getElementById('model-select');
const deviceSelect = document.getElementById('device-select');
const resetBtn = document.getElementById('reset-filters');

const fieldFilters = {
  rank: document.getElementById('f-rank'),
  model: document.getElementById('f-model'),
  device: document.getElementById('f-device'),
  runs: document.getElementById('f-runs'),
  ttft_avg: document.getElementById('f-ttft-avg'),
  tps_avg: document.getElementById('f-tps-avg'),
  ttft_last: document.getElementById('f-ttft-last'),
  tps_last: document.getElementById('f-tps-last'),
};

const state = {
  normal: [],
  benchmark: [],
  mode: 'benchmark',
  sortCol: 'tps_avg',
  sortDir: 'desc',
};

function num(v, digits = 3) {
  const n = Number(v || 0);
  return Number.isFinite(n) ? n.toFixed(digits) : '0.000';
}

function applyData(data) {
  state.normal = data.normal || [];
  state.benchmark = data.benchmark || [];
  const stamp = data.generated_at_utc ? new Date(data.generated_at_utc).toLocaleString() : 'Unknown';
  stampEl.textContent = `Snapshot: ${stamp}`;
  hydrateTopFilters();
}

function hydrateTopFilters() {
  const rows = state[state.mode] || [];
  const models = [...new Set(rows.map((r) => r.model))].sort((a, b) => a.localeCompare(b));
  const devices = [...new Set(rows.map((r) => r.device))].sort((a, b) => a.localeCompare(b));

  const prevModel = modelSelect.value;
  const prevDevice = deviceSelect.value;

  modelSelect.innerHTML = '<option value="">All models</option>';
  models.forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    modelSelect.appendChild(opt);
  });

  deviceSelect.innerHTML = '<option value="">All devices</option>';
  devices.forEach((d) => {
    const opt = document.createElement('option');
    opt.value = d;
    opt.textContent = d;
    deviceSelect.appendChild(opt);
  });

  if (models.includes(prevModel)) modelSelect.value = prevModel;
  if (devices.includes(prevDevice)) deviceSelect.value = prevDevice;
}

function parseNumber(s) {
  if (s === null || s === undefined) return null;
  const t = String(s).trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

function matchRow(row, idx) {
  const selectedModel = modelSelect.value;
  const selectedDevice = deviceSelect.value;

  if (selectedModel && row.model !== selectedModel) return false;
  if (selectedDevice && row.device !== selectedDevice) return false;

  const rankFilter = parseNumber(fieldFilters.rank.value);
  if (rankFilter !== null && idx + 1 !== rankFilter) return false;

  const modelText = fieldFilters.model.value.trim().toLowerCase();
  if (modelText && !row.model.toLowerCase().includes(modelText)) return false;

  const deviceText = fieldFilters.device.value.trim().toLowerCase();
  if (deviceText && !row.device.toLowerCase().includes(deviceText)) return false;

  const runsEq = parseNumber(fieldFilters.runs.value);
  if (runsEq !== null && Number(row.runs) !== runsEq) return false;

  const ttftAvgMax = parseNumber(fieldFilters.ttft_avg.value);
  if (ttftAvgMax !== null && Number(row.ttft_avg) > ttftAvgMax) return false;

  const tpsAvgMin = parseNumber(fieldFilters.tps_avg.value);
  if (tpsAvgMin !== null && Number(row.tps_avg) < tpsAvgMin) return false;

  const ttftLastMax = parseNumber(fieldFilters.ttft_last.value);
  if (ttftLastMax !== null && Number(row.ttft_last) > ttftLastMax) return false;

  const tpsLastMin = parseNumber(fieldFilters.tps_last.value);
  if (tpsLastMin !== null && Number(row.tps_last) < tpsLastMin) return false;

  return true;
}

function sortRows(rows) {
  const col = state.sortCol;
  const dir = state.sortDir === 'asc' ? 1 : -1;

  const cmp = (a, b) => {
    if (col === 'rank') return 0;

    const numericCols = new Set(['runs', 'ttft_avg', 'tps_avg', 'ttft_last', 'tps_last']);
    if (numericCols.has(col)) {
      return (Number(a[col]) - Number(b[col])) * dir;
    }
    return String(a[col]).localeCompare(String(b[col])) * dir;
  };

  return rows.slice().sort(cmp);
}

function updateSortIndicators() {
  thEls.forEach((th) => {
    th.querySelectorAll('.sort-label').forEach((el) => el.remove());
    const col = th.dataset.col;
    if (!col || col !== state.sortCol || col === 'rank') return;
    const span = document.createElement('span');
    span.className = 'sort-label';
    span.textContent = state.sortDir === 'asc' ? '↑' : '↓';
    th.appendChild(span);
  });
}

function render() {
  const sourceRows = state[state.mode] || [];
  const filtered = sourceRows.filter((r, idx) => matchRow(r, idx));
  const rows = sortRows(filtered);
  bodyEl.innerHTML = '';

  if (!rows.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = 8;
    td.textContent = 'No benchmark rows for this mode yet.';
    tr.appendChild(td);
    bodyEl.appendChild(tr);
    return;
  }

  const selectedModel = modelSelect.value;
  const selectedDevice = deviceSelect.value;

  rows.forEach((r, idx) => {
    const tr = document.createElement('tr');
    if (selectedModel && !selectedDevice && r.model === selectedModel) {
      tr.classList.add('hl-model');
    }
    tr.innerHTML = `
      <td class="rank">${idx + 1}</td>
      <td>${r.model}</td>
      <td>${r.device}</td>
      <td>${r.runs}</td>
      <td>${num(r.ttft_avg)}</td>
      <td>${num(r.tps_avg)}</td>
      <td>${num(r.ttft_last)}</td>
      <td>${num(r.tps_last)}</td>
    `;
    bodyEl.appendChild(tr);
  });

  updateSortIndicators();
}

function setMode(mode) {
  state.mode = mode;
  tabs.forEach((btn) => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  hydrateTopFilters();
  render();
}

function setSort(col) {
  if (col === 'rank') return;
  if (state.sortCol === col) {
    state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
  } else {
    state.sortCol = col;
    state.sortDir = col === 'model' || col === 'device' ? 'asc' : 'desc';
  }
  render();
}

function resetFilters() {
  modelSelect.value = '';
  deviceSelect.value = '';
  Object.values(fieldFilters).forEach((el) => {
    el.value = '';
  });
  render();
}

async function boot() {
  if (window.BENCHMARKS_DATA) {
    applyData(window.BENCHMARKS_DATA);
    render();
    return;
  }

  try {
    const res = await fetch('benchmarks.json', { cache: 'no-store' });
    const data = await res.json();
    applyData(data);
    render();
  } catch (err) {
    stampEl.textContent = 'Could not load benchmark data';
    bodyEl.innerHTML = '<tr><td colspan="8">Failed to load benchmark data.</td></tr>';
  }
}

tabs.forEach((btn) => {
  btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

thEls.forEach((th) => {
  const col = th.dataset.col;
  if (!col) return;
  th.addEventListener('click', () => setSort(col));
});

[modelSelect, deviceSelect].forEach((el) => el.addEventListener('change', render));
Object.values(fieldFilters).forEach((el) => el.addEventListener('input', render));
resetBtn.addEventListener('click', resetFilters);

boot();
