/* ═══════════════════════════════════════════════════════════════
   EuroMLions — Site Logic
   Fetches predictions + history JSON, renders the UI
   ═══════════════════════════════════════════════════════════════ */

// ─── Configuration ───────────────────────────────────────────
// Data file paths (relative — works with netlify.toml build step)
const PREDICTIONS_URL = 'predictions.json';
const HISTORY_URL = 'history-predictions.json';

// ─── Init ────────────────────────────────────────────────────
// Password protection handled by Netlify — no JS gate needed.

function initApp() {
  document.getElementById('app').classList.remove('hidden');
  loadData();
}

// ─── Navigation ──────────────────────────────────────────────

function initNav() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.section).classList.add('active');
    });
  });
}

// ─── Data Loading ────────────────────────────────────────────

async function loadData() {
  try {
    const [predictionsRes, historyRes] = await Promise.all([
      fetch(PREDICTIONS_URL),
      fetch(HISTORY_URL),
    ]);

    const predictions = await predictionsRes.json();
    const history = await historyRes.json();

    renderPredictions(predictions);
    renderModels(predictions);
    renderHistory(history);
    renderStats(predictions, history);
  } catch (err) {
    console.error('Failed to load data:', err);
  }
}

// ─── Render: Predictions ─────────────────────────────────────

function renderPredictions(data) {
  const ens = data.ensemble;
  if (!ens) return;

  document.getElementById('ensemble-main').innerHTML = renderBalls(ens.main_numbers, 'main');
  document.getElementById('ensemble-stars').innerHTML = renderBalls(ens.lucky_stars, 'star');

  // Uncertainty display
  if (ens.main_uncertainty) {
    const avgUncertainty = average(ens.main_uncertainty);
    const confidence = uncertaintyToLabel(avgUncertainty);
    document.getElementById('ensemble-uncertainty').textContent =
      `Model confidence: ${confidence}`;
  }

  // Meta info
  const meta = document.getElementById('prediction-meta');
  const generated = data.generated ? formatDate(data.generated) : '—';
  const trainedOn = data.trained_on || '—';
  meta.innerHTML = `
    <div class="meta-item">Generated: <span>${generated}</span></div>
    <div class="meta-item">Trained on: <span>${trainedOn}</span></div>
  `;

  // Countdown
  renderCountdown();
}

// ─── Render: Models ──────────────────────────────────────────

function renderModels(data) {
  // Model A
  if (data.model_A_combined) {
    const a = data.model_A_combined;
    document.getElementById('model-a-main').innerHTML = renderBalls(a.main_numbers, 'main', true);
    document.getElementById('model-a-stars').innerHTML = renderBalls(a.lucky_stars, 'star', true);
  }

  // Model B
  if (data.model_B_main_only) {
    document.getElementById('model-b-main').innerHTML =
      renderBalls(data.model_B_main_only.main_numbers, 'main', true);
  }

  // Model C
  if (data.model_C_stars_only) {
    document.getElementById('model-c-stars').innerHTML =
      renderBalls(data.model_C_stars_only.lucky_stars, 'star', true);
  }

  // Consensus analysis
  renderConsensus(data);
}

function renderConsensus(data) {
  const el = document.getElementById('consensus');
  const ens = data.ensemble;
  const a = data.model_A_combined;
  const b = data.model_B_main_only;
  const c = data.model_C_stars_only;

  if (!ens || !a || !b) { el.innerHTML = ''; return; }

  // Find numbers that appear in multiple models' main picks
  const allMainPicks = [
    ...(ens.main_numbers || []),
    ...(a.main_numbers || []),
    ...(b.main_numbers || []),
  ];
  const mainCounts = {};
  allMainPicks.forEach(n => { mainCounts[n] = (mainCounts[n] || 0) + 1; });
  const consensus = Object.entries(mainCounts)
    .filter(([, count]) => count >= 3)
    .map(([num]) => Number(num))
    .sort((a, b) => a - b);

  // Stars consensus
  const allStarPicks = [
    ...(ens.lucky_stars || []),
    ...(a.lucky_stars || []),
    ...(c ? c.lucky_stars || [] : []),
  ];
  const starCounts = {};
  allStarPicks.forEach(n => { starCounts[n] = (starCounts[n] || 0) + 1; });
  const starConsensus = Object.entries(starCounts)
    .filter(([, count]) => count >= 3)
    .map(([num]) => Number(num))
    .sort((a, b) => a - b);

  let html = '<strong>Consensus:</strong> ';
  if (consensus.length > 0) {
    html += `Numbers <strong>${consensus.join(', ')}</strong> are picked by all main number models. `;
  } else {
    html += 'No main numbers are unanimous across all models. ';
  }
  if (starConsensus.length > 0) {
    html += `Lucky Star${starConsensus.length > 1 ? 's' : ''} <strong>${starConsensus.join(', ')}</strong> ${starConsensus.length > 1 ? 'are' : 'is'} unanimous.`;
  }

  el.innerHTML = html;
}

// ─── Render: History ─────────────────────────────────────────

function renderHistory(history) {
  const el = document.getElementById('history-content');

  if (!history || history.length === 0) {
    el.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">📭</div>
        <p>No prediction history yet. Check back after the first draw!</p>
      </div>`;
    return;
  }

  // Show newest first
  const sorted = [...history].reverse();

  el.innerHTML = sorted.map(entry => {
    const hasActual = entry.actual_main !== null;
    const ens = entry.ensemble || {};
    const mA = entry.model_A || {};
    const mB = entry.model_B || {};
    const mC = entry.model_C || {};

    // Ensemble prediction (headline)
    let predictedHtml = `
      <div class="history-row">
        <span class="history-label">Ensemble</span>
        <div class="balls-row small">
          ${renderBalls(ens.main, 'main', true, hasActual ? entry.actual_main : null)}
          ${renderBalls(ens.stars, 'star', true, hasActual ? entry.actual_stars : null)}
        </div>
      </div>`;

    // Individual models (collapsible detail)
    predictedHtml += `
      <div class="history-models">
        <div class="history-row">
          <span class="history-label sub">Model A</span>
          <div class="balls-row small">
            ${renderBalls(mA.main, 'main', true, hasActual ? entry.actual_main : null)}
            ${renderBalls(mA.stars, 'star', true, hasActual ? entry.actual_stars : null)}
          </div>
        </div>
        <div class="history-row">
          <span class="history-label sub">Model B</span>
          <div class="balls-row small">
            ${renderBalls(mB.main, 'main', true, hasActual ? entry.actual_main : null)}
          </div>
        </div>
        <div class="history-row">
          <span class="history-label sub">Model C</span>
          <div class="balls-row small">
            ${renderBalls(mC.stars, 'star', true, hasActual ? entry.actual_stars : null)}
          </div>
        </div>
      </div>`;

    let actualHtml = '';
    if (hasActual) {
      actualHtml = `
        <div class="history-row">
          <span class="history-label">Actual</span>
          <div class="balls-row small">
            ${renderBalls(entry.actual_main, 'main', true)}
            ${renderBalls(entry.actual_stars, 'star', true)}
          </div>
        </div>
        <div class="history-row">
          <span class="history-label">Result</span>
          <span class="history-result ${entry.main_matched === 0 && entry.stars_matched === 0 ? 'none' : ''}">
            ${entry.main_matched}/5 main, ${entry.stars_matched}/2 stars
          </span>
        </div>`;
    } else {
      actualHtml = `
        <div class="history-row">
          <span class="history-label">Actual</span>
          <span class="history-pending">Awaiting draw results...</span>
        </div>`;
    }

    return `
      <div class="history-entry">
        <div class="history-date">Draw: ${formatDateShort(entry.draw_date)}</div>
        <div class="history-version">${entry.model_version || ''}</div>
        ${predictedHtml}
        ${actualHtml}
      </div>`;
  }).join('');
}

// ─── Render: Stats ───────────────────────────────────────────

function renderStats(predictions, history) {
  const el = document.getElementById('stats-content');

  // Compute running stats from history — per model
  const completed = history.filter(h => h.actual_main !== null);
  const totalDraws = completed.length;

  function modelStats(completed, getMain, getStars) {
    let mainSum = 0, starsSum = 0, bestMain = 0, bestStars = 0;
    completed.forEach(h => {
      const pm = getMain(h) || [];
      const ps = getStars(h) || [];
      const mm = pm.filter(n => h.actual_main.includes(n)).length;
      const sm = ps.filter(n => h.actual_stars.includes(n)).length;
      mainSum += mm;
      starsSum += sm;
      bestMain = Math.max(bestMain, mm);
      bestStars = Math.max(bestStars, sm);
    });
    return {
      avgMain: totalDraws > 0 ? (mainSum / totalDraws).toFixed(2) : '—',
      avgStars: totalDraws > 0 ? (starsSum / totalDraws).toFixed(2) : '—',
      bestMain, bestStars,
    };
  }

  const ens  = modelStats(completed, h => (h.ensemble || {}).main, h => (h.ensemble || {}).stars);
  const modA = modelStats(completed, h => (h.model_A || {}).main, h => (h.model_A || {}).stars);
  const modB = modelStats(completed, h => (h.model_B || {}).main, () => []);
  const modC = modelStats(completed, () => [],                     h => (h.model_C || {}).stars);

  function statRow(label, s, showMain, showStars) {
    return `
      <tr>
        <td><strong>${label}</strong></td>
        <td>${showMain ? s.avgMain : '—'}</td>
        <td>${showStars ? s.avgStars : '—'}</td>
        <td>${showMain ? s.bestMain + '/5' : '—'}</td>
        <td>${showStars ? s.bestStars + '/2' : '—'}</td>
      </tr>`;
  }

  // Summary stats
  let statsHtml = `
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">${totalDraws}</div>
        <div class="stat-label">Draws tracked</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${ens.avgMain}</div>
        <div class="stat-label">Avg main matched</div>
        <div class="stat-baseline">Random: 0.50</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${ens.avgStars}</div>
        <div class="stat-label">Avg stars matched</div>
        <div class="stat-baseline">Random: 0.33</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${ens.bestMain}/5</div>
        <div class="stat-label">Best main</div>
      </div>
    </div>

    <h3 style="font-size: 1rem; margin: 24px 0 12px;">Per-Model Performance</h3>
    <div class="stats-table-wrap">
      <table class="stats-table">
        <thead>
          <tr>
            <th>Model</th><th>Avg Main</th><th>Avg Stars</th><th>Best Main</th><th>Best Stars</th>
          </tr>
        </thead>
        <tbody>
          ${statRow('Ensemble', ens, true, true)}
          ${statRow('Model A (Combined)', modA, true, true)}
          ${statRow('Model B (Main)', modB, true, false)}
          ${statRow('Model C (Stars)', modC, false, true)}
        </tbody>
      </table>
    </div>`;

  // Validation evaluation (from training)
  const eval_ = predictions.evaluation;
  if (eval_) {
    statsHtml += '<h3 style="font-size: 1rem; margin: 24px 0 12px;">Training Evaluation (Validation Set)</h3>';

    // Helper to render a match distribution bar chart
    function evalCard(title, evalData, outOf) {
      const dist = evalData.match_distribution;
      const total = Object.values(dist).reduce((a, b) => a + b, 0);
      return `
        <div class="eval-card">
          <div class="eval-title">${title}</div>
          ${Object.entries(dist).map(([matches, count]) => `
            <div class="eval-bar-row">
              <span class="eval-bar-label">${matches} matched</span>
              <div class="eval-bar-track">
                <div class="eval-bar-fill" style="width: ${(count / total * 100).toFixed(1)}%"></div>
              </div>
              <span style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono); width: 40px;">${count}</span>
            </div>
          `).join('')}
          <div style="margin-top: 8px; font-size: 0.8rem; color: var(--text-secondary);">
            Avg: <strong>${evalData.avg_matched}</strong>/${outOf} matched
            (random baseline: ${evalData.random_baseline})
          </div>
        </div>`;
    }

    if (eval_.model_A_combined) {
      if (eval_.model_A_combined.main_numbers) {
        statsHtml += evalCard('Model A (Combined) — Main Numbers', eval_.model_A_combined.main_numbers, 5);
      }
      if (eval_.model_A_combined.lucky_stars) {
        statsHtml += evalCard('Model A (Combined) — Lucky Stars', eval_.model_A_combined.lucky_stars, 2);
      }
    }

    if (eval_.model_B_main_only && eval_.model_B_main_only.main_numbers) {
      statsHtml += evalCard('Model B (Main Numbers) — Match Distribution', eval_.model_B_main_only.main_numbers, 5);
    }

    if (eval_.model_C_stars_only && eval_.model_C_stars_only.lucky_stars) {
      statsHtml += evalCard('Model C (Lucky Stars) — Match Distribution', eval_.model_C_stars_only.lucky_stars, 2);
    }
  }

  if (totalDraws === 0) {
    statsHtml += `
      <div class="empty-state" style="padding: 24px;">
        <p style="color: var(--text-muted);">Live prediction stats will appear after the first draw results come in.</p>
      </div>`;
  }

  el.innerHTML = statsHtml;
}

// ─── Helpers ─────────────────────────────────────────────────

function renderBalls(numbers, type, small = false, matchAgainst = null) {
  if (!numbers) return '';
  return numbers.map(n => {
    const cls = type === 'star' ? 'ball-star' : 'ball-main';
    const sizeClass = small ? 'small' : '';
    let matchClass = '';
    if (matchAgainst) {
      matchClass = matchAgainst.includes(n) ? 'match' : 'miss';
    }
    return `<span class="ball ${cls} ${sizeClass} ${matchClass}">${n}</span>`;
  }).join('');
}

function average(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function uncertaintyToLabel(avgUncertainty) {
  if (avgUncertainty < 0.015) return 'Surprisingly confident (for a lottery)';
  if (avgUncertainty < 0.025) return 'Mildly confident (probably shouldn\'t be)';
  if (avgUncertainty < 0.04) return 'Appropriately uncertain';
  return 'Extremely uncertain (as expected)';
}

function formatDate(isoString) {
  const d = new Date(isoString);
  return d.toLocaleDateString('en-GB', {
    weekday: 'short', day: 'numeric', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function formatDateShort(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-GB', {
    weekday: 'short', day: 'numeric', month: 'short', year: 'numeric',
  });
}

function getNextDrawDate() {
  const now = new Date();
  const day = now.getDay();
  let daysUntil;

  // EuroMillions: Tuesday (2) and Friday (5)
  if (day < 2) daysUntil = 2 - day;
  else if (day < 5) daysUntil = 5 - day;
  else if (day === 5) {
    // If it's Friday, check if draw time has passed (~20:45 CET / 19:45 GMT)
    const drawHour = 20;
    if (now.getHours() >= drawHour) daysUntil = 4; // next Tuesday
    else daysUntil = 0;
  } else if (day === 6) daysUntil = 3; // Saturday -> Tuesday
  else daysUntil = 2; // Sunday -> Tuesday

  if (day === 2 && now.getHours() >= 20) daysUntil = 3; // Tuesday after draw -> Friday

  const next = new Date(now);
  next.setDate(now.getDate() + daysUntil);
  next.setHours(20, 45, 0, 0);
  return next;
}

function renderCountdown() {
  const el = document.getElementById('draw-countdown');
  const nextDraw = getNextDrawDate();

  function update() {
    const now = new Date();
    const diff = nextDraw - now;

    if (diff <= 0) {
      el.textContent = 'Draw is happening now!';
      return;
    }

    const days = Math.floor(diff / 86400000);
    const hours = Math.floor((diff % 86400000) / 3600000);
    const mins = Math.floor((diff % 3600000) / 60000);

    let parts = [];
    if (days > 0) parts.push(`${days}d`);
    parts.push(`${hours}h`);
    parts.push(`${mins}m`);

    const dayName = nextDraw.toLocaleDateString('en-GB', { weekday: 'long' });
    el.textContent = `Next draw: ${dayName} (${parts.join(' ')})`;
  }

  update();
  setInterval(update, 60000);
}

// ─── Initialise ──────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initApp();
  initNav();
});
