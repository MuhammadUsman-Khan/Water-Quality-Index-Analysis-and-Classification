"use strict";

/* ═══════════════════════════════
   CONSTANTS
═══════════════════════════════ */
const DANGER_THRESHOLDS = {
  aluminium:   2.8,
  ammonia:     32.5,
  arsenic:     0.01,
  barium:      2.0,
  cadmium:     0.005,
  chloramine:  4.0,
  chromium:    0.1,
  copper:      1.3,
  flouride:    1.5,
  bacteria:    0,
  viruses:     0,
  lead:        0.015,
  nitrates:    10.0,
  nitrites:    1.0,
  mercury:     0.002,
  perchlorate: 56.0,
  radium:      5.0,
  selenium:    0.5,
  silver:      0.1,
  uranium:     0.3,
};

const RESULTS = {
  safe: {
    iconClass: "safe-ico",
    iconHtml:  '<i class="fas fa-shield-check"></i>',
    labelClass: "safe",
    label:     "WATER IS SAFE",
    cardClass:  "safe",
    desc: "This sample meets the safety classification criteria. All or sufficient parameters fall within WHO-acceptable ranges according to the trained Decision Tree classifier.",
  },
  unsafe: {
    iconClass: "unsafe-ico",
    iconHtml:  '<i class="fas fa-triangle-exclamation"></i>',
    labelClass: "unsafe",
    label:     "WATER IS UNSAFE",
    cardClass:  "unsafe",
    desc: "This sample does not meet safety standards. One or more parameters exceed their danger thresholds. Detailed breakdown is shown below. Further lab testing is strongly advised.",
  },
};

/* ═══════════════════════════════
   ELEMENTS
═══════════════════════════════ */
const form        = document.getElementById("predForm");
const analyzeBtn  = document.getElementById("analyzeBtn");
const resetBtn    = document.getElementById("resetBtn");
const newRunBtn   = document.getElementById("newRunBtn");
const progressFill = document.getElementById("progressFill");
const progressPct  = document.getElementById("progressPct");
const resultSection = document.getElementById("resultSection");

const verdictCard   = document.getElementById("verdictCard");
const vcIcon        = document.getElementById("vcIcon");
const vcLabel       = document.getElementById("vcLabel");
const vcDesc        = document.getElementById("vcDesc");
const vcConfidence  = document.getElementById("vcConfidence");
const confFill      = document.getElementById("confFill");
const confNum       = document.getElementById("confNum");

const flaggedSection = document.getElementById("flaggedSection");
const flaggedGrid    = document.getElementById("flaggedGrid");
const summaryTbody   = document.getElementById("summaryTbody");

const allInputs = Array.from(form.querySelectorAll(".pcard-input"));
const totalFields = allInputs.length;

/* ═══════════════════════════════
   PROGRESS TRACKER
═══════════════════════════════ */
function updateProgress() {
  const filled = allInputs.filter(i => i.value.trim() !== "").length;
  const pct = Math.round((filled / totalFields) * 100);
  progressFill.style.width = pct + "%";
  progressPct.textContent = `${filled} / ${totalFields}`;
}

/* ═══════════════════════════════
   LIVE DANGER BADGES
═══════════════════════════════ */
function updateBadge(input) {
  const feat  = input.id;
  const badge = document.getElementById("badge-" + feat);
  const card  = document.getElementById("card-" + feat);
  const val   = parseFloat(input.value);
  const thresh = DANGER_THRESHOLDS[feat];

  if (input.value.trim() === "" || isNaN(val)) {
    badge.textContent = "";
    badge.style.cssText = "";
    card.classList.remove("is-filled", "is-danger");
    return;
  }

  card.classList.add("is-filled");

  if (val > thresh) {
    badge.textContent = "⚠ HIGH";
    badge.style.cssText = "color: var(--amber); background: var(--amber-dim); border: 1px solid rgba(255,184,63,0.25);";
    card.classList.add("is-danger");
  } else {
    badge.textContent = "✓ OK";
    badge.style.cssText = "color: var(--green); background: var(--green-dim); border: 1px solid rgba(0,232,150,0.2);";
    card.classList.remove("is-danger");
  }
}

/* Attach live listeners */
allInputs.forEach(input => {
  input.addEventListener("input", () => {
    updateProgress();
    updateBadge(input);
    // Clear error state on type
    const card = document.getElementById("card-" + input.id);
    const err  = document.getElementById("err-" + input.id);
    card.classList.remove("has-error");
    err.textContent = "";
  });

  input.addEventListener("focus", () => {
    const card = document.getElementById("card-" + input.id);
    if (!card.classList.contains("has-error") && !card.classList.contains("is-danger")) {
      card.style.borderColor = "var(--border-hi)";
    }
  });

  input.addEventListener("blur", () => {
    const card = document.getElementById("card-" + input.id);
    if (!card.classList.contains("has-error") && !card.classList.contains("is-danger")) {
      card.style.borderColor = "";
    }
  });
});

/* ═══════════════════════════════
   FORM SUBMIT
═══════════════════════════════ */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearErrors();

  const payload = {};
  let valid = true;
  let firstErrCard = null;

  allInputs.forEach(input => {
    const val = input.value.trim();
    const card = document.getElementById("card-" + input.id);
    const err  = document.getElementById("err-" + input.id);

    if (val === "") {
      markError(card, err, "Required");
      if (!firstErrCard) firstErrCard = card;
      valid = false;
    } else if (isNaN(parseFloat(val)) || !isFinite(val)) {
      markError(card, err, "Enter a valid number");
      if (!firstErrCard) firstErrCard = card;
      valid = false;
    } else {
      payload[input.id] = parseFloat(val);
    }
  });

  if (!valid) {
    firstErrCard && firstErrCard.scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }

  setLoading(true);

  try {
    const res  = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (!res.ok || data.error) {
      showGlobalError(data.error || "Prediction failed. Please try again.");
      return;
    }

    renderResult(data);
  } catch (err) {
    showGlobalError("Network error — could not reach the server. Please try again.");
  } finally {
    setLoading(false);
  }
});

/* ═══════════════════════════════
   RENDER RESULT
═══════════════════════════════ */
function renderResult(data) {
  const type = data.prediction === 1 ? "safe" : "unsafe";
  const r = RESULTS[type];

  // Verdict card
  verdictCard.className = "verdict-card " + r.cardClass;
  vcIcon.className  = "vc-icon " + r.iconClass;
  vcIcon.innerHTML  = r.iconHtml;
  vcLabel.className = "vc-label " + r.labelClass;
  vcLabel.textContent = r.label;
  vcDesc.textContent  = r.desc;

  // Confidence bar
  if (data.confidence != null) {
    vcConfidence.style.display = "block";
    confFill.className = "conf-fill " + type;
    confNum.textContent = data.confidence + "%";
    setTimeout(() => { confFill.style.width = data.confidence + "%"; }, 150);
  } else {
    vcConfidence.style.display = "none";
  }

  // Flagged parameters
  if (data.flagged && data.flagged.length > 0) {
    flaggedSection.style.display = "block";
    flaggedGrid.innerHTML = "";
    data.flagged.forEach(f => {
      const div = document.createElement("div");
      div.className = "flag-item";
      div.innerHTML = `
        <div class="flag-name">${f.label}</div>
        <div class="flag-val">${f.value} <span style="font-size:11px;color:var(--txt-2)">${f.unit}</span></div>
        <div class="flag-thresh">Threshold: ${f.threshold} ${f.unit}</div>`;
      flaggedGrid.appendChild(div);
    });
  } else {
    flaggedSection.style.display = "none";
  }

  // Summary table
  summaryTbody.innerHTML = "";
  const entries = Object.entries(data.inputs);
  entries.forEach(([label, valueStr], idx) => {
    // Determine if this row is flagged
    const flaggedLabels = (data.flagged || []).map(f => f.label);
    const isFlagged = flaggedLabels.includes(label);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>${label}</td>
      <td>${valueStr}</td>
      <td>${getThresholdForLabel(label)}</td>
      <td>${isFlagged
        ? '<span class="status-warn"><i class="fas fa-triangle-exclamation"></i> High</span>'
        : '<span class="status-ok"><i class="fas fa-check"></i> OK</span>'
      }</td>`;
    summaryTbody.appendChild(tr);
  });

  // Show result section
  resultSection.style.display = "block";
  resultSection.style.animation = "none";
  void resultSection.offsetWidth;
  resultSection.style.animation = "fadeUp 0.45s ease both";

  setTimeout(() => {
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 80);
}

/* ═══════════════════════════════
   HELPERS
═══════════════════════════════ */
// Map display labels back to thresholds for the table
const LABEL_TO_THRESH = {
  "Aluminium":   "2.8 mg/L",
  "Ammonia":     "32.5 mg/L",
  "Arsenic":     "0.01 mg/L",
  "Barium":      "2.0 mg/L",
  "Cadmium":     "0.005 mg/L",
  "Chloramine":  "4.0 mg/L",
  "Chromium":    "0.1 mg/L",
  "Copper":      "1.3 mg/L",
  "Fluoride":    "1.5 mg/L",
  "Bacteria":    "0 count",
  "Viruses":     "0 count",
  "Lead":        "0.015 mg/L",
  "Nitrates":    "10.0 mg/L",
  "Nitrites":    "1.0 mg/L",
  "Mercury":     "0.002 mg/L",
  "Perchlorate": "56.0 mg/L",
  "Radium":      "5.0 pCi/L",
  "Selenium":    "0.5 mg/L",
  "Silver":      "0.1 mg/L",
  "Uranium":     "0.3 mg/L",
};

function getThresholdForLabel(label) {
  return LABEL_TO_THRESH[label] || "—";
}

function setLoading(on) {
  analyzeBtn.disabled = on;
  analyzeBtn.querySelector(".btn-def").style.display = on ? "none" : "inline-flex";
  analyzeBtn.querySelector(".btn-spin").style.display = on ? "inline-flex" : "none";
}

function markError(card, errEl, msg) {
  card.classList.add("has-error");
  errEl.textContent = msg;
}

function clearErrors() {
  form.querySelectorAll(".pcard").forEach(c => c.classList.remove("has-error"));
  form.querySelectorAll(".pcard-error").forEach(e => e.textContent = "");
  const t = document.getElementById("globalErrToast");
  if (t) t.remove();
}

function showGlobalError(msg) {
  const existing = document.getElementById("globalErrToast");
  if (existing) existing.remove();
  const el = document.createElement("div");
  el.id = "globalErrToast";
  el.className = "err-toast";
  el.innerHTML = `<i class="fas fa-circle-exclamation"></i><span>${msg}</span>`;
  form.insertAdjacentElement("afterend", el);
  el.scrollIntoView({ behavior: "smooth", block: "center" });
}

/* ═══════════════════════════════
   RESET & NEW RUN
═══════════════════════════════ */
resetBtn.addEventListener("click", () => {
  form.reset();
  clearErrors();
  allInputs.forEach(input => {
    updateBadge(input);
    document.getElementById("card-" + input.id).style.borderColor = "";
  });
  updateProgress();
  resultSection.style.display = "none";
  document.getElementById("form-section").scrollIntoView({ behavior: "smooth", block: "start" });
});

newRunBtn.addEventListener("click", () => {
  resultSection.style.display = "none";
  document.getElementById("form-section").scrollIntoView({ behavior: "smooth", block: "start" });
});
