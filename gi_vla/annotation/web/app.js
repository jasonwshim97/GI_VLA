"use strict";

const state = {
  items: [],
  decisions: {},
  summary: null,
  filtered: [],
  cursor: 0,
  item: null,
  image: null,
  fixMode: false,
  draftBox: null,
  dragStart: null,
};

const $ = (id) => document.getElementById(id);
const canvas = $("review-canvas");
const ctx = canvas.getContext("2d");

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.style.background = error ? "#8b2830" : "#292f38";
  node.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.add("hidden"), 1800);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function currentDecision(item = state.item) {
  return item ? state.decisions[String(item.qc_index)] || null : null;
}

function updateProgress() {
  const summary = state.summary;
  if (!summary) return;
  const percent = summary.total ? (100 * summary.reviewed / summary.total) : 0;
  $("progress-text").textContent = `${summary.reviewed.toLocaleString()} / ${summary.total.toLocaleString()} reviewed (${percent.toFixed(1)}%) · ${summary.remaining.toLocaleString()} remaining`;
  $("progress-bar").style.width = `${percent}%`;
  const counts = summary.decision_counts;
  $("counts").innerHTML = `KEEP: <b>${counts.KEEP.toLocaleString()}</b><br>FIX: <b>${counts.FIX.toLocaleString()}</b><br>REMOVE: <b>${counts.REMOVE.toLocaleString()}</b>`;
  $("finalize").disabled = !summary.complete;
}

function rebuildBatchFilter() {
  const select = $("batch-filter");
  const previous = select.value;
  const split = $("split-filter").value;
  const batches = [...new Set(state.items.filter(item => split === "all" || item.model_split === split).map(item => item.batch))].sort((a, b) => a - b);
  select.innerHTML = '<option value="all">All</option>' + batches.map(value => `<option value="${value}">${String(value).padStart(4, "0")}</option>`).join("");
  if ([...select.options].some(option => option.value === previous)) select.value = previous;
}

function applyFilters({preserveQcIndex = null} = {}) {
  const split = $("split-filter").value;
  const status = $("status-filter").value;
  const batch = $("batch-filter").value;
  const search = $("search").value.trim().toLowerCase();
  state.filtered = state.items.filter(item => {
    const decision = state.decisions[String(item.qc_index)];
    if (split !== "all" && item.model_split !== split) return false;
    if (batch !== "all" && item.batch !== Number(batch)) return false;
    if (status === "unreviewed" && decision) return false;
    if (["KEEP", "FIX", "REMOVE"].includes(status) && decision?.decision !== status) return false;
    if (status === "interpolated" && !item.interpolated_hint) return false;
    if (search) {
      const haystack = `${item.qc_index} ${item.image_id} ${item.episode} ${item.frame} ${item.sample_id}`.toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  });
  let nextCursor = 0;
  if (preserveQcIndex !== null) {
    const match = state.filtered.findIndex(item => item.qc_index === preserveQcIndex);
    if (match >= 0) nextCursor = match;
  }
  state.cursor = Math.min(nextCursor, Math.max(0, state.filtered.length - 1));
  loadCurrent();
}

function boxFromFields() {
  const values = ["x1", "y1", "x2", "y2"].map(id => Number($(id).value));
  if (values.some(value => !Number.isFinite(value))) return null;
  const [x1, y1, x2, y2] = values;
  if (!(0 <= x1 && x1 < x2 && x2 <= canvas.width && 0 <= y1 && y1 < y2 && y2 <= canvas.height)) return null;
  return values;
}

function fillFields(box) {
  ["x1", "y1", "x2", "y2"].forEach((id, index) => {
    $(id).value = box ? Number(box[index]).toFixed(1) : "";
  });
}

function drawBox(box, color, width, label) {
  if (!box) return;
  const [x1, y1, x2, y2] = box;
  ctx.save();
  ctx.strokeStyle = "#000";
  ctx.lineWidth = width + 4;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.font = "bold 17px sans-serif";
  const textWidth = ctx.measureText(label).width;
  const labelY = Math.max(0, y1 - 24);
  ctx.fillStyle = "#000";
  ctx.fillRect(x1, labelY, textWidth + 12, 24);
  ctx.fillStyle = color;
  ctx.fillText(label, x1 + 5, labelY + 18);
  ctx.restore();
}

function renderCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.image || !state.item) return;
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  drawBox(state.item.bbox_xyxy, "#21ed60", 3, "GT");
  const decision = currentDecision();
  const fixed = state.draftBox || (decision?.decision === "FIX" ? decision.bbox_xyxy : null);
  if (fixed) drawBox(fixed, "#ffad22", 3, "FIX");
}

function updateMetadata() {
  const item = state.item;
  if (!item) {
    $("metadata").innerHTML = "";
    $("position").textContent = `0 / ${state.filtered.length}`;
    return;
  }
  const flags = ["partial", "occluded", "blurred"].filter(name => item[name]);
  $("metadata").innerHTML = `
    <dt>QC index</dt><dd>${item.qc_index} / ${state.items.length}</dd>
    <dt>Filtered</dt><dd>${state.cursor + 1} / ${state.filtered.length}</dd>
    <dt>Split</dt><dd>${item.model_split} · #${item.split_index} · batch ${item.batch}</dd>
    <dt>Episode/frame</dt><dd>${item.episode} / ${item.frame}</dd>
    <dt>Image ID</dt><dd>${item.image_id}</dd>
    <dt>Sample ID</dt><dd>${item.sample_id}</dd>
    <dt>Current GT</dt><dd>${item.bbox_xyxy.map(v => v.toFixed(1)).join(", ")}</dd>
    <dt>Flags</dt><dd>${flags.length ? flags.join(", ") : "-"}</dd>
    <dt>Center jump</dt><dd>${item.normalized_center_jump === null ? "-" : item.normalized_center_jump.toFixed(3)}</dd>`;
  $("position").textContent = `${state.cursor + 1} / ${state.filtered.length}`;
  const warning = $("warning");
  warning.classList.toggle("hidden", !item.interpolated_hint);
  warning.textContent = item.interpolated_hint ? "⚠ Temporal interpolation hint에서 생성된 GT입니다. 특히 주의해서 검수하세요." : "";
  const decision = currentDecision();
  const badge = $("decision-badge");
  badge.textContent = decision?.decision || "UNREVIEWED";
  badge.className = `decision-badge ${(decision?.decision || "unreviewed").toLowerCase()}`;
  $("notes").value = decision?.notes || "";
  fillFields(decision?.decision === "FIX" ? decision.bbox_xyxy : item.bbox_xyxy);
}

async function loadCurrent() {
  state.fixMode = false;
  state.draftBox = null;
  state.dragStart = null;
  canvas.classList.remove("fixing");
  $("save-fix").classList.add("hidden");
  $("cancel-fix").classList.add("hidden");
  if (!state.filtered.length) {
    state.item = null;
    state.image = null;
    $("loading").textContent = "No items match the current filter.";
    $("loading").classList.remove("hidden");
    updateMetadata();
    renderCanvas();
    return;
  }
  state.cursor = Math.max(0, Math.min(state.cursor, state.filtered.length - 1));
  state.item = state.filtered[state.cursor];
  updateMetadata();
  $("loading").textContent = "Loading image…";
  $("loading").classList.remove("hidden");
  const image = new Image();
  image.onload = () => {
    if (state.item && Number(image.dataset.qcIndex) === state.item.qc_index) {
      state.image = image;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      $("loading").classList.add("hidden");
      renderCanvas();
    }
  };
  image.onerror = () => {
    $("loading").textContent = "Could not load image.";
  };
  image.dataset.qcIndex = state.item.qc_index;
  image.src = `/api/image/${state.item.qc_index}`;
}

function navigate(delta) {
  if (!state.filtered.length) return;
  state.cursor = Math.max(0, Math.min(state.filtered.length - 1, state.cursor + delta));
  loadCurrent();
}

function enterFixMode() {
  if (!state.item) return;
  state.fixMode = true;
  state.draftBox = currentDecision()?.decision === "FIX" ? [...currentDecision().bbox_xyxy] : null;
  canvas.classList.add("fixing");
  $("save-fix").classList.remove("hidden");
  $("cancel-fix").classList.remove("hidden");
  fillFields(state.draftBox || state.item.bbox_xyxy);
  renderCanvas();
  toast("Drag a new bbox, then press Enter to save FIX.");
}

function cancelFixMode() {
  state.fixMode = false;
  state.draftBox = null;
  state.dragStart = null;
  canvas.classList.remove("fixing");
  $("save-fix").classList.add("hidden");
  $("cancel-fix").classList.add("hidden");
  updateMetadata();
  renderCanvas();
}

function canvasPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  return [
    Math.max(0, Math.min(canvas.width, (event.clientX - bounds.left) * canvas.width / bounds.width)),
    Math.max(0, Math.min(canvas.height, (event.clientY - bounds.top) * canvas.height / bounds.height)),
  ];
}

async function saveDecision(decision) {
  if (!state.item) return;
  let bbox = null;
  if (decision === "FIX") {
    bbox = state.draftBox || boxFromFields();
    if (!bbox) {
      toast("Draw or enter a valid FIX bbox first.", true);
      return;
    }
  }
  const qcIndex = state.item.qc_index;
  const nextQcIndex = state.filtered[state.cursor + 1]?.qc_index ?? state.filtered[state.cursor - 1]?.qc_index ?? null;
  $("save-status").textContent = "Saving…";
  try {
    const response = await api("/api/decision", {
      method: "POST",
      body: JSON.stringify({qc_index: qcIndex, decision, bbox_xyxy: bbox, notes: $("notes").value}),
    });
    state.decisions[String(qcIndex)] = response.decision;
    state.summary = response.summary;
    updateProgress();
    $("save-status").textContent = "Saved locally";
    const statusFilter = $("status-filter").value;
    if (statusFilter === "unreviewed") {
      applyFilters({preserveQcIndex: nextQcIndex});
    } else {
      updateMetadata();
      renderCanvas();
      navigate(1);
    }
  } catch (error) {
    $("save-status").textContent = "Save failed";
    toast(error.message, true);
  }
}

async function clearDecision() {
  if (!state.item) return;
  const qcIndex = state.item.qc_index;
  try {
    const response = await api("/api/clear", {method: "POST", body: JSON.stringify({qc_index: qcIndex})});
    delete state.decisions[String(qcIndex)];
    state.summary = response.summary;
    updateProgress();
    applyFilters({preserveQcIndex: qcIndex});
    toast("Decision cleared.");
  } catch (error) {
    toast(error.message, true);
  }
}

canvas.addEventListener("pointerdown", event => {
  if (!state.fixMode) return;
  canvas.setPointerCapture(event.pointerId);
  state.dragStart = canvasPoint(event);
  state.draftBox = [state.dragStart[0], state.dragStart[1], state.dragStart[0], state.dragStart[1]];
});
canvas.addEventListener("pointermove", event => {
  if (!state.fixMode || !state.dragStart) return;
  const point = canvasPoint(event);
  state.draftBox = [
    Math.min(state.dragStart[0], point[0]), Math.min(state.dragStart[1], point[1]),
    Math.max(state.dragStart[0], point[0]), Math.max(state.dragStart[1], point[1]),
  ];
  fillFields(state.draftBox);
  renderCanvas();
});
canvas.addEventListener("pointerup", event => {
  if (!state.fixMode || !state.dragStart) return;
  canvas.releasePointerCapture(event.pointerId);
  state.dragStart = null;
  const [x1, y1, x2, y2] = state.draftBox;
  if (x2 - x1 < 2 || y2 - y1 < 2) state.draftBox = null;
  fillFields(state.draftBox);
  renderCanvas();
});

["x1", "y1", "x2", "y2"].forEach(id => $(id).addEventListener("input", () => {
  if (!state.fixMode) return;
  state.draftBox = boxFromFields();
  renderCanvas();
}));

$("previous").addEventListener("click", () => navigate(-1));
$("next").addEventListener("click", () => navigate(1));
$("keep").addEventListener("click", () => saveDecision("KEEP"));
$("fix-mode").addEventListener("click", enterFixMode);
$("save-fix").addEventListener("click", () => saveDecision("FIX"));
$("cancel-fix").addEventListener("click", cancelFixMode);
$("remove").addEventListener("click", () => saveDecision("REMOVE"));
$("clear").addEventListener("click", clearDecision);
$("split-filter").addEventListener("change", () => { rebuildBatchFilter(); applyFilters(); });
$("status-filter").addEventListener("change", () => applyFilters());
$("batch-filter").addEventListener("change", () => applyFilters());
$("search").addEventListener("input", () => applyFilters());
$("jump-unreviewed").addEventListener("click", () => {
  $("status-filter").value = "unreviewed";
  $("search").value = "";
  applyFilters();
});
$("finalize").addEventListener("click", async () => {
  if (!confirm("All 8,204 decisions are present. Mark this review complete?")) return;
  try {
    const result = await api("/api/finalize", {method: "POST", body: "{}"});
    toast(`Review completed at ${result.completed_at}`);
    $("save-status").textContent = "REVIEW_COMPLETE.json created";
  } catch (error) {
    toast(error.message, true);
  }
});

document.addEventListener("keydown", event => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (event.key === "ArrowLeft") navigate(-1);
  else if (event.key === "ArrowRight") navigate(1);
  else if (event.key.toLowerCase() === "k") saveDecision("KEEP");
  else if (event.key.toLowerCase() === "r") saveDecision("REMOVE");
  else if (event.key.toLowerCase() === "f") enterFixMode();
  else if (event.key.toLowerCase() === "u") clearDecision();
  else if (event.key === "Enter" && state.fixMode) saveDecision("FIX");
  else if (event.key === "Escape" && state.fixMode) cancelFixMode();
});

async function initialize() {
  try {
    const payload = await api("/api/state");
    state.items = payload.items;
    state.decisions = payload.decisions;
    state.summary = payload.summary;
    updateProgress();
    rebuildBatchFilter();
    applyFilters();
  } catch (error) {
    $("loading").textContent = `Initialization failed: ${error.message}`;
    toast(error.message, true);
  }
}

initialize();
