/* Binaural Assist — front-end controller
 * 只做 UI 状态与 HTTP 调用；实时音频在 Python 后端。
 */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const ICONS = {
  footsteps: `<svg viewBox="0 0 24 24" fill="none"><path d="M8.5 14.5c-1.6 0-2.8-1.6-2.8-3.8 0-2.7 1.2-5.2 3.1-5.2 1.5 0 2.2 1.7 2.2 3.7 0 2.6-1 5.3-2.5 5.3Z" stroke="currentColor" stroke-width="1.6"/><path d="M7.2 17.2c1.2.7 2.6.5 3.4-.4M14.8 9.6c1.4.1 2.5 1.4 2.5 3.1 0 2.1-1.1 3.9-2.7 3.9-1.3 0-1.8-1.2-1.8-2.7 0-1.8.6-3.6 2-4.3Z" stroke="currentColor" stroke-width="1.6"/><path d="M13.6 18.5c1.1.5 2.2.3 2.9-.4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  gunshot: `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="7.2" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="12" r="2.1" stroke="currentColor" stroke-width="1.6"/><path d="M12 2.8v3M12 18.2v3M2.8 12h3M18.2 12h3" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  sliders: `<svg viewBox="0 0 24 24" fill="none"><path d="M4 7h8M16 7h4M4 17h4M12 17h8" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><circle cx="14" cy="7" r="2.2" stroke="currentColor" stroke-width="1.6"/><circle cx="9" cy="17" r="2.2" stroke="currentColor" stroke-width="1.6"/></svg>`,
};

const state = {
  bootstrap: null,
  params: {},
  status: null,
  running: false,
  busy: false,
  pending: {},
  timer: null,
  lastError: "",
  preset: null,
  configs: null,
  configName: "current",
  lastAudio: {},
  games: null,
  gamesTimer: null,
  training: { items: [], stats: {}, labels: {} },
  pollTimer: null,
  shuttingDown: false,
  trainJob: null,
  trainTimer: null,
  classifier: null,
};

/* ------------------------------------------------------------------ utils */
async function fetchJSON(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

let toastTimer = null;
function toast(message, type = "info") {
  const box = $("#toast");
  box.textContent = message;
  box.className = `toast show ${type === "error" ? "error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.className = "toast"; }, 3600);
}

function h(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
}

function getParam(key) {
  return state.params[key];
}

function stepDecimals(step) {
  const s = String(step);
  return s.includes(".") ? s.split(".")[1].length : 0;
}

function formatValue(spec, value) {
  const dec = stepDecimals(spec.step);
  if (typeof value === "number") return value.toFixed(dec);
  return String(value);
}

/* --------------------------------------------------------------- rendering */
function renderDevices(devices) {
  const inputs = devices.inputs || [];
  const outputs = devices.outputs || [];
  const fill = (select, list, preferred, kind) => {
    const oldName = select.selectedOptions[0]?.dataset?.name || state[`${kind}Name`];
    select.innerHTML = "";
    if (!list.length) {
      const opt = new Option("无可用设备（需要 2 声道）", "");
      select.add(opt);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    list.forEach((d) => {
      const opt = new Option(
        `${d.is_default ? "★ " : ""}${d.name} (#${d.index})`,
        String(d.index)
      );
      opt.dataset.name = d.name;
      opt.title = `${d.channels} 声道 · ${d.default_samplerate} Hz · ${d.hostapi}`;
      select.add(opt);
    });

    // 设备索引会随插拔变化，优先按“上次使用的设备名”恢复选择
    const lastAudioName = state.lastAudio?.[`${kind}_name`];
    const wantedName = oldName || lastAudioName;
    const nameMatch = wantedName && list.find((d) => d.name === wantedName);
    if (nameMatch) {
      select.value = String(nameMatch.index);
    } else if (preferred !== undefined && preferred !== null && list.some((d) => String(d.index) === String(preferred))) {
      select.value = String(preferred);
    } else {
      const nameRe =
        kind === "input" ? /(线路|Line)/i : /(DAWN|耳机|Headphone|DAC)/i;
      const fallbackRe =
        kind === "input" ? /(麦克风|Mic)/i : /(耳机|Headphone)/i;
      const pick =
        list.find((d) => nameRe.test(d.name)) ||
        list.find((d) => fallbackRe.test(d.name)) ||
        list.find((d) => d.is_default) ||
        list[0];
      select.value = String(pick.index);
    }
    const selected = select.selectedOptions[0];
    if (selected) state[`${kind}Name`] = selected.dataset.name;
  };
  const inPref = state.params["audio.input_device"];
  const outPref = state.params["audio.output_device"];
  fill($("#inputSelect"), inputs, inPref, "input");
  fill($("#outputSelect"), outputs, outPref, "output");
}

function renderPresets(configs) {
  const wrap = $("#presetChips");
  wrap.innerHTML = "";
  if (!configs || !configs.items) return;
  state.configs = configs;
  state.configName = configs.last || "current";
  configs.items.forEach((item) => {
    const btn = h("button", "chip", item.label);
    btn.dataset.name = item.name;
    btn.title = item.desc || item.label;
    btn.classList.toggle("active", item.name === state.configName);
    btn.addEventListener("click", () => applyPreset(item.name));
    wrap.appendChild(btn);
  });
  const current = configs.items.find((i) => i.name === state.configName);
  $("#presetDesc").textContent = current?.desc || "选择一个预设，或直接微调参数。";
}

async function loadConfigs() {
  try {
    renderPresets(await fetchJSON("/api/configs"));
  } catch (err) {
    /* 后端未就绪时静默 */
  }
}

async function applyPreset(name) {
  try {
    const res = await fetchJSON("/api/configs/load", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    state.params = res.params;
    state.preset = name;
    state.configName = name;
    syncUI();
    await loadConfigs();
    if (res.needs_restart) showRestart(true);
    const label = state.configs?.items.find((i) => i.name === name)?.label || name;
    toast(`已切换配置：${label}`);
  } catch (err) {
    toast(`切换配置失败：${err.message}`, "error");
  }
}

async function savePreset() {
  const name = (prompt("保存为自定义预设，名称：") || "").trim();
  if (!name) return;
  try {
    const res = await fetchJSON("/api/configs/save", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    renderPresets(res);
    toast(`已保存预设：${name}`);
  } catch (err) {
    toast(`保存失败：${err.message}`, "error");
  }
}

async function importPresetFile(file) {
  if (!file) return;
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    const params = data.params && typeof data.params === "object" ? data.params : data;
    const name = (data.name || file.name.replace(/\.json$/i, "") || "导入配置").trim();
    const res = await fetchJSON("/api/configs/import", {
      method: "POST",
      body: JSON.stringify({ name, params }),
    });
    state.params = res.params;
    state.configName = res.name;
    syncUI();
    renderPresets(res);
    toast(`已导入配置：${res.name}`);
  } catch (err) {
    toast(`导入失败：${err.message}`, "error");
  }
}

function renderGroups(groups) {
  const host = $("#paramGroups");
  host.innerHTML = "";
  groups.forEach((group, index) => {
    const card = h("article", "card");
    card.dataset.accent = group.accent || "cyan";
    card.style.animationDelay = `${index * 45}ms`;

    const head = h("div", "card-head");
    const icon = h("div", "card-icon");
    icon.innerHTML = ICONS[group.icon] || ICONS.sliders;
    const title = h("div", "card-title");
    title.appendChild(h("h3", "", group.title));
    title.appendChild(h("p", "", group.subtitle || ""));
    head.append(icon, title);

    if (group.toggle) {
      const toggle = h("div", "card-toggle");
      const label = h("label", "switch");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.dataset.toggle = group.toggle;
      input.checked = getParam(group.toggle) !== false;
      input.addEventListener("change", () => {
        queuePatch(group.toggle, input.checked);
      });
      const track = h("span", "switch-track");
      track.appendChild(h("span", "switch-thumb"));
      label.append(input, track);
      toggle.appendChild(label);
      head.appendChild(toggle);
    }

    const params = h("div", "params");
    group.params.forEach((spec) => {
      const row = h("div", "param");
      row.dataset.key = spec.key;

      const rowHead = h("div", "param-head");
      const label = h("label", "", spec.label);
      const value = h("span", "param-value");
      rowHead.append(label, value);

      const input = document.createElement("input");
      input.type = "range";
      input.min = spec.min;
      input.max = spec.max;
      input.step = spec.step;
      input.value = getParam(spec.key);
      input.dataset.range = spec.key;
      input.addEventListener("input", () => {
        const v = Number(input.value);
        updateParamVisual(row, spec, v);
        queuePatch(spec.key, v);
      });

      const desc = h("p", "param-desc", spec.desc);

      row.append(rowHead, input, desc);
      params.appendChild(row);
      updateParamVisual(row, spec, Number(getParam(spec.key)));
    });

    card.append(head, params);
    host.appendChild(card);
  });
}

function updateParamVisual(row, spec, value) {
  const valueEl = $(".param-value", row);
  valueEl.innerHTML = "";
  valueEl.append(document.createTextNode(formatValue(spec, value)));
  if (spec.unit) {
    const small = h("small", "", spec.unit);
    valueEl.appendChild(small);
  }
  const input = $("input[type=range]", row);
  const min = Number(spec.min);
  const max = Number(spec.max);
  const fill = max > min ? ((value - min) / (max - min)) * 100 : 0;
  input.style.setProperty("--fill", `${Math.max(0, Math.min(100, fill))}%`);
}

/* ------------------------------------------------------------------ params */
function queuePatch(key, value) {
  state.params[key] = value;
  state.configName = "current";
  $$("#presetChips .chip").forEach((chip) => chip.classList.toggle("active", chip.dataset.name === "current"));
  state.pending[key] = value;
  clearTimeout(state.timer);
  state.timer = setTimeout(flushPatch, 140);
}

async function flushPatch() {
  if (!Object.keys(state.pending).length) return;
  const patch = state.pending;
  state.pending = {};
  try {
    const res = await fetchJSON("/api/params", {
      method: "POST",
      body: JSON.stringify({ patch }),
    });
    if (res.needs_restart) showRestart(true);
  } catch (err) {
    toast(`参数保存失败：${err.message}`, "error");
  }
}

function syncUI() {
  state.params["audio.bypass"] = Boolean(state.params["audio.bypass"]);
  $$("[data-range]").forEach((input) => {
    const key = input.dataset.range;
    const value = getParam(key);
    if (value === undefined) return;
    input.value = value;
    const row = input.closest(".param");
    const group = state.bootstrap.groups.find((g) => g.params.some((p) => p.key === key));
    const spec = group.params.find((p) => p.key === key);
    updateParamVisual(row, spec, Number(value));
  });
  $$("[data-toggle]").forEach((input) => {
    const value = getParam(input.dataset.toggle);
    input.checked = value !== false;
  });
  const block = getParam("audio.block_size");
  if (block !== undefined) $("#blockSelect").value = String(block);
  const bypass = Boolean(getParam("audio.bypass"));
  $("#bypassBtn").classList.toggle("active", bypass);
  $("#bypassBtn").lastChild.textContent = bypass ? " 直通中" : " 直通 A/B";
}

/* ------------------------------------------------------------------ status */
function updateStatus(status) {
  if (!status) return;
  state.status = status;
  state.running = Boolean(status.running);

  const pill = $("#statusPill");
  pill.className = "status-pill";
  if (status.running) pill.classList.add("running");
  if (status.error) pill.classList.add("error");
  $("#statusText").textContent = status.error
    ? "运行异常"
    : status.running
      ? "运行中"
      : "未运行";

  $("#startLabel").textContent = status.running ? "停止音频" : "启动音频";
  $("#startBtn").classList.toggle("stopping", status.running);
  $("#startBtn").querySelector(".play-icon").textContent = status.running ? "■" : "▶";

  const meters = status.meters || {};
  const setMeter = (id, db) => {
    const el = $(id);
    const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
    el.style.width = `${pct}%`;
  };
  setMeter("#meterL", Number(meters.rms_l_db ?? -120));
  setMeter("#meterR", Number(meters.rms_r_db ?? -120));
  const fmtDb = (db) => (db <= -100 ? "-∞" : `${Number(db).toFixed(1)}`);
  $("#dbL").textContent = fmtDb(Number(meters.rms_l_db ?? -120));
  $("#dbR").textContent = fmtDb(Number(meters.rms_r_db ?? -120));

  const foot = Math.max(0, Math.min(1, Number(meters.foot ?? 0)));
  const gun = Math.max(0, Math.min(1, Number(meters.gun ?? 0)));
  $("#footRing").style.setProperty("--value", (foot * 100).toFixed(1));
  $("#gunRing").style.setProperty("--value", (gun * 100).toFixed(1));
  $("#footValue").textContent = `${Math.round(foot * 100)}%`;
  $("#gunValue").textContent = `${Math.round(gun * 100)}%`;

  const corr = Math.max(-1, Math.min(1, Number(meters.corr ?? 0)));
  $("#corrValue").textContent = `${corr >= 0 ? "+" : ""}${corr.toFixed(2)}`;
  const bar = $("#corrBar");
  const width = Math.abs(corr) * 50;
  bar.style.width = `${width}%`;
  bar.style.left = corr >= 0 ? "50%" : `${50 - width}%`;
  const monoWarning = corr > 0.985 && Number(meters.rms_l_db) > -45;
  bar.style.background = monoWarning ? "var(--red)" : "var(--cyan)";
  $("#corrHint").textContent = monoWarning
    ? "⚠ 左右几乎相同：输入可能被单声道化，方向增强无效"
    : "接近 +1 时请检查输入是否被单声道化";

  const lat = status.latency || {};
  if (status.running) {
    const total = Number(lat.total_ms || 0);
    $("#latencyValue").textContent = `${total.toFixed(1)} ms`;
    $("#heroLatency").textContent = `${Math.round(total)} ms`;
    const pct = Math.max(4, Math.min(100, (total / 70) * 100));
    $("#latencyBar").style.width = `${pct}%`;
    $("#latencyHint").textContent = `in ${Number(lat.input_ms).toFixed(1)} · out ${Number(lat.output_ms).toFixed(1)} · block ${Number(lat.block_ms).toFixed(1)} ms`;
  } else {
    $("#latencyValue").textContent = "-- ms";
    $("#heroLatency").textContent = "--";
    $("#latencyBar").style.width = "0%";
    $("#latencyHint").textContent = "启动后显示 WASAPI 缓冲延迟";
  }

  if (status.running) {
    state.params["audio.block_size"] = status.block_size;
  } else {
    const block = getParam("audio.block_size");
    if (block !== undefined) $("#blockSelect").value = String(block);
  }
  const xruns = Number(status.xruns ?? 0);
  $("#xrunTag").textContent = `xrun ${xruns}`;
  $("#xrunTag").classList.toggle("warn", xruns > 0);
  if (status.classifier) renderClassifier(status.classifier);
  showRestart(Boolean(status.needs_restart));

  if (status.error && status.error !== state.lastError) {
    state.lastError = status.error;
    toast(status.error, "error");
  }
  if (!status.error) state.lastError = "";
}

function showRestart(show) {
  $("#restartBanner").classList.toggle("hidden", !show);
}

async function pollStatus() {
  if (state.shuttingDown) return;
  try {
    const status = await fetchJSON("/api/status");
    updateStatus(status);
  } catch (err) {
    /* 后端关闭时静默；下一轮继续 */
  }
}

/* ------------------------------------------------------------------ actions */
async function toggleStart() {
  if (state.busy) return;
  const startBtn = $("#startBtn");
  state.busy = true;
  startBtn.disabled = true;
  try {
    if (state.running) {
      await fetchJSON("/api/stop", { method: "POST", body: "{}" });
      toast("音频已停止");
    } else {
      // 插拔设备后索引会变，启动前重新拉一次并尽量按设备名恢复选择
      renderDevices(await fetchJSON("/api/devices"));
      const payload = {
        input_index: Number($("#inputSelect").value),
        output_index: Number($("#outputSelect").value),
        block_size: Number($("#blockSelect").value),
      };
      if (!$("#inputSelect").value || !$("#outputSelect").value) {
        throw new Error("请选择输入和输出设备");
      }
      const res = await fetchJSON("/api/start", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      toast("音频已启动（WASAPI 共享）");
    }
    const status = await fetchJSON("/api/status");
    updateStatus(status);
  } catch (err) {
    toast(err.message, "error");
  } finally {
    state.busy = false;
    startBtn.disabled = false;
  }
}

async function toggleBypass() {
  const next = !Boolean(state.params["audio.bypass"]);
  state.params["audio.bypass"] = next;
  $("#bypassBtn").classList.toggle("active", next);
  $("#bypassBtn").lastChild.textContent = next ? " 直通中" : " 直通 A/B";
  try {
    await fetchJSON("/api/params", {
      method: "POST",
      body: JSON.stringify({ patch: { "audio.bypass": next } }),
    });
    toast(next ? "已切到直通（A/B 对比）" : "已恢复处理");
  } catch (err) {
    toast(`切换失败：${err.message}`, "error");
  }
}

async function resetAll() {
  try {
    const res = await fetchJSON("/api/reset", { method: "POST", body: "{}" });
    state.params = res.params;
    state.preset = null;
    state.configName = "current";
    syncUI();
    await loadConfigs();
    toast("已恢复默认参数");
  } catch (err) {
    toast(`恢复失败：${err.message}`, "error");
  }
}

function exportConfig() {
  const blob = new Blob([JSON.stringify(state.params, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "binaural-assist-config.json";
  a.click();
  URL.revokeObjectURL(url);
  toast("配置已导出");
}

async function refreshDevices() {
  try {
    const devices = await fetchJSON("/api/devices");
    renderDevices(devices);
    toast("设备列表已刷新");
  } catch (err) {
    toast(`刷新失败：${err.message}`, "error");
  }
}

/* ------------------------------------------------------------------ training */
const LABEL_FALLBACK = {
  gunshot: { zh: "枪声" },
  footstep: { zh: "脚步" },
  other: { zh: "其他" },
  background: { zh: "背景" },
  unlabeled: { zh: "待标注" },
};

function labelName(key) {
  return (state.training.labels[key] || LABEL_FALLBACK[key] || {}).zh || key;
}

function suggestLabel(det) {
  const gun = Number(det?.gun || 0);
  const foot = Number(det?.foot || 0);
  if (gun >= 0.35 && gun >= foot * 0.8) return "gunshot";
  if (foot >= 0.35) return "footstep";
  if (gun < 0.05 && foot < 0.05) return "background";
  return "unlabeled";
}

async function loadTraining() {
  try {
    const data = await fetchJSON("/api/train");
    state.training = data;
    renderTraining();
  } catch (err) {
    /* 后端旧版本时静默 */
  }
}

function renderTraining() {
  const data = state.training;
  const items = data.items || [];
  const stats = data.stats || {};
  const count = items.length;
  $("#trainCount").textContent = String(count);
  $("#trainStats").textContent = `${count} clips`;
  const list = $("#clipList");
  list.innerHTML = "";
  if (!items.length) {
    list.appendChild(h("div", "empty-clips", "还没有训练素材。启动音频后截取第一段。"));
    return;
  }
  items.forEach((item) => {
    const row = h("div", "clip-row");
    row.dataset.id = item.id;

    const meta = h("div", "clip-meta");
    const title = h("b", "", labelName(item.label));
    if (item.detector) {
      const suggested = suggestLabel(item.detector);
      if (suggested !== "unlabeled" && suggested !== item.label) {
        title.appendChild(h("span", "clip-suggestion", `建议：${labelName(suggested)}`));
      }
    }
    const sub = h("span", "", `${item.created_at?.replace("T", " ") || ""} · ${item.duration}s · ${item.sample_rate} Hz`);
    meta.append(title, sub);

    const audioWrap = h("div", "clip-audio");
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = `/api/train/audio/${encodeURIComponent(item.id)}`;
    audioWrap.appendChild(audio);

    const actions = h("div", "clip-actions");
    ["gunshot", "footstep", "other", "background"].forEach((key) => {
      const btn = h("button", "clip-label", labelName(key));
      btn.classList.toggle("active", item.label === key);
      btn.addEventListener("click", () => relabelClip(item.id, key));
      actions.appendChild(btn);
    });
    const del = h("button", "clip-delete", "✕");
    del.title = "删除这段素材";
    del.addEventListener("click", () => deleteClip(item.id));
    actions.appendChild(del);

    row.append(meta, audioWrap, actions);
    list.appendChild(row);
  });
}

async function captureClip(label = "unlabeled") {
  const seconds = Number($("#captureSeconds").value || 1);
  $("#captureBtn").disabled = true;
  try {
    const res = await fetchJSON("/api/train/capture", {
      method: "POST",
      body: JSON.stringify({ label, seconds }),
    });
    state.training.items = [res.item, ...(state.training.items || [])];
    if (res.stats) state.training.stats = res.stats;
    renderTraining();
    const suggest = res.suggested && res.suggested !== "unlabeled" ? `，检测器建议：${labelName(res.suggested)}` : "";
    toast(`已保存 ${res.item.duration}s${suggest}`);
    setTab("training");
  } catch (err) {
    toast(`截取失败：${err.message}`, "error");
  } finally {
    $("#captureBtn").disabled = false;
  }
}

async function relabelClip(id, label) {
  try {
    const res = await fetchJSON("/api/train/label", {
      method: "POST",
      body: JSON.stringify({ id, label }),
    });
    const item = state.training.items.find((x) => x.id === id);
    if (item) Object.assign(item, res.item);
    if (res.stats) state.training.stats = res.stats;
    renderTraining();
    toast(`已标记为：${labelName(label)}`);
  } catch (err) {
    toast(`标记失败：${err.message}`, "error");
  }
}

async function deleteClip(id) {
  try {
    await fetchJSON("/api/train/delete", {
      method: "POST",
      body: JSON.stringify({ id }),
    });
    state.training.items = state.training.items.filter((x) => x.id !== id);
    renderTraining();
    toast("已删除");
  } catch (err) {
    toast(`删除失败：${err.message}`, "error");
  }
}

async function exportTraining() {
  try {
    const res = await fetchJSON("/api/train/export", { method: "POST", body: "{}" });
    toast(`已导出 ${res.rows} 条到 ${res.path}`);
  } catch (err) {
    toast(`导出失败：${err.message}`, "error");
  }
}

/* --------------------------------------------------------------- model train */
async function refreshTrainJob() {
  try {
    const wasRunning = Boolean(state.trainJob?.running);
    const job = await fetchJSON("/api/train/job");
    state.trainJob = job;
    const log = $("#trainLog");
    log.textContent = (job.log || []).join("\n") || "训练日志会显示在这里。";
    log.scrollTop = log.scrollHeight;
    const running = Boolean(job.running);
    $("#trainStartBtn").disabled = running;
    $("#trainStopBtn").disabled = !running;
    if (running) {
      $("#trainStartBtn").textContent = "训练中…";
      clearTimeout(state.trainTimer);
      state.trainTimer = setTimeout(refreshTrainJob, 1000);
    } else {
      $("#trainStartBtn").textContent = "开始训练";
      if (wasRunning && job.returncode === 0) {
        toast(`训练完成${job.checkpoint ? "，模型已就绪" : ""}`);
        loadClassifier();
      } else if (wasRunning && job.returncode !== null && job.returncode !== 0) {
        toast(`训练失败：${job.error || "退出码 " + job.returncode}`, "error");
      }
    }
  } catch (err) {
    /* ignore */
  }
}

async function startTrain() {
  const device = $("#trainDevice").value;
  const epochs = Number($("#trainEpochs").value || 20);
  const batch = Number($("#trainBatch").value || 16);
  try {
    const res = await fetchJSON("/api/train/start", {
      method: "POST",
      body: JSON.stringify({
        device,
        epochs,
        batch_size: batch,
        auto_replace: $("#autoReplace").checked,
      }),
    });
    state.trainJob = res;
    toast(`训练已启动（${device.toUpperCase()}）`);
    refreshTrainJob();
  } catch (err) {
    toast(`启动训练失败：${err.message}`, "error");
  }
}

async function stopTrain() {
  try {
    const res = await fetchJSON("/api/train/stop", { method: "POST", body: "{}" });
    state.trainJob = res;
    toast("已停止训练");
    refreshTrainJob();
  } catch (err) {
    toast(`停止失败：${err.message}`, "error");
  }
}

function renderClassifier(c) {
  state.classifier = c;
  const loaded = Boolean(c && c.loaded);
  const enabled = Boolean(c && c.enabled);
  $("#classifierToggle").checked = loaded && enabled;
  const status = $("#modelStatus");
  status.textContent = enabled ? "模型已启用" : loaded ? "模型已加载" : "未加载";
  status.classList.toggle("warn", false);
  if (loaded) {
    const name = String(c.checkpoint || "").split(/[\\/]/).pop();
    $("#classifierInfo").textContent =
      `${name} · foot ${(Number(c.foot || 0) * 100).toFixed(0)}% · gun ${(Number(c.gun || 0) * 100).toFixed(0)}%`;
  } else {
    $("#classifierInfo").textContent =
      "加载训练好的 best.pt 后，用 2ch 模型替换启发式事件检测；DSP 不变。";
  }
}

async function loadClassifier() {
  try {
    renderClassifier(await fetchJSON("/api/classifier"));
  } catch (err) {
    /* ignore */
  }
}

async function toggleClassifier() {
  const enabled = $("#classifierToggle").checked;
  try {
    const c = await fetchJSON("/api/classifier", {
      method: "POST",
      body: JSON.stringify({ enabled }),
    });
    renderClassifier(c);
    toast(enabled ? "实时检测器已切换到模型" : "实时检测器已切回启发式");
  } catch (err) {
    $("#classifierToggle").checked = false;
    toast(`切换失败：${err.message}`, "error");
  }
}

/* --------------------------------------------------------------- game list */
function gameIconEl(item) {
  if (item.icon) {
    const img = document.createElement("img");
    img.className = "game-icon";
    img.alt = item.display_name;
    img.src = `/api/games/icon/${encodeURIComponent(item.id)}?t=${Date.now()}`;
    img.addEventListener("error", () => {
      const fallback = document.createElement("div");
      fallback.className = "game-icon-fallback";
      fallback.textContent = "🎮";
      img.replaceWith(fallback);
    });
    return img;
  }
  const fallback = document.createElement("div");
  fallback.className = "game-icon-fallback";
  fallback.textContent = "🎮";
  return fallback;
}

function renderGames(payload) {
  state.games = payload;
  const items = payload.items || [];
  $("#gameCount").textContent = String(items.length);
  $("#gameAutoStatus").textContent = payload.auto_enabled ? "自动启动：开启" : "自动启动：关闭";
  const detected = Number(payload.detected_count || 0);
  const status = $("#gameMonitorStatus");
  if (payload.last_error) {
    status.textContent = `检测异常：${payload.last_error}`;
    status.classList.remove("live");
  } else if (detected > 0) {
    status.textContent = `已检测到 ${detected} 个白名单游戏`;
    status.classList.add("live");
  } else {
    status.textContent = "未检测到白名单游戏";
    status.classList.remove("live");
  }

  const wrap = $("#gameList");
  wrap.innerHTML = "";
  if (!items.length) {
    const empty = h("div", "empty-clips", "还没有游戏。点击“浏览 EXE 添加”。");
    wrap.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const row = h("div", `game-row${item.detected ? " detected" : ""}`);
    row.dataset.id = item.id;
    row.appendChild(gameIconEl(item));

    const info = h("div", "game-info");
    const titleLine = h("div", "game-title-line");
    titleLine.appendChild(h("span", "game-title", item.display_name || item.exe_name));
    if (item.detected) titleLine.appendChild(h("span", "game-detected", "运行中"));
    info.appendChild(titleLine);
    info.appendChild(h("div", "game-exe", item.exe_name));
    info.appendChild(h("div", "game-path", item.path));
    row.appendChild(info);

    const actions = h("div", "game-actions");
    const toggle = document.createElement("label");
    toggle.className = "switch";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = item.enabled;
    check.title = "启用/禁用该游戏";
    check.addEventListener("change", () => updateGame(item.id, { enabled: check.checked }));
    const track = h("span", "switch-track");
    track.appendChild(h("span", "switch-thumb"));
    toggle.append(check, track);
    actions.appendChild(toggle);

    const rename = h("button", "btn tiny ghost", "改名");
    rename.addEventListener("click", () => renameGame(item));
    actions.appendChild(rename);

    const iconFile = document.createElement("input");
    iconFile.type = "file";
    iconFile.accept = "image/png,image/jpeg,image/webp,image/gif,image/bmp";
    iconFile.hidden = true;
    iconFile.addEventListener("change", () => {
      const file = iconFile.files?.[0];
      if (file) changeGameIcon(item.id, file);
      iconFile.value = "";
    });
    const iconBtn = h("button", "btn tiny ghost", "换图标");
    iconBtn.addEventListener("click", () => iconFile.click());
    actions.append(iconFile, iconBtn);

    const del = h("button", "btn tiny ghost danger", "删除");
    del.addEventListener("click", () => deleteGame(item));
    actions.appendChild(del);
    row.appendChild(actions);
    wrap.appendChild(row);
  });
}

async function loadGames() {
  try {
    renderGames(await fetchJSON("/api/games"));
  } catch (err) {
    /* 服务启动中静默 */
  }
}

async function addGame(path) {
  const value = String(path || "").trim();
  if (!value) {
    toast("请先选择或输入 exe 路径", "error");
    return;
  }
  try {
    const res = await fetchJSON("/api/games/add", {
      method: "POST",
      body: JSON.stringify({ path: value }),
    });
    renderGames(res);
    $("#gamePathInput").value = "";
    toast("已添加游戏进程");
  } catch (err) {
    toast(`添加失败：${err.message}`, "error");
  }
}

async function browseGame() {
  try {
    const res = await fetch("/api/games/browse", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const data = await res.json().catch(() => ({}));
    if (data.cancelled) return;
    if (!res.ok || !data.ok) throw new Error(data.error || "无法打开文件选择框");
    await addGame(data.path);
  } catch (err) {
    toast(`浏览失败：${err.message}`, "error");
  }
}

async function updateGame(id, patch) {
  try {
    const res = await fetchJSON("/api/games/update", {
      method: "POST",
      body: JSON.stringify({ id, ...patch }),
    });
    renderGames(res);
  } catch (err) {
    toast(`更新失败：${err.message}`, "error");
    loadGames();
  }
}

async function renameGame(item) {
  const name = prompt("游戏显示名称：", item.display_name || item.exe_name);
  if (name === null) return;
  if (!name.trim()) {
    toast("显示名称不能为空", "error");
    return;
  }
  await updateGame(item.id, { display_name: name.trim() });
}

async function changeGameIcon(id, file) {
  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`/api/games/icon/${encodeURIComponent(id)}`, {
      method: "POST",
      body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) throw new Error(data.error || "上传失败");
    renderGames(data);
    toast("图标已更新");
  } catch (err) {
    toast(`图标更新失败：${err.message}`, "error");
  }
}

async function deleteGame(item) {
  if (!confirm(`从游戏列表删除“${item.display_name || item.exe_name}”？`)) return;
  try {
    const res = await fetchJSON("/api/games/delete", {
      method: "POST",
      body: JSON.stringify({ id: item.id }),
    });
    renderGames(res);
    toast("已删除");
  } catch (err) {
    toast(`删除失败：${err.message}`, "error");
  }
}

function setTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".tab-pane").forEach((p) => p.classList.toggle("hidden", p.id !== `${name}Pane`));
  clearInterval(state.gamesTimer);
  state.gamesTimer = null;
  if (name === "training") {
    loadTraining();
    refreshTrainJob();
    loadClassifier();
  }
  if (name === "games") {
    loadGames();
    state.gamesTimer = setInterval(loadGames, 3000);
  }
}

function initTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => setTab(tab.dataset.tab));
  });
  $("#captureBtn").addEventListener("click", () => captureClip("unlabeled"));
  $("#exportTrainBtn").addEventListener("click", exportTraining);
  $("#gameBrowseBtn").addEventListener("click", browseGame);
  $("#gameAddPathBtn").addEventListener("click", () => addGame($("#gamePathInput").value));
  $("#gamePathInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") addGame($("#gamePathInput").value);
  });
  $$(".quick-label").forEach((btn) => {
    btn.addEventListener("click", () => captureClip(btn.dataset.label));
  });
  $("#captureSeconds").addEventListener("change", () => {
    const s = $("#captureSeconds").value;
    $("#captureBtn").textContent = `截取最近 ${s} s`;
  });
}

/* ------------------------------------------------------------------ shutdown */
function openShutdownModal() {
  $("#shutdownModal").classList.remove("hidden");
  $("#shutdownCancel").focus();
}

function closeShutdownModal() {
  $("#shutdownModal").classList.add("hidden");
}

async function confirmShutdown() {
  if (state.shuttingDown) return;
  state.shuttingDown = true;
  closeShutdownModal();
  clearInterval(state.pollTimer);
  $("#shutdownScreen").classList.remove("hidden");
  try {
    await fetch("/api/shutdown", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
  } catch (err) {
    /* 进程可能在响应前退出，这是预期行为 */
  }
}

/* -------------------------------------------------------------------- init */
async function init() {
  bindStaticControls();
  try {
    const bootstrap = await fetchJSON("/api/bootstrap");
    state.bootstrap = bootstrap;
    state.params = bootstrap.params;
    state.lastAudio = bootstrap.last_audio || {};
    renderDevices(bootstrap.devices);
    renderPresets(bootstrap.configs || { items: [] });
    renderGroups(bootstrap.groups);
    syncUI();
    initTabs();
    if (!bootstrap.capabilities?.cuda) {
      const cudaOpt = $("#trainDevice option[value='cuda']");
      if (cudaOpt) {
        cudaOpt.disabled = true;
        cudaOpt.textContent = "CUDA（NVIDIA · 当前不可用）";
      }
    }
    loadConfigs();
    loadTraining();
    refreshTrainJob();
    loadClassifier();
    updateStatus(bootstrap.status);
    state.pollTimer = setInterval(pollStatus, 250);
  } catch (err) {
    toast(`无法连接后端：${err.message}`, "error");
  }
}

function bindStaticControls() {
  $("#startBtn").addEventListener("click", toggleStart);
  $("#bypassBtn").addEventListener("click", toggleBypass);
  $("#resetBtn").addEventListener("click", resetAll);
  $("#exportBtn").addEventListener("click", exportConfig);
  $("#savePresetBtn").addEventListener("click", savePreset);
  $("#importPresetBtn").addEventListener("click", () => $("#importPresetFile").click());
  $("#importPresetFile").addEventListener("change", (e) => {
    importPresetFile(e.target.files?.[0]);
    e.target.value = "";
  });
  $("#refreshDevices").addEventListener("click", refreshDevices);
  $("#trainStartBtn").addEventListener("click", startTrain);
  $("#trainStopBtn").addEventListener("click", stopTrain);
  $("#classifierToggle").addEventListener("change", toggleClassifier);
  $("#shutdownBtn").addEventListener("click", openShutdownModal);
  $("#shutdownCancel").addEventListener("click", closeShutdownModal);
  $("#shutdownConfirm").addEventListener("click", confirmShutdown);
  $("#shutdownModal").addEventListener("click", (e) => {
    if (e.target.id === "shutdownModal") closeShutdownModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeShutdownModal();
  });

  const structural = (key, value) => {
    state.params[key] = value;
    queuePatch(key, value);
    showRestart(true);
  };
  $("#blockSelect").addEventListener("change", (e) => structural("audio.block_size", Number(e.target.value)));
  $("#inputSelect").addEventListener("change", (e) => {
    state.inputName = e.target.selectedOptions[0]?.dataset?.name || "";
    structural("audio.input_device", Number(e.target.value));
  });
  $("#outputSelect").addEventListener("change", (e) => {
    state.outputName = e.target.selectedOptions[0]?.dataset?.name || "";
    structural("audio.output_device", Number(e.target.value));
  });
}

document.addEventListener("DOMContentLoaded", init);
