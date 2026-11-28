"use strict";

const $ = (id) => document.getElementById(id);
const sourceCanvas = $("sourceCanvas");
const maskCanvas = $("maskCanvas");
const sourceContext = sourceCanvas.getContext("2d");
const overlayContext = maskCanvas.getContext("2d");
// The submitted mask is separate from the translucent blue display layer.
const maskData = document.createElement("canvas");
maskData.width = maskData.height = 256;
const maskContext = maskData.getContext("2d", { willReadFrequently: true });
const state = { mode: "brush", tool: "draw", hasImage: false, hasMask: false,
  busy: false, pointer: null, lastPoint: null, history: [], area: 0, hasResult: false };
const modeInputs = [...document.querySelectorAll('input[name="maskMode"]')];

function showError(message) {
  $("errorBox").textContent = message;
  $("errorBox").hidden = false;
}
function clearError() { $("errorBox").hidden = true; $("errorBox").textContent = ""; }
function status(message, badge) {
  $("statusText").textContent = message;
  if (badge) $("resultBadge").textContent = badge;
}
function updateControls() {
  const locked = state.busy || state.pointer !== null;
  for (const id of ["fileInput", "budgetSelect", "seedInput"]) $(id).disabled = locked;
  modeInputs.forEach((input) => { input.disabled = locked; });
  for (const id of ["brushSize", "drawTool", "eraseTool", "clearMask"]) {
    $(id).disabled = locked || !state.hasImage || state.mode !== "brush";
  }
  $("undoMask").disabled = locked || !state.history.length || state.mode !== "brush";
  $("maskFileInput").disabled = locked || !state.hasImage;
  for (const id of ["previewButton", "repairButton"]) {
    $(id).disabled = locked || !state.hasImage || (state.mode === "upload" && !state.hasMask);
  }
  $("uploadZone").setAttribute("aria-disabled", String(locked));
  maskCanvas.classList.toggle("passive", state.busy || !state.hasImage || state.mode !== "brush");
}
function setBusy(busy, action = "") {
  state.busy = busy;
  $("workspace").setAttribute("aria-busy", String(busy));
  $("statusText").classList.toggle("busy", busy);
  $("previewButton").textContent = busy && action === "preview" ? "生成预览中…" : "预览修复区域";
  $("repairButton").textContent = busy && action === "repair" ? "正在修复…" : "开始修复 →";
  updateControls();
}
function invalidateResult(message = "输入已更新，请重新预览或开始修复。") {
  state.hasResult = false;
  $("emptyState").hidden = false;
  for (const id of ["resultView", "resultFigure", "compareView", "resultControls", "metrics", "downloadRow"]) $(id).hidden = true;
  for (const id of ["resultImage", "previewImage", "maskPreviewImage", "compareBefore", "compareAfter"]) $(id).removeAttribute("src");
  for (const id of ["resultDownload", "configDownload"]) $(id).removeAttribute("href");
  clearError();
  status(message, state.hasImage ? "待修复" : "等待输入");
}
function renderMask() {
  const pixels = maskContext.getImageData(0, 0, 256, 256);
  const overlay = overlayContext.createImageData(256, 256);
  let count = 0;
  for (let i = 0; i < pixels.data.length; i += 4) {
    const missing = state.mode === "brush" ? pixels.data[i + 3] > 0 : pixels.data[i] >= 128;
    if (missing) {
      overlay.data.set([65, 114, 255, 140], i);
      count++;
    }
  }
  overlayContext.putImageData(overlay, 0, 0);
  state.area = count / (256 * 256);
  $("maskStatus").textContent = !state.hasImage ? "尚未上传图片" : count ? `已标记 ${(state.area * 100).toFixed(1)}% 的区域` : "未标记区域 · 将直接返回原图";
}
function saveUndo() {
  state.history.push(maskContext.getImageData(0, 0, 256, 256));
  if (state.history.length > 20) state.history.shift();
}
function resetMask() {
  maskContext.clearRect(0, 0, 256, 256);
  state.hasMask = false;
  state.history = [];
  $("maskFileInput").value = "";
  renderMask();
  updateControls();
}
async function decodeFile(file) {
  if (file.type && !["image/png", "image/jpeg", "image/webp", "image/bmp", "image/x-ms-bmp"].includes(file.type)) {
    throw new Error("请选择 PNG、JPG、WebP 或 BMP 图片。");
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  } catch {
    throw new Error("无法读取这张图片，请检查文件或换一张图片。");
  } finally { URL.revokeObjectURL(url); }
}
async function loadFile(file, asMask = false) {
  if (!file || state.busy || state.pointer !== null) return;
  clearError();
  setBusy(true);
  status(asMask ? "正在读取掩码…" : "正在读取图片…", "读取中");
  try {
    const image = await decodeFile(file);
    if (asMask) {
      maskContext.clearRect(0, 0, 256, 256);
      maskContext.imageSmoothingEnabled = false;
      maskContext.drawImage(image, 0, 0, 256, 256);
      const pixels = maskContext.getImageData(0, 0, 256, 256);
      for (let i = 0; i < pixels.data.length; i += 4) {
        const value = Math.round(.299 * pixels.data[i] + .587 * pixels.data[i + 1] + .114 * pixels.data[i + 2]) >= 128 ? 255 : 0;
        pixels.data.set([value, value, value, 255], i);
      }
      maskContext.putImageData(pixels, 0, 0);
      state.hasMask = true;
      renderMask();
      invalidateResult("掩码已载入，可预览区域或开始修复。");
    } else {
      sourceContext.fillStyle = "#fff";
      sourceContext.fillRect(0, 0, 256, 256);
      sourceContext.drawImage(image, 0, 0, 256, 256);
      state.hasImage = true;
      $("canvasPlaceholder").hidden = true;
      $("uploadTitle").textContent = "图片已载入 · 点击更换";
      $("fileLabel").textContent = `${file.name} · ${image.naturalWidth} × ${image.naturalHeight}`;
      $("fileLabel").title = file.name;
      resetMask();
      invalidateResult(state.mode === "brush" ? "图片已就绪，请在左侧涂抹需要修复的区域。" : "图片已就绪，请选择对应的黑白掩码。");
    }
  } catch (error) {
    showError(error.message);
    status("文件未载入，请重新选择。现有图片和结果仍可使用。", "读取失败");
    // Allow selecting the same file again after a decode failure.
    (asMask ? $("maskFileInput") : $("fileInput")).value = "";
  } finally {
    setBusy(false);
  }
}
function pointerPosition(event) {
  const rect = maskCanvas.getBoundingClientRect();
  return { x: Math.max(0, Math.min(256, (event.clientX - rect.left) * 256 / rect.width)),
    y: Math.max(0, Math.min(256, (event.clientY - rect.top) * 256 / rect.height)) };
}
function drawStroke(point, previous) {
  maskContext.save();
  maskContext.globalCompositeOperation = state.tool === "erase" ? "destination-out" : "source-over";
  maskContext.fillStyle = maskContext.strokeStyle = "#fff";
  maskContext.lineWidth = Number($("brushSize").value);
  maskContext.lineCap = maskContext.lineJoin = "round";
  maskContext.beginPath();
  if (previous) {
    maskContext.moveTo(previous.x, previous.y);
    maskContext.lineTo(point.x, point.y);
    maskContext.stroke();
  } else {
    maskContext.arc(point.x, point.y, maskContext.lineWidth / 2, 0, Math.PI * 2);
    maskContext.fill();
  }
  maskContext.restore();
  renderMask();
}
function stopDrawing(event) {
  if (state.pointer !== event.pointerId) return;
  state.pointer = null;
  state.lastPoint = null;
  updateControls();
}
maskCanvas.addEventListener("pointerdown", (event) => {
  if (!state.hasImage || state.busy || state.mode !== "brush" || state.pointer !== null || event.button !== 0) return;
  event.preventDefault();
  saveUndo();
  invalidateResult();
  state.pointer = event.pointerId;
  state.lastPoint = pointerPosition(event);
  maskCanvas.setPointerCapture(event.pointerId);
  drawStroke(state.lastPoint);
  updateControls();
});
maskCanvas.addEventListener("pointermove", (event) => {
  if (state.pointer !== event.pointerId) return;
  drawStroke(pointerPosition(event), state.lastPoint);
  state.lastPoint = pointerPosition(event);
});
for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) maskCanvas.addEventListener(type, stopDrawing);

$("fileInput").addEventListener("change", (event) => loadFile(event.target.files[0]));
$("maskFileInput").addEventListener("change", (event) => loadFile(event.target.files[0], true));
for (const type of ["dragenter", "dragover"]) $("uploadZone").addEventListener(type, (event) => {
  event.preventDefault();
  if (!state.busy && state.pointer === null) $("uploadZone").classList.add("dragging");
});
$("uploadZone").addEventListener("dragleave", () => $("uploadZone").classList.remove("dragging"));
$("uploadZone").addEventListener("drop", (event) => {
  event.preventDefault();
  $("uploadZone").classList.remove("dragging");
  loadFile(event.dataTransfer.files[0]);
});
// Prevent dropping a file outside the upload zone from navigating away.
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());
modeInputs.forEach((input) => input.addEventListener("change", () => {
  state.mode = input.value;
  $("uploadMaskField").hidden = state.mode !== "upload";
  $("brushToolbar").hidden = state.mode !== "brush";
  resetMask();
  invalidateResult(state.mode === "upload" ? "请选择黑白掩码：白色为修复区，黑色为保留区。" : "请在图片上涂抹需要修复的区域。");
}));
$("brushSize").addEventListener("input", () => { $("brushSizeValue").textContent = $("brushSize").value; });
for (const [id, tool] of [["drawTool", "draw"], ["eraseTool", "erase"]]) $(id).addEventListener("click", () => {
  state.tool = tool;
  $("drawTool").setAttribute("aria-pressed", String(tool === "draw"));
  $("eraseTool").setAttribute("aria-pressed", String(tool === "erase"));
});
$("clearMask").addEventListener("click", () => {
  saveUndo();
  maskContext.clearRect(0, 0, 256, 256);
  renderMask();
  invalidateResult("标记已清空。未标记区域时，将直接返回原图。");
  updateControls();
});
$("undoMask").addEventListener("click", () => {
  if (!state.history.length) return;
  maskContext.putImageData(state.history.pop(), 0, 0);
  renderMask();
  invalidateResult("已撤销上一步标记，请重新预览或开始修复。");
  updateControls();
});
$("budgetSelect").addEventListener("change", () => invalidateResult());
$("seedInput").addEventListener("input", () => invalidateResult());

function canvasBlob(canvas) {
  return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("无法编码图片，请重新上传。")), "image/png"));
}
async function requestJson(endpoint) {
  const form = new FormData();
  form.append("image", await canvasBlob(sourceCanvas), "image.png");
  form.append("mask", await canvasBlob(maskData), "mask.png");
  form.append("mode", state.mode);
  form.append("budget", $("budgetSelect").value);
  form.append("seed", $("seedInput").value);
  let response;
  try { response = await fetch(endpoint, { method: "POST", body: form }); }
  catch { throw new Error("无法连接修复服务，请检查服务是否运行后重试。"); }
  let payload;
  try { payload = await response.json(); }
  catch { throw new Error("服务返回了无法解析的响应，请重试或查看服务日志。"); }
  if (!response.ok) {
    const detail = Array.isArray(payload.detail) ? payload.detail.map((item) => item.msg).join("；") : payload.detail;
    throw new Error(detail || "请求失败，请重试。");
  }
  return payload;
}
function selectResultView(compare) {
  $("resultFigure").hidden = compare;
  $("compareView").hidden = !compare;
  $("resultTab").setAttribute("aria-pressed", String(!compare));
  $("compareTab").setAttribute("aria-pressed", String(compare));
}
function updateComparison() {
  const value = $("compareRange").value;
  $("compareBefore").style.clipPath = `inset(0 ${100 - Number(value)}% 0 0)`;
  $("compareDivider").style.left = `${value}%`;
  $("compareValue").textContent = `${value}%`;
}
$("resultTab").addEventListener("click", () => selectResultView(false));
$("compareTab").addEventListener("click", () => selectResultView(true));
$("compareRange").addEventListener("input", updateComparison);
async function run(action) {
  if (state.busy || state.pointer !== null || !state.hasImage) return;
  clearError();
  if (state.mode === "upload" && !state.hasMask) return showError("请先上传黑白掩码。");
  if (state.area === 1) return showError("请保留一部分原图，不能修复整张全部遮挡的图片。");
  if (action === "repair" && !$("seedInput").checkValidity()) {
    showError("随机种子应为 0 到 4294967295 的整数。");
    $("seedInput").focus();
    return;
  }
  if (action === "repair") invalidateResult();
  setBusy(true, action);
  status(action === "preview" ? "正在生成区域预览…" : "正在加载模型并修复，请稍候。首次运行可能需要更长时间。", action === "preview" ? "预览中" : "修复中");
  try {
    const payload = await requestJson(`/api/${action}`);
    $("previewImage").src = payload.preview_url || payload.preview_data;
    $("maskPreviewImage").src = payload.mask_url || payload.mask_data;
    $("emptyState").hidden = true;
    $("resultView").hidden = false;
    if (action === "repair") {
      state.hasResult = true;
      $("resultImage").src = $("compareAfter").src = payload.result_url;
      $("compareBefore").src = payload.preview_url;
      $("resultDownload").href = payload.result_url;
      $("configDownload").href = payload.config_url;
      for (const id of ["resultControls", "metrics", "downloadRow"]) $(id).hidden = false;
      selectResultView(false);
      $("compareRange").value = "50";
      updateComparison();
      const info = payload.status;
      $("areaMetric").textContent = `${(info.missing_area * 100).toFixed(1)}%`;
      $("timeMetric").textContent = `${info.elapsed_seconds.toFixed(1)} s`;
      $("nfeMetric").textContent = String(info.nfe);
      status(`${info.message} · K=${info.k}`, "已完成");
    } else {
      status(state.hasResult ? "修复结果已保留，区域预览已更新。" : `预览已生成 · 修复面积 ${(payload.missing_area * 100).toFixed(1)}%。确认后点击「开始修复」。`, state.hasResult ? "已完成" : "预览就绪");
    }
  } catch (error) {
    showError(error.message);
    status(action === "preview" ? "预览失败，请检查提示后重试。" : "修复失败，请检查提示后重试。", "需要处理");
  } finally { setBusy(false); }
}
$("previewButton").addEventListener("click", () => run("preview"));
$("repairButton").addEventListener("click", () => run("repair"));
renderMask();
updateControls();
