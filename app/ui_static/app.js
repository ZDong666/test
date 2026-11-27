const sourceCanvas = document.getElementById("sourceCanvas");
const maskCanvas = document.getElementById("maskCanvas");
const sourceContext = sourceCanvas.getContext("2d");
const maskContext = maskCanvas.getContext("2d");
const fileInput = document.getElementById("fileInput");
const maskFileInput = document.getElementById("maskFileInput");
const uploadZone = document.getElementById("uploadZone");
const fileLabel = document.getElementById("fileLabel");
const maskStatus = document.getElementById("maskStatus");
const uploadMaskField = document.getElementById("uploadMaskField");
const brushSize = document.getElementById("brushSize");
const brushSizeValue = document.getElementById("brushSizeValue");
const clearMaskButton = document.getElementById("clearMask");
const previewButton = document.getElementById("previewButton");
const repairButton = document.getElementById("repairButton");
const errorBox = document.getElementById("errorBox");
const emptyState = document.getElementById("emptyState");
const resultView = document.getElementById("resultView");
const resultImage = document.getElementById("resultImage");
const previewImage = document.getElementById("previewImage");
const maskPreviewImage = document.getElementById("maskPreviewImage");
const resultDownload = document.getElementById("resultDownload");
const configDownload = document.getElementById("configDownload");
const statusText = document.getElementById("statusText");
const modeRadios = Array.from(document.querySelectorAll('input[name="maskMode"]'));

const state = {
  mode: "brush",
  brushSize: Number(brushSize.value),
  hasImage: false,
  drawing: false,
  lastPoint: null,
  maskFile: null,
};

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function drawPlaceholder() {
  sourceContext.fillStyle = "#cccccc";
  sourceContext.fillRect(0, 0, 256, 256);
  sourceContext.fillStyle = "#72878a";
  sourceContext.font = '12px "Microsoft YaHei", sans-serif';
  sourceContext.textAlign = "center";
  sourceContext.fillText("等待原图", 128, 132);
}

function setBusy(busy, label) {
  previewButton.disabled = busy;
  repairButton.disabled = busy;
  repairButton.textContent = busy ? label : "开始修复";
}

function clearMask() {
  maskContext.clearRect(0, 0, 256, 256);
  updateMaskStatus();
}

function updateMaskStatus() {
  const pixels = maskContext.getImageData(0, 0, 256, 256).data;
  let marked = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    const alpha = pixels[index + 3];
    const luminance = (pixels[index] + pixels[index + 1] + pixels[index + 2]) / 3;
    if (state.mode === "upload" ? luminance >= 128 && alpha > 0 : alpha > 0) {
      marked += 1;
    }
  }
  const ratio = marked / (256 * 256);
  maskStatus.textContent = ratio > 0 ? `已标记 ${(ratio * 100).toFixed(1)}%` : "未标记区域";
}

function setMode(mode) {
  state.mode = mode;
  uploadMaskField.hidden = mode !== "upload";
  maskCanvas.classList.toggle("passive", mode === "upload");
  brushSize.disabled = mode === "upload";
  state.maskFile = null;
  maskFileInput.value = "";
  clearMask();
  clearError();
}

function loadSourceFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("请选择 PNG 或 JPG 图片。");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    const image = new Image();
    image.onload = () => {
      sourceContext.clearRect(0, 0, 256, 256);
      sourceContext.drawImage(image, 0, 0, 256, 256);
      state.hasImage = true;
      fileLabel.textContent = file.name;
      state.maskFile = null;
      maskFileInput.value = "";
      clearMask();
      clearError();
    };
    image.onerror = () => showError("图片读取失败，请更换文件后重试。");
    image.src = reader.result;
  };
  reader.onerror = () => showError("图片读取失败，请更换文件后重试。");
  reader.readAsDataURL(file);
}

function loadMaskFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("请选择 PNG 或 JPG 掩码。");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    const image = new Image();
    image.onload = () => {
      maskContext.clearRect(0, 0, 256, 256);
      maskContext.globalCompositeOperation = "source-over";
      maskContext.fillStyle = "#000000";
      maskContext.fillRect(0, 0, 256, 256);
      maskContext.drawImage(image, 0, 0, 256, 256);
      state.maskFile = file;
      updateMaskStatus();
      clearError();
    };
    image.onerror = () => showError("掩码读取失败，请更换文件后重试。");
    image.src = reader.result;
  };
  reader.onerror = () => showError("掩码读取失败，请更换文件后重试。");
  reader.readAsDataURL(file);
}

function pointerPosition(event) {
  const rect = maskCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * 256,
    y: ((event.clientY - rect.top) / rect.height) * 256,
  };
}

function applyContext() {
  maskContext.save();
  maskContext.lineWidth = state.brushSize;
  maskContext.lineCap = "round";
  maskContext.lineJoin = "round";
  maskContext.globalCompositeOperation = "source-over";
  maskContext.fillStyle = "#ffffff";
  maskContext.strokeStyle = "#ffffff";
}

function paintDot(point) {
  applyContext();
  maskContext.beginPath();
  maskContext.arc(point.x, point.y, state.brushSize / 2, 0, Math.PI * 2);
  maskContext.fill();
  maskContext.restore();
}

function paintSegment(from, to) {
  applyContext();
  maskContext.beginPath();
  maskContext.moveTo(from.x, from.y);
  maskContext.lineTo(to.x, to.y);
  maskContext.stroke();
  maskContext.restore();
}

function canvasBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("图片导出失败，请重试。"));
      }
    }, "image/png");
  });
}

async function buildForm() {
  const form = new FormData();
  form.append("image", await canvasBlob(sourceCanvas), "source.png");
  form.append("mask", await canvasBlob(maskCanvas), "mask.png");
  form.append("mode", state.mode);
  form.append("budget", document.getElementById("budgetSelect").value);
  form.append("seed", document.getElementById("seedInput").value);
  return form;
}

async function requestJson(endpoint, form) {
  const response = await fetch(endpoint, { method: "POST", body: form });
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error("服务返回格式异常，请查看终端日志。");
  }
  if (!response.ok) {
    if (Array.isArray(payload.detail)) {
      throw new Error(payload.detail.map((item) => item.msg).join("；"));
    }
    throw new Error(payload.detail || "请求失败，请重试。");
  }
  return payload;
}

function renderResult(payload) {
  const status = payload.status || {};
  resultImage.src = payload.result_url;
  previewImage.src = payload.preview_url;
  maskPreviewImage.src = payload.mask_url;
  resultDownload.href = payload.result_url;
  configDownload.href = payload.config_url;
  const details = [];
  if (typeof status.k === "number") details.push(`K=${status.k}`);
  if (typeof status.nfe === "number") details.push(`NFE=${status.nfe}`);
  if (typeof status.missing_area === "number") details.push(`面积 ${(status.missing_area * 100).toFixed(1)}%`);
  const message = status.message || "修复完成";
  statusText.textContent = details.length ? `${message} · ${details.join(", ")}` : message;
  emptyState.hidden = true;
  resultView.hidden = false;
}

function validateRequest() {
  if (!state.hasImage) {
    showError("请先选择原图。");
    return false;
  }
  if (state.mode === "upload" && !state.maskFile) {
    showError("请上传黑白掩码。");
    return false;
  }
  clearError();
  return true;
}

fileInput.addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) {
    loadSourceFile(file);
  }
});

maskFileInput.addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (file) {
    loadMaskFile(file);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("is-dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("is-dragging");
  });
});

uploadZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) {
    loadSourceFile(file);
  }
});

modeRadios.forEach((radio) => {
  radio.addEventListener("change", () => {
    if (radio.checked) {
      setMode(radio.value);
    }
  });
});

brushSize.addEventListener("input", () => {
  state.brushSize = Number(brushSize.value);
  brushSizeValue.textContent = brushSize.value;
});

clearMaskButton.addEventListener("click", clearMask);

maskCanvas.addEventListener("pointerdown", (event) => {
  if (!state.hasImage || state.mode !== "brush") {
    return;
  }
  event.preventDefault();
  state.drawing = true;
  state.lastPoint = pointerPosition(event);
  maskCanvas.setPointerCapture(event.pointerId);
  paintDot(state.lastPoint);
});

maskCanvas.addEventListener("pointermove", (event) => {
  if (!state.drawing) {
    return;
  }
  event.preventDefault();
  const point = pointerPosition(event);
  paintSegment(state.lastPoint, point);
  state.lastPoint = point;
});

function stopDrawing(event) {
  if (!state.drawing) {
    return;
  }
  state.drawing = false;
  state.lastPoint = null;
  if (event.pointerId !== undefined && maskCanvas.hasPointerCapture(event.pointerId)) {
    maskCanvas.releasePointerCapture(event.pointerId);
  }
  updateMaskStatus();
}

maskCanvas.addEventListener("pointerup", stopDrawing);
maskCanvas.addEventListener("pointercancel", stopDrawing);
maskCanvas.addEventListener("lostpointercapture", stopDrawing);

previewButton.addEventListener("click", async () => {
  if (!validateRequest()) {
    return;
  }
  setBusy(true, "生成预览中…");
  try {
    const payload = await requestJson("/api/preview", await buildForm());
    previewImage.src = payload.preview_data;
    maskPreviewImage.src = payload.mask_data;
    emptyState.hidden = true;
    resultView.hidden = false;
    statusText.textContent = "预览效果";
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
});

repairButton.addEventListener("click", async () => {
  if (!validateRequest()) {
    return;
  }
  setBusy(true, "模型加载与修复中…");
  statusText.textContent = "修复中...";
  try {
    const payload = await requestJson("/api/repair", await buildForm());
    renderResult(payload);
  } catch (error) {
    showError(error.message);
    statusText.textContent = "修复未完成。";
  } finally {
    setBusy(false);
  }
});

drawPlaceholder();
updateMaskStatus();
