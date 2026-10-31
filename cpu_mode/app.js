"use strict";
const $ = (id) => document.getElementById(id);
let previousAudioUrl = null;

async function updateHealth() {
  try {
    const response = await fetch("/api/backend-health", { cache: "no-store" });
    const data = await response.json();
    const ready = response.ok && data.status === "ok" && data.device === "cpu";
    $("health").textContent = ready ? "آماده · CPU" : "Backend آماده نیست";
    $("health").classList.toggle("ready", ready);
    $("generate").disabled = !ready;
  } catch (_) {
    $("health").textContent = "سرویس صوت در دسترس نیست";
    $("generate").disabled = true;
  }
}

function wavFromBase64(encoded) {
  const binary = atob(encoded);
  const chunks = [];
  const size = 32 * 1024;
  for (let start = 0; start < binary.length; start += size) {
    const portion = binary.slice(start, start + size);
    const array = new Uint8Array(portion.length);
    for (let i = 0; i < portion.length; i++) array[i] = portion.charCodeAt(i);
    chunks.push(array);
  }
  return new Blob(chunks, { type: "audio/wav" });
}

async function generate() {
  const text = $("speech").value.trim();
  const frames = Number($("frames").value);
  const file = $("reference").files[0] || null;
  if (!text || text.length > 1200) {
    $("message").textContent = "متن گفتار باید بین ۱ تا ۱۲۰۰ نویسه باشد.";
    return;
  }
  if (!Number.isInteger(frames) || frames < 20 || frames > 375) {
    $("message").textContent = "مقدار فریم باید بین ۲۰ تا ۳۷۵ باشد.";
    return;
  }
  if (file && file.size > 20 * 1024 * 1024) {
    $("message").textContent = "فایل مرجع باید حداکثر ۲۰ MiB باشد.";
    return;
  }
  const form = new FormData();
  form.append("text", text);
  form.append("demo_id", $("demo").value);
  form.append("cpu_threads", "8");
  form.append("max_new_frames", String(frames));
  form.append("enable_text_normalization", "0");
  form.append("enable_normalize_tts_text", "1");
  form.append("do_sample", "0");
  if (file) form.append("prompt_audio", file, file.name);
  $("generate").disabled = true;
  $("message").textContent = "در حال تولید گفتار روی پردازنده...";
  $("details").textContent = "";
  const started = performance.now();
  try {
    const response = await fetch("/api/generate", { method: "POST", body: form, cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "خطا در تولید گفتار");
    if (typeof data.audio_base64 !== "string" || !data.audio_base64) {
      throw new Error("خروجی صوت معتبر دریافت نشد.");
    }
    if (previousAudioUrl) URL.revokeObjectURL(previousAudioUrl);
    previousAudioUrl = URL.createObjectURL(wavFromBase64(data.audio_base64));
    $("player").src = previousAudioUrl;
    $("download").href = previousAudioUrl;
    $("download").hidden = false;
    $("message").textContent = "تولید صوت کامل شد.";
    $("details").textContent = "نرخ نمونه‌برداری: " + data.sample_rate + " Hz · زمان پردازش: " +
      ((performance.now() - started) / 1000).toFixed(1) + " ثانیه\n" + (data.run_status || "");
  } catch (error) {
    $("message").textContent = "خطا: " + error.message;
  } finally {
    $("generate").disabled = false;
  }
}

$("generate").addEventListener("click", generate);
updateHealth();
