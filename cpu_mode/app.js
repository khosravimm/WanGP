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

// Browser-side audio conversion ensures whisper.cpp receives a validated
// mono 16 kHz 16-bit PCM WAV while leaving the server stdlib-only.
let previousTranscriptUrl = null;

async function checkASR() {
  try {
    const response = await fetch("/api/asr-health", {cache:"no-store"});
    const data = await response.json();
    const ready = response.ok && data.status === "ok";
    $("asr-health").textContent = ready ? "ASR آماده · CPU" : "ASR در دسترس نیست";
    $("asr-health").classList.toggle("ready", ready);
    $("transcribe").disabled = !ready;
  } catch (_) {
    $("asr-health").textContent = "ASR در دسترس نیست";
    $("transcribe").disabled = true;
  }
}

function encodeWav16k(audioBuffer) {
  const samples = audioBuffer.getChannelData(0);
  const wav = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(wav);
  const str = (offset, text) => {
    for (let i = 0; i < text.length; ++i) view.setUint8(offset + i, text.charCodeAt(i));
  };
  str(0,"RIFF"); view.setUint32(4,36+samples.length*2,true);
  str(8,"WAVE"); str(12,"fmt ");
  view.setUint32(16,16,true); view.setUint16(20,1,true);
  view.setUint16(22,1,true); view.setUint32(24,16000,true);
  view.setUint32(28,32000,true); view.setUint16(32,2,true);
  view.setUint16(34,16,true); str(36,"data");
  view.setUint32(40,samples.length*2,true);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1,Math.min(1,samples[i]));
    view.setInt16(44+i*2, value < 0 ? value*32768 : value*32767,true);
  }
  return new Blob([wav],{type:"audio/wav"});
}

async function decodeAsWav16k(file) {
  const buffer = await file.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    const decoded = await audioContext.decodeAudioData(buffer);
    if (!decoded.duration || decoded.duration > 180)
      throw new Error("طول فایل صوتی باید حداکثر ۱۸۰ ثانیه باشد.");
    const frames = Math.ceil(decoded.duration * 16000);
    const offline = new OfflineAudioContext(1,frames,16000);
    const source = offline.createBufferSource();
    source.buffer = decoded;
    source.connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    return encodeWav16k(rendered);
  } finally {
    await audioContext.close();
  }
}

async function transcribeAudio() {
  const file = $("asr-file").files[0];
  if (!file || !file.size || file.size > 20*1024*1024) {
    $("asr-message").textContent = "یک فایل صوتی معتبر و حداکثر ۲۰ MiB انتخاب کنید.";
    return;
  }
  $("transcribe").disabled = true;
  $("asr-output").value = "";
  $("asr-message").textContent = "در حال تبدیل صوت به 16 kHz...";
  const started = performance.now();
  try {
    const wav = await decodeAsWav16k(file);
    const form = new FormData();
    form.append("language",$("asr-lang").value);
    form.append("file",wav,"audio.wav");
    $("asr-message").textContent = "در حال رونویسی با Whisper روی CPU...";
    const response = await fetch("/api/transcribe",{
      method:"POST", body:form, cache:"no-store"
    });
    const result = await response.json();
    if(!response.ok) throw new Error(result.error || "خطای رونویسی");
    $("asr-output").value = result.text || "";
    const sec = ((performance.now()-started)/1000).toFixed(1);
    $("asr-message").textContent = "رونویسی انجام شد.";
    $("asr-meta").textContent = "زبان تشخیص‌داده‌شده: "+(result.language_detected||"نامشخص")+
      " · مدت صوت: "+result.duration_seconds+" ثانیه · پردازش: "+sec+" ثانیه"+
      " · تعداد بخش‌ها: "+(result.segments||[]).length;
    if(previousTranscriptUrl) URL.revokeObjectURL(previousTranscriptUrl);
    previousTranscriptUrl=URL.createObjectURL(
      new Blob([result.text||""],{type:"text/plain;charset=utf-8"}));
    $("asr-download").href=previousTranscriptUrl;
    $("asr-download").hidden=false;
  }catch(error) {
    $("asr-message").textContent = "خطا: "+error.message;
  }finally {
    $("transcribe").disabled = false;
  }
}
$("transcribe").addEventListener("click",transcribeAudio);
checkASR();
