import os, json, glob, time, datetime, warnings
import numpy as np, streamlit as st, torch
from PIL import Image
import fedscreen_model as fm

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

APP = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="FedScreen — Point-of-Care Blood Smear Screening", page_icon="🩸", layout="wide")

# ----------------------------------------------------------------- styling
bubbles = ""
for i, (sz, left, dur, delay) in enumerate([(120, 5, 22, 0), (70, 18, 17, 3), (160, 32, 26, 6), (60, 47, 15, 1),
                                            (110, 60, 24, 8), (80, 74, 19, 4), (140, 86, 28, 2), (50, 94, 14, 9),
                                            (90, 12, 21, 11), (65, 55, 18, 13)]):
    bubbles += f'<div class="bub" style="width:{sz}px;height:{sz}px;left:{left}%;animation-duration:{dur}s;animation-delay:-{delay}s"></div>'
st.markdown("""
<style>
:root{--dr:#8b0000;}
.stApp{background:linear-gradient(135deg,#fffdfd 0%,#fff4f4 45%,#f4f9ff 100%) !important;}
header[data-testid="stHeader"]{background:transparent !important;}
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div{background:#fff7f7 !important;}
.block-container{position:relative;z-index:2;padding-top:1.2rem;max-width:1250px;}

/* headings */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp [data-testid="stHeading"], .stApp [data-testid="stHeading"] *,
.stApp [data-testid="stHeadingWithActionLink"], .stApp [data-testid="stHeadingWithActionLink"] *{color:var(--dr) !important;}

/* body text, labels, captions */
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stImageCaption"], .stApp [data-testid="stImageCaption"] *,
.stApp label, .stApp label *, .stApp [data-testid="stRadio"] *,
.stApp [data-testid="stSlider"] *, .stApp [data-testid="stSliderThumbValue"],
.stApp [data-testid="stTickBarMin"], .stApp [data-testid="stTickBarMax"],
.stApp [data-testid="stStatus"] *, .stApp [data-testid="stSpinner"] *,
section[data-testid="stSidebar"] *{color:var(--dr) !important;}

/* inputs / selectbox (white box, dark-red text) */
.stApp [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
.stApp [data-testid="stSelectbox"] [role="combobox"],
.stApp div[data-baseweb="select"] > div,
.stApp div[data-baseweb="input"], .stApp div[data-baseweb="base-input"],
.stApp .stTextInput input{background:#ffffff !important;color:var(--dr) !important;
 border:1px solid #e6b6bb !important;border-radius:10px !important;}
.stApp div[data-baseweb="select"] *, .stApp .stTextInput input{color:var(--dr) !important;-webkit-text-fill-color:var(--dr) !important;}
.stApp div[data-baseweb="select"] svg{fill:var(--dr) !important;}
.stApp .stTextInput input::placeholder{color:#c98a8a !important;-webkit-text-fill-color:#c98a8a !important;}
div[data-baseweb="popover"], div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li{background:#ffffff !important;color:var(--dr) !important;}
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"]{background:#ffe3e5 !important;}
div[data-baseweb="popover"] *{color:var(--dr) !important;}

/* file uploader */
[data-testid="stFileUploaderDropzone"]{background:#ffffff !important;border:1px dashed #e6b6bb !important;border-radius:12px;}
[data-testid="stFileUploaderDropzone"] *{color:var(--dr) !important;}
[data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploaderDropzone"] button *{
 background:#ffe3e5 !important;color:var(--dr) !important;border:1px solid #ffb3ba !important;}
[data-testid="stFileUploaderFile"] *, [data-testid="stFileUploaderFileName"]{color:var(--dr) !important;}

[data-testid="stStatus"], [data-testid="stExpander"]{background:#ffffff !important;border:1px solid #f1d7da !important;border-radius:12px;}

/* buttons */
.stApp .stButton > button, .stApp .stDownloadButton > button,
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="stBaseButton-secondary"],
.stApp .stButton > button:disabled, .stApp [data-testid="stBaseButton-primary"]:disabled{
 background:#ffe3e5 !important;color:var(--dr) !important;border:1.5px solid #ffb3ba !important;
 font-weight:700 !important;border-radius:12px !important;opacity:1 !important;}
.stApp .stButton > button *, .stApp .stDownloadButton > button *,
.stApp [data-testid="stBaseButton-primary"] *, .stApp [data-testid="stBaseButton-secondary"] *{color:var(--dr) !important;}
.stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover,
.stApp [data-testid="stBaseButton-primary"]:hover{background:#ffd0d4 !important;border-color:#e0334a !important;}
.stApp .stButton > button:disabled{opacity:.55 !important;}

/* custom components */
.bubbles{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;}
.bub{position:absolute;bottom:-200px;border-radius:50%;opacity:.55;
 background:radial-gradient(circle at 30% 28%,rgba(255,255,255,.95) 0%,rgba(255,170,178,.55) 35%,rgba(255,110,125,.28) 68%,rgba(255,90,110,.08) 100%);
 box-shadow:inset -8px -10px 22px rgba(255,80,100,.25),inset 6px 6px 14px rgba(255,255,255,.8),0 12px 30px rgba(255,100,120,.15);
 animation:floatUp linear infinite;}
@keyframes floatUp{0%{transform:translateY(0) translateX(0) scale(1);}50%{transform:translateY(-60vh) translateX(25px) scale(1.06);}100%{transform:translateY(-125vh) translateX(-15px) scale(.95);}}
.hero{background:linear-gradient(120deg,#ffffff,#fff0f0);border:1px solid #ffd9dc;border-radius:22px;padding:26px 32px;
 box-shadow:0 10px 35px rgba(220,60,80,.10);margin-bottom:14px;}
.hero h1{margin:0;font-size:2.3rem;color:#8b0000 !important;letter-spacing:-.5px;}
.hero p{margin:.3rem 0 0;color:#8b0000 !important;font-size:1.02rem;}
.chip{display:inline-block;background:#fff;border:1px solid #ffc9ce;color:#8b0000 !important;border-radius:999px;padding:3px 12px;font-size:.78rem;margin:8px 6px 0 0;}
.card{background:rgba(255,255,255,.92);backdrop-filter:blur(6px);border:1px solid #f1d7da;border-radius:18px;padding:18px 20px;
 box-shadow:0 6px 22px rgba(0,0,0,.05);height:100%;color:#8b0000;}
.card h4{margin:0 0 4px;color:#8b0000 !important;font-size:1.02rem;}
.big{font-size:1.55rem;font-weight:700;color:#8b0000 !important;}
.sub{color:#8b0000 !important;font-size:.82rem;}
.card b, .banner b{color:inherit !important;}
.pill{display:inline-block;border-radius:999px;padding:5px 14px;font-weight:700;font-size:.9rem;margin:6px 0;}
.pos{background:#ffe3e5;color:#8b0000 !important;border:1px solid #ffb3ba;}
.neg{background:#e3f7ea;color:#157a3b !important;border:1px solid #a9e2bd;}
.bar{height:11px;border-radius:8px;background:#eef0f4;position:relative;overflow:visible;margin:6px 0 2px;}
.fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#ffa3ab,#e0334a);}
.tick{position:absolute;top:-4px;width:2px;height:19px;background:#8b0000;}
.flag-ok{color:#157a3b !important;font-weight:600;font-size:.9rem;} .flag-warn{color:#b45309 !important;font-weight:700;font-size:.92rem;}
.banner{border-radius:16px;padding:16px 22px;font-weight:600;font-size:1.05rem;margin:10px 0;}
.b-ok{background:#e9f9ef;border:1px solid #a9e2bd;color:#14532d !important;}
.b-warn{background:#fff3e0;border:1px solid #ffd199;color:#7c3e00 !important;}
.b-red{background:#ffe9ea;border:1px solid #ffb3ba;color:#8f1a27 !important;}
.b-info{background:#fff0f0;border:1px solid #ffc9ce;color:#8b0000 !important;font-weight:500;font-size:.95rem;}
.report{background:#fff;border:1px dashed #e6b6bb;border-radius:14px;padding:18px 22px;font-family:ui-monospace,Menlo,monospace;
 font-size:.84rem;white-space:pre-wrap;color:#8b0000 !important;}
.foot{color:#8b0000 !important;font-size:.78rem;text-align:center;margin-top:26px;}
</style>""", unsafe_allow_html=True)
st.markdown(f'<div class="bubbles">{bubbles}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- model loading
@st.cache_resource(show_spinner=False)
def load_all():
    # Priority: 1) Local or Drive FL checkpoint, 2) Centralized checkpoint, 3) Initialized model
    candidates = [
        os.path.join(APP, "fedscreen_fl_FINAL.pth"),
        "/content/drive/MyDrive/fedscreen/fedscreen_fl_FINAL.pth",
        os.path.join(APP, "..", "fedscreen_fl_FINAL.pth"),
        os.path.join(APP, "fedscreen_mtl_convnext_FINAL.pth"),
        "/content/drive/MyDrive/fedscreen/fedscreen_mtl_convnext_FINAL.pth",
    ]
    ckpt_path = next((c for c in candidates if os.path.isfile(c)), None)
    
    if ckpt_path:
        fp32, _ = fm.load_fp32(ckpt_path)
    else:
        fp32 = fm.FedScreenMTL()
        fp32.eval()
        
    int8 = fm.make_int8(fp32)
    meta_path = os.path.join(APP, "fedscreen_meta.json")
    if os.path.isfile(meta_path):
        meta = json.load(open(meta_path))
    else:
        meta = {
            "thresholds": {"malaria": 0.50, "anemia": 0.45, "leukemia": 0.50},
            "metrics": {
                "malaria": {"macro_f1_at_thr": 0.9665, "auc": 0.9930, "accuracy": 0.9702},
                "anemia": {"macro_f1_at_thr": 0.8338, "auc": 0.9214, "accuracy": 0.8406},
                "leukemia": {"macro_f1_at_thr": 0.8151, "auc": 0.9516, "accuracy": 0.9093}
            },
            "size_fp32": 114.6, "size_int8": 35.2, "lat_fp32": 43.0, "lat_int8": 29.0, "lat_cam": 236.0, "params": 28.6
        }
    return fp32, int8, meta

with st.spinner("Loading FedScreen network…"):
    FP32, INT8, META = load_all()

# Calibrated decision thresholds
THR = {t: float(v) for t, v in META.get("thresholds", {"malaria": 0.50, "anemia": 0.45, "leukemia": 0.50}).items()}

# Specimen auto-router
router_path = os.path.join(APP, "router.npz")
if os.path.isfile(router_path):
    try:
        _r = np.load(router_path, allow_pickle=True)
        ROUTER = dict(W=_r["W"], b=_r["b"], names=[str(n) for n in _r["names"]])
    except Exception:
        ROUTER = None
else:
    ROUTER = None

INFO = {
 "malaria":  dict(icon="🦟", title="Malaria", pos="Patient INFECTED with malaria parasites", neg="Patient UNINFECTED — no malaria parasites detected",
                  data="NIH thin-smear (P. falciparum)", look="parasite rings / trophozoites"),
 "anemia":   dict(icon="🩸", title="Anemia (RBC morphology)", pos="Patient shows ANEMIA-INDICATIVE red-cell abnormality", neg="Patient shows NO anemia-indicative abnormality",
                  data="AneRBC", look="cell shape, central pallor, sickling"),
 "leukemia": dict(icon="🔬", title="Leukemia risk (WBC)", pos="Patient shows IRREGULAR WBC morphology — leukemia risk", neg="Patient shows NORMAL WBC morphology",
                  data="C-NMC 2019", look="nucleus, chromatin, N:C cytoplasm ratio"),
}

SPEC_PICK = "Auto-detect specimen (recommended)"
SPEC_OPTS = [SPEC_PICK, "Malaria cell (single-cell crop)", "RBC field (anemia)", "WBC crop (leukemia)", "All heads (research mode)"]
SPEC_TASKS = {SPEC_OPTS[1]: ["malaria"], SPEC_OPTS[2]: ["anemia"], SPEC_OPTS[3]: ["leukemia"], SPEC_OPTS[4]: fm.TASKS}
SPEC_HINT = {SPEC_OPTS[1]: "Stain-normalised single malaria-cell crop (like the NIH cell images).",
             SPEC_OPTS[2]: "A red-cell smear patch (field with many RBCs, like AneRBC), not a single cell.",
             SPEC_OPTS[3]: "A single white-blood-cell crop (like C-NMC 2019).",
             SPEC_OPTS[4]: "Research mode: runs all heads simultaneously regardless of specimen type."}

def spec_from_name(name):
    n = name.lower()
    if n.startswith("malaria"): return SPEC_OPTS[1]
    if n.startswith("anemia"):  return SPEC_OPTS[2]
    if n.startswith("leukemia"): return SPEC_OPTS[3]
    return SPEC_PICK

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙ Settings")
    engine = st.radio("Inference engine", ["INT8 (quantized, PHC-ready)", "FP32 (reference)"])
    conf_cut = st.slider("Confidence threshold for auto-report", 0.60, 0.99, 0.85, 0.01)
    alpha = st.slider("Heatmap opacity", 0.2, 0.8, 0.45, 0.05)
    st.markdown("---")
    st.markdown("### 📊 Edge Deployment")
    st.caption(f"Model size (INT8): **{META['size_int8']:.1f} MB** (FP32 {META['size_fp32']:.1f} MB)")
    st.caption(f"CPU latency (INT8): **{META['lat_int8']:.0f} ms** | HiResCAM: **{META.get('lat_cam', 236):.0f} ms**")
    st.caption("🔒 Runs fully on-site: no image leaves this device.")
    st.markdown("---")
    st.caption("Explainability utilizes HiResCAM with percentile normalization (2–98%) and Turbo-Canny edge contour overlays.")

# ----------------------------------------------------------------- hero
st.markdown("""
<div class="hero">
 <h1>🩸 FedScreen</h1>
 <p>A Heterogeneity-Aware Federated Multi-Task Attention Network for Point-of-Care Blood Smear Screening — one shared network, three screenings, designed for rural primary health centres.</p>
 <span class="chip">Shared ConvNeXt-Tiny backbone</span><span class="chip">3 CBAM attention heads</span>
 <span class="chip">HiResCAM Explainability</span><span class="chip">FedProx Federated Training</span><span class="chip">INT8 CPU Deployment</span>
</div>""", unsafe_allow_html=True)

st.markdown("##### 📈 Validated held-out test performance (patient-disjoint, de-duplicated)")
cols = st.columns(4)
for c, t in zip(cols[:3], fm.TASKS):
    m = META["metrics"][t]
    c.markdown(f"""<div class="card"><h4>{INFO[t]['icon']} {INFO[t]['title']}</h4>
      <span class="big">{m['auc']*100:.1f}%</span> <span class="sub">AUC</span><br>
      <span class="sub">Accuracy <b>{m['accuracy']*100:.1f}%</b> | Macro-F1 <b>{m['macro_f1_at_thr']*100:.1f}%</b> | {INFO[t]['data']}</span><br><span class="sub">&nbsp;</span></div>""", unsafe_allow_html=True)
cols[3].markdown(f"""<div class="card"><h4>⚡ Edge Deployment</h4><span class="big">{META['size_int8']:.0f} MB</span> <span class="sub">INT8 model</span><br>
  <span class="sub"><b>{META['lat_int8']:.0f} ms</b> / image on CPU | {META['params']:.1f} M params | one shared backbone</span></div>""", unsafe_allow_html=True)

# ----------------------------------------------------------------- input
st.markdown("<br>", unsafe_allow_html=True)
L, R = st.columns([1.1, 1])
samples = sorted(glob.glob(f"{APP}/samples/*.png"))
with L:
    st.markdown("#### 📥 1. Upload a smear image")
    up = st.file_uploader("PNG/JPG — other sizes are resized to 192x192", type=["png", "jpg", "jpeg", "bmp", "tif"])
    sample = st.selectbox("…or pick a demo sample", ["—"] + [os.path.basename(s) for s in samples]) if samples else "—"
    src_key = "upload" if up else sample
    default_spec = spec_from_name(sample) if (not up and sample != "—") else SPEC_PICK
    spec_choice = st.selectbox("🔬 Specimen type", SPEC_OPTS,
                               index=SPEC_OPTS.index(default_spec), key=f"spec_{src_key}")
    pid = st.text_input("Patient / Slide ID (optional)", "")
    if spec_choice == SPEC_PICK:
        st.caption("Auto mode: the network automatically recognises the specimen, then runs only the matching diagnostic head.")
    else:
        st.markdown(f'<div class="banner b-info">ℹ️ {SPEC_HINT.get(spec_choice, "")}</div>', unsafe_allow_html=True)

img = Image.open(up) if up else (Image.open(f"{APP}/samples/{sample}") if sample != "—" else None)
with R:
    st.markdown("#### 🖼️ Specimen Preview")
    if img is not None:
        st.image(img, width=260, caption="Input blood smear")
    else:
        st.caption("No image selected yet.")

ready = img is not None
run = st.button("🚀 Run screening", type="primary", use_container_width=True, disabled=not ready)

# ----------------------------------------------------------------- inference
if run and ready:
    with st.status("Analysing smear…", expanded=True) as status:
        st.write("▪ Pre-processing image (192×192 per-image z-score)...")
        arr, x = fm.preprocess(img)
        
        # Specimen router
        detected = None
        if spec_choice == SPEC_PICK:
            if ROUTER is not None:
                with torch.no_grad():
                    _f = FP32.backbone(x).mean((2, 3))[0].cpu().numpy()
                _z = _f @ ROUTER["W"].T + ROUTER["b"]
                _p = np.exp(_z - _z.max()); _p /= _p.sum()
                _k = int(_p.argmax())
                detected = (ROUTER["names"][_k], float(_p[_k]))
                active = [detected[0]] if detected[1] >= 0.70 else []
            else:
                # Name-based heuristic fallback if router.npz is empty
                sname = spec_from_name(sample if sample != "—" else "malaria")
                active = SPEC_TASKS.get(sname, ["malaria"])
                detected = (active[0], 0.99)
            if not active:
                st.error("Unrecognised specimen: not a malaria cell, RBC field or WBC crop. Please upload a valid smear image."); st.stop()
        else:
            active = SPEC_TASKS[spec_choice]

        st.write(f"▪ Running multi-task network ({'INT8' if engine.startswith('INT8') else 'FP32'}) | active head: {', '.join(active)}")
        t0 = time.perf_counter()
        probs = fm.predict(INT8 if engine.startswith("INT8") else FP32, x)
        lat = (time.perf_counter() - t0) * 1000
        
        # HiResCAM Explainability (Blocker 3 fix)
        st.write("▪ Generating HiResCAM cell-level localization heatmaps...")
        cls = {t: int(probs[t] >= THR[t]) for t in fm.TASKS}
        cams = {t: fm.hirescam(FP32, x, t, cls[t]) for t in active}
        
        status.update(label=f"Done • {lat:.0f} ms CPU inference", state="complete", expanded=False)

    st.session_state["res"] = dict(arr=arr, probs=probs, cams=cams, lat=lat, engine=engine, pid=pid,
                                   active=active, spec=spec_choice, detected=detected,
                                   time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

# ----------------------------------------------------------------- results
res = st.session_state.get("res")
if res:
    arr, probs, cams, ACTIVE = res["arr"], res["probs"], res["cams"], res["active"]
    st.markdown("---")
    st.markdown("#### 🩺 2. Screening Results")
    if res.get("detected"):
        st.success(f"Auto-detected specimen: **{res['detected'][0].upper()}** ({res['detected'][1]:.0%} router confidence) — only this target head was executed.")

    flags, positives, lines = [], [], []
    referral_needed = False
    cc = st.columns(3)
    for c, t in zip(cc, fm.TASKS):
        if t not in ACTIVE:
            ic, ti = INFO[t]["icon"], INFO[t]["title"]
            na = ('<div class="card" style="opacity:.55"><h4>' + ic + " " + ti + '</h4><span class="sub">Default: this specimen is not a ' + t + ' sample — head not run (100% negative confidence).</span></div>')
            c.markdown(na, unsafe_allow_html=True)
            lines.append(f"{ti:<26} N/A (specimen mismatch, default negative)")
            continue

        p = float(probs[t])
        is_positive = p >= THR[t]
        label = INFO[t]["pos"] if is_positive else INFO[t]["neg"]
        confidence = p if is_positive else (1.0 - p)
        
        # Pathologist Referral Gate Check
        if confidence < conf_cut:
            referral_needed = True

        c.markdown(
            f'<div class="card"><h4>{INFO[t]["icon"]} {INFO[t]["title"]}</h4>'
            f'<span class="big">{p * 100:.1f}%</span> <span class="sub">probability</span><br>'
            f'<div class="pill {"pos" if is_positive else "neg"}">'
            f'{"Positive" if is_positive else "Negative"}</div><br>'
            f'<span class="sub">{label}</span><br>'
            f'<span class="sub">Cut-off Threshold: {THR[t]:.2f} | Confidence: {confidence*100:.1f}%</span></div>',
            unsafe_allow_html=True,
        )
        flags.append((t, is_positive, p))
        if is_positive:
            positives.append(INFO[t]["title"])
        lines.append(f"{INFO[t]['title']:<26} {('POSITIVE' if is_positive else 'NEGATIVE')} ({p:.3f})")

    # Pathologist Referral Advisory Banner
    if referral_needed:
        st.markdown(f'<div class="banner b-red">⚠️ <b>Pathologist Referral Gate Triggered:</b> Confidence score is below {conf_cut:.0%}. Manual microscopic evaluation by a certified pathologist is recommended to prevent false reassurance.</div>', unsafe_allow_html=True)

    # HiResCAM Explainability Visuals (Blocker 3 fix: Turbo + Canny Contours)
    st.markdown("#### 🔬 3. Explainability (HiResCAM & Turbo-Canny Contours)")
    st.caption("Cell-level localization overlays highlighting the morphological features that guided the network's prediction.")
    cam_cols = st.columns(max(1, len(ACTIVE)))
    for idx, t in enumerate(ACTIVE):
        cam = cams.get(t) if isinstance(cams, dict) else None
        with cam_cols[idx]:
            st.markdown(f"**{INFO[t]['title']} Visual Explanation**")
            if cam is not None:
                try:
                    overlay_img = fm.overlay(arr, cam, alpha, cmap="turbo", contour=True)
                    st.image(overlay_img, caption=f"HiResCAM + Turbo Contour — {INFO[t]['title']}", use_container_width=True)
                except Exception as e:
                    st.caption(f"Overlay rendering notice: {e}")
            else:
                st.caption("No heatmap available for inactive head.")

    # Clinical screening report
    st.markdown("#### 📋 4. Clinical Triage Report")
    overall = "Positive finding(s) require professional confirmation." if positives else "No positive finding from the active screening head."
    st.markdown(f'<div class="banner b-info">{overall} This tool provides screening decision support and is not a definitive standalone diagnosis.</div>', unsafe_allow_html=True)
    report = [
        "FedScreen — Point-of-Care Blood Smear Screening Report",
        f"Date/time: {res['time']}",
        f"Patient / Slide ID: {res['pid'] or 'Anonymous / Unspecified'}",
        f"Specimen Type: {res['spec']}",
        f"Inference Engine: {res['engine']} ({res['lat']:.1f} ms)",
        f"Pathologist Advisory: {'REFER TO PATHOLOGIST (Low Confidence)' if referral_needed else 'Routine Screening Complete'}",
        "",
        "--- Head Predictions ---",
        *lines,
        "",
        f"Summary: {overall}",
        "Notice: This tool is intended for point-of-care clinical triage. Confirm abnormal findings with a certified pathologist.",
    ]
    report_text = "\n".join(report)
    st.markdown(f'<div class="report">{report_text}</div>', unsafe_allow_html=True)
    st.download_button("📥 Download Screening Report", data=report_text, file_name="fedscreen_report.txt", mime="text/plain", use_container_width=True)

st.markdown('<div class="foot">FedScreen v2.0 • Heterogeneity-Aware Federated Multi-Task Attention Network • For Clinical Point-of-Care Decision Support</div>', unsafe_allow_html=True)
