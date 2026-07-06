# Detector Fine-Tuning Plan

> Working plan for training a person detector specialised to this deployment's
> cameras. **This is a draft for you to revise** — decision points and tunable
> parameters are called out explicitly. Last updated 2026-07-06.

---

## 1. Objective

Produce a person detector that runs at the current speed on CPU (OpenVINO) but
**counts the people the stock model misses** — specifically the seated / occluded
/ traditional-dress (thobe + ghutra, viewed from behind) occupants in the waiting
areas, and small/distant people on the fisheye ceiling cameras.

**Definition of done (first iteration):** the fine-tuned model, deployed on CPU,
narrows the gap between the validation-image ROI counts and the `/occupancy/spaces`
endpoint, without regressing the entry/exit door or adding false positives on empty
corridors.

---

## 2. Problem & root cause (recap)

The live occupancy undercounts vs. a manual count of the ROIs. Established causes,
in order of impact:

1. **Model capacity / domain gap** — `yolo11n`/`yolo11m` (COCO-trained) do not even
   *propose* a box for a seated person in a thobe seen from behind (verified: nano
   gives no box at conf 0.01; `yolo11m` only ~0.42). This is a training-distribution
   problem, not a threshold problem → **fine-tuning on local data is the real fix.**
2. **Tracking threshold** — `track_thresh 0.5` > detector `conf 0.20` dropped
   0.2–0.5 detections before counting (mitigated: `track_thresh` lowered to 0.20).
3. **Input resolution** — occupancy runs on the 720p sub-stream; the 4K main stream
   recovers detail (separate change, not part of this plan).

This plan addresses **#1**.

---

## 3. Approach: teacher → student distillation

We cannot run a big accurate model in production (too slow on CPU). So we run it
**once, offline, on the GPU** to auto-label the captured frames, then train the
small deployable model to imitate those labels (knowledge distillation via
pseudo-labels).

```
  153k captured frames
        │  curate (GPU): keep diverse people-frames + sample of empties
        ▼
  ~5k curated frames
        │  label (GPU): ensemble + open-vocab teacher → pseudo-labels
        ▼
  ~5k images + YOLO labels
        │  split (stratified by camera) → data.yaml
        ▼
  train/val dataset
        │  fine-tune student (GPU) → export OpenVINO FP16
        ▼
  deployable CPU model  →  swap in via /engine/reset  →  measure  →  iterate
```

---

## 4. Key decisions — **revise these**

| # | Decision | Current choice | Alternatives / notes |
|---|----------|----------------|----------------------|
| D1 | **Human review of labels** | **Teacher-only (no review)** | Hybrid (review val set + hard cases) — the only way to fully fix thobe-from-behind. Full manual revise. |
| D2 | **Teacher (label source)** | **Ensemble: `yolo11x`+TTA, `yolo26x`, YOLO-World**, fused with WBF | Plain `yolo11x`; add GroundingDINO (stronger open-vocab, heavier setup). |
| D3 | **Student (deployed model)** | **`yolo11n` (nano)** — the cheap CPU model that runs across all 13 cameras (the whole point) | `yolo11m`/`yolo26m` (higher recall but too slow on CPU for all cameras — the original problem). |
| D4 | **Classes** | **Single class: person** | Keep multi-class (not needed for occupancy/entry-exit). |
| D5 | **Curated set size** | ≤ 400 people-frames + 80 empties **per camera** (~5k total) | Raise caps for more data; lower for a faster first pass. |
| D6 | **Train imgsz** | **1280** (matches prod inference) | 1536 for more small-person detail (slower train + inference). |
| D7 | **Teacher label imgsz** | **1920** | Higher = better recall on small people, more GPU time. |
| D8 | **WBF keep threshold** | **0.25** fused score | Lower = more recall / more false labels; higher = cleaner but fewer. Critical knob since teacher-only. |
| D9 | **Source stream for training frames** | 720p sub-stream (what's captured) | Main 4K stream (needs the separate main-stream capture change). |

---

## 5. Pipeline (scripts, commands, parameters)

Isolated GPU venv: `PY=.venv-train/Scripts/python`

### Stage 1 — Curate  (`curate_and_label.py`)
Detection-driven selection: run a detector on all frames, keep — per camera — those
with a **new person configuration** (count + quantised box centres), plus an even
sample of empty frames as negatives. (Whole-frame hashing failed: a small person in
a static scene doesn't change the hash, so it discarded the frames we need.)
```
$PY scripts/training/curate_and_label.py --src training_data --out dataset/curated \
    --model yolo11x.pt --imgsz 1280 --batch 8 --stride 2 --people-cap 400 --empty-cap 80
```
Params: `--stride` (process every Nth frame), `--people-cap`, `--empty-cap`,
`--conf` (keep decision), `--batch` (GPU memory).

### Stage 2 — Label with the teacher  (`label_ensemble.py`)
Ensemble + open-vocab, merged with Weighted Box Fusion; overwrites the rough labels.
```
$PY scripts/training/label_ensemble.py --images dataset/curated/images \
    --out dataset/curated/labels --imgsz 1920 --keep 0.25
```
Members: `yolo11x` (+TTA, weight 2.0), `yolo26x` (2.0), `yolov8x-worldv2` (open-vocab
"person", 1.5). Params: `--keep` (D8), `--iou-thr` (fusion IoU), `--conf` (per-model floor).

### Stage 2b — Human review  *(SKIPPED per D1; slot reserved)*
If enabled later: load `dataset/curated/{images,labels}` into Label Studio / CVAT /
Roboflow with boxes pre-loaded; correct the val set + hard cases; export back.

### Stage 3 — Build dataset  (`build_dataset.py`)
Stratified **by camera** (so val isn't a whole unseen camera). Missing labels → empty
(explicit negatives).
```
$PY scripts/training/build_dataset.py --images dataset/curated/images \
    --labels dataset/curated/labels --out dataset/yolo --val-frac 0.2
```

### Stage 4 — Fine-tune + export  (`train.py`)
```
$PY scripts/training/train.py --data dataset/yolo/data.yaml --model yolo11m.pt \
    --imgsz 1280 --epochs 100 --device 0
```
Auto-exports `training_runs/person_ft/weights/best_openvino_model/` (FP16, dynamic).

---

## 6. Environment / hardware

- **GPU:** NVIDIA RTX 5090 Laptop, 24 GB (Blackwell / sm_120).
- **Training env:** `.venv-train` — CUDA torch (`cu128`), `ultralytics` 8.4.89,
  `ensemble-boxes`. **Isolated** from the server's system Python so training never
  disturbs the running capture/inference.
- **Server env:** unchanged (CPU-only torch, OpenVINO inference).

---

## 7. Data specifics

- **Source:** `training_data/cam<id>_<ip>/<date>/*.jpg` — captured by the live
  `dataset_capture` feature. Occupancy (ISAPI) cameras save original JPEG bytes; the
  RTSP door saves decoded frames. ~153k frames and growing (~10.4k/occupancy cam,
  ~28k door). `min_interval_s: 1.0`.
- **Capture stays ON** during and after training → feeds the next iteration.
- **Curation output:** `dataset/curated/{images,labels}` (flat, `cam..__<name>.jpg`).
- All of `dataset/`, `training_runs/`, `.venv-train/` are gitignored.

---

## 8. Training config (hyperparameters — revise in `train.py`)

- Base weights: COCO-pretrained `yolo11m.pt` (transfer learning — keeps general
  person knowledge, specialises to our scenes).
- `epochs 100`, `patience 20` (early stop on val mAP plateau), `batch -1` (auto-fit
  24 GB), `single_cls`, `cos_lr`.
- Augmentation tuned for fixed indoor CCTV: `fliplr 0.5`, **no** `flipud`, no rotation,
  `mosaic 1.0` (closed last 10 epochs), mild HSV.

---

## 9. Deployment

Export → point the server at it, apply without a process restart:
- `.env`: `MODEL_DETECTOR_PATH=training_runs/person_ft/weights/best_openvino_model`
- `config.local.yaml`: `detector.model_path` = same
- `POST /api/v1/engine/reset`

---

## 10. Validation & success metrics

1. **Held-out mAP / recall** on the camera-stratified val split (ultralytics reports it).
2. **Real-world check:** re-run the validation-image ROI count vs `/occupancy/spaces`
   for `b1-waiting-area` — the undercount gap should shrink.
3. **Regression guards:** entry/exit door not over/under-counting; empty corridors
   (CAM-25) not generating false positives.
4. **Latency:** exported model still fits the CPU budget at the deployed imgsz.

---

## 11. Iteration (active learning)

After deploying v1: run it live, collect frames where it **disagrees with the teacher**
or is **low-confidence**, add those to the curated set, retrain. 2–3 loops chase the
remaining blind spots. This is the natural place to insert a human-review round (Stage
2b) once labeling effort is available — that's what closes the hardest thobe cases.

---

## 12. Risks & limitations (honest)

- **Teacher-only ceiling (biggest risk):** no COCO teacher — even this ensemble —
  reliably sees the seated-thobe-from-behind case. v1 will improve medium-hard cases
  a lot but **will not fully fix that hardest case.** Measuring after v1 tells us how
  far the ensemble got; if it's short, Stage 2b (human review) is the fix.
- **Label noise:** teacher false positives (chairs, reflections) become training
  labels with no human to catch them. Controlled by the WBF `--keep` threshold (D8),
  but it's a precision/recall trade with no safety net.
- **Overfitting to 13 fixed views:** the model specialises to these exact cameras;
  new cameras/angles may need their own data. Camera-stratified val partly measures this.
- **Domain drift:** lighting/season/crowd changes over time → periodic re-capture &
  retrain (capture is always-on to support this).

---

## 13. Open questions for you

- **[D1]** Accept teacher-only for v1, or budget a human-review round now?
- **[D3]** Student = `yolo11m` (safe, matches prod) or `yolo26m` (better/cheaper in tests)?
- **[D5/D6]** Happy with ~5k frames @ imgsz 1280, or scale up?
- **[D8]** WBF keep threshold 0.25 — bias toward recall (lower) or precision (higher)?
- Add **GroundingDINO** to the teacher for the unusual poses (heavier setup, best recall)?
- Do we also want the **main-stream 4K** capture change so future training data is higher-res?

---

## 14. Status snapshot (2026-07-06 — will change)

- ✅ GPU env ready (CUDA verified on the 5090); pipeline scripts + this plan written.
- ✅ Capture running & untouched (~153k+ frames, growing into business hours).
- ✅ Ensemble labeler validated (5 boxes on a known daytime frame; correctly 0 on empty night frames).
- ⚠️ **Learning:** capture ran overnight, so the first curation was night-biased and
  the busy-camera 400-caps filled on empty frames. **Fixed:** curation now runs
  **newest-first (`--desc`) with an early break at the cap**, so it keeps recent
  *daytime* people (verified: newest CAM-26 frame @ ~11:20 local has people).
  Filenames are UTC; local = UTC+3.
- ✅ Decision **D1 → train v0 now** on current data + retrain v1 on fuller daytime data.
- 🔄 **Re-curation running** (descending). Next (auto): ensemble labeling → build split
  → fine-tune v0 → export → measure. Capture keeps feeding v1.
