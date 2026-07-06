# Detector fine-tuning pipeline

Fine-tune the person detector on frames captured from *our* cameras so it learns
this deployment's poses (seated, thobe-from-behind, fisheye ceiling mounts) that
the stock COCO model misses. Runs on the GPU (RTX 5090) in an isolated venv
(`.venv-train`) so it never touches the running server's environment.

## Strategy (chosen)

**Teacher → student distillation, teacher-only (no human review).** A big, slow,
accurate *teacher* auto-labels the data; the small, fast *student* (the model we
actually deploy on CPU) trains to imitate those labels.

- **Teacher = ensemble + open-vocabulary**, run once offline on the GPU:
  `yolo11x` (+TTA) + `yolo26x` + **YOLO-World** (open-vocab, prompt "person"),
  merged with Weighted Box Fusion. Since there is no human review, this ensemble's
  recall is the ceiling on the student.
- **Student = `yolo11n` (nano)** — the cheap CPU model that runs across all 13
  cameras. It's what the distillation is *for*: teach nano the harder people it
  currently misses. Fine-tuned then exported to **OpenVINO FP16** for the CPU runtime.

> ⚠️ Honest limitation: no COCO-trained teacher — even an ensemble — reliably sees
> the seated-thobe-from-behind case (it's out of distribution). Teacher-only will
> improve the medium-hard cases a lot but won't fully close that hardest gap; a
> human-review round (Stage 2b) is what would. Revisit after the first iteration.

## Pipeline

```bash
PY=.venv-train/Scripts/python

# 1. Curate: pick diverse people-frames (+ some empties) from the ~150k captures
$PY scripts/training/curate_and_label.py --src training_data --out dataset/curated \
    --model yolo11x.pt --imgsz 1280 --batch 8 --stride 2 --people-cap 400 --empty-cap 80

# 2. Label with the ensemble/open-vocab teacher (overwrites curation's rough labels)
$PY scripts/training/label_ensemble.py --images dataset/curated/images \
    --out dataset/curated/labels --imgsz 1920 --keep 0.25

# 3. Build the YOLO train/val split (stratified by camera) + data.yaml
$PY scripts/training/build_dataset.py --images dataset/curated/images \
    --labels dataset/curated/labels --out dataset/yolo --val-frac 0.2

# 4. Fine-tune on GPU, then auto-export OpenVINO FP16 for the CPU server
#    (workers=0 avoids a Windows page-file/CUDA-DLL crash; batch 8 if OOM)
$PY scripts/training/train.py --data dataset/yolo/data.yaml --model yolo11n.pt \
    --imgsz 1280 --epochs 100 --batch 16 --workers 0 --device 0
```

## Deploy the result

`train.py` exports `training_runs/person_ft/weights/best_openvino_model/`. Point the
server at it (same swap as any model):

- `.env`: `MODEL_DETECTOR_PATH=training_runs/person_ft/weights/best_openvino_model`
- `config.local.yaml`: `detector.model_path` to the same path
- apply with `POST /api/v1/engine/reset`

## Validate

Compare the fine-tuned model's recall to the current one on a held-out set, and
re-run the validation-image vs `/occupancy/spaces` comparison — the undercount gap
should shrink.

## Next iteration (active learning)

Run the fine-tuned model live, collect frames where it disagrees with the teacher
or is low-confidence, add them to the set, retrain. 2–3 loops chases the remaining
blind spots. A human-review round on the hard cases can be inserted here when
labeling effort is available.

All outputs (`dataset/`, `training_runs/`, `.venv-train/`) are gitignored.
