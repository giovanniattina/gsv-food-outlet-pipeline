# GSV Food Outlet Pipeline

Inference pipeline for classifying Google Street View panoramas as food purchasing
places (healthy vs. unhealthy) in informal settlements, combining a DINOv2-based
visual classifier with OCR-based signage text matching.

This is the Stage 1 (urban-function filter), Stage 2 (food-health classification), and
Stage 3 (OCR late fusion) inference code from:

> Nascimento, G. A., Giannotti, M. A., Fernandes, C. N. "Mapping food outlets with
> deep learning and street view images" (submitted to *Computers, Environment and
> Urban Systems*).

See [`pipeline/README.md`](pipeline/README.md) for usage instructions, input/output
format, and configuration.

## What's here

- `pipeline/run_pipeline_s02_u02.py` — the inference script (Stage 1 urban-function
  filter + Stage 2 food-health classifier + Stage 3 OCR late fusion).
- `pipeline/README.md` — usage, checkpoint format, and configuration reference.
- `docs/` — the portfolio site (GitHub Pages, served from `/docs`): dataset, DINOv2
  embeddings, how the models were chosen, the OCR stage, final results, and how the
  detections differ from RAIS. It shows a sample of 12 Street View crops only.

## What's not here

Model checkpoints (`.pt` files) are not included — they are regenerated from the
project's training notebooks (see `pipeline/README.md`, "Obtaining the Model
Checkpoints"). Raw Google Street View imagery and RAIS administrative records are
not redistributed due to third-party licensing and terms of service; processed
data and label mappings are available from the corresponding author upon
reasonable request.

## Citation

If you use this code, please cite the article above (citation details to be
updated on acceptance/publication).
