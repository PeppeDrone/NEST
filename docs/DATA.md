# Data

The neonatal recordings, the poses extracted from them and every clip-level artifact derived from them
are **not distributed**: they are clinical data of identifiable infants. This document describes the
layout the code expects, so that the pipeline can be run on equivalent data (or on the original data by
authorised researchers; see the Data availability statement of the paper).

## Expected layout (all paths git-ignored)

```
data/dwpose_woface_62_v1/
  teacher_training_manifest.jsonl      one JSON object per clip (fields below)
  artifacts/<clip_id>.npz + .json      checksummed teacher pose pair (format below)
Data/processed/samples/<clip_id>/ref/ref.png   first frame of the clip (Stage-2 reference image; image size)
Data/Clips/<patient_id>/<clip_id>.mp4          source clip (native frame timing)
```

Paths inside the manifest are relative to the manifest file (e.g. `../../Data/processed/...`).

## Manifest fields

| field | meaning |
|---|---|
| `clip_id` | `<patient_id>_clip<NNNN>` |
| `patient_id` | pseudonymous infant identifier; must match `data_refs/patient_split_v1.json` |
| `split` | `train`, `val` or `test` (verified against the frozen split at load time) |
| `manual_prompt` | sentence compiled from the structured clinical form |
| `automatic_prompt` | caption produced by a generic video captioner |
| `pose_npz`, `pose_sha256`, `pose_json_sha256` | teacher pose pair and checksums |
| `reference_image`, `source_video`, `target_frames_dir` | paths to the reference still, clip and frames |
| `status` | `ok` for accepted teachers |

## Teacher pose pair (`dmtbaby.pose` format, schema `dwpose_woface_62_v1`)

An NPZ with `raw_xy_norm`, `aligned_xy_norm` (20 frames × 62 joints × 2, normalised image coordinates,
NaN where unavailable), `raw_score`, `detector_valid`, `render_visible`, `ref_xy_norm`, `ref_score`,
`ref_visible` (reference frame) and `source_frame` (frame indices in the source clip), plus a JSON
sidecar holding the NPZ checksum and metadata. Writers and validators: `src/dmtbaby/pose/io.py` and
`src/neomotion/common/pose_io.py`. The body-20 subset (nose, neck, shoulders, elbows, wrists, hips, knees,
ankles, eyes, ears, foot centres) is used by Stage 1.

Teachers were extracted with YOLOX-L + DWPose (`dw-ll_ucoco_384`), checksums in
`stage2/third_party/model_provenance.json`, and quality-filtered (complete torso anchors, valid
confidences and coordinates): 1,203 accepted clips.

## Patient split

`data_refs/patient_split_v1.json` lists the pseudonymous infants and clips of each split
(train 33 infants / 875 clips, validation 4 / 115, test 5 / 213). It is frozen; every training run and
evaluation verifies the manifest against it and records its SHA-256.
