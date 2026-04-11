# Hugging Face assets — alignment with paper Table 1

This note ties **checkpoints** and **datasets** to the paper’s three properties (friction, viscosity, elasticity), **absolute vs relative** tasks, and **train + Test-1/2/3** splits. Paths below are relative to the project root  
`/work/guanqi/proj5/Inferring_dynamic_physics` (i.e. `checkpoints/…`, `dataset/…`).

---

## 1. Dataset layout (24 logical bundles)

Paper convention: **3 properties × 2 task types × 4 splits = 24**.

| Factors | Values |
|--------|--------|
| Properties | friction, viscosity, elasticity |
| Task type | **absolute** (value prediction), **relative** (pairwise comparison) |
| Splits | **train**, **test1** (synthetic, in-distribution), **test2** (synthetic, OOD), **test3** (real / harder) |

So: **4 × 2 × 3 = 24** (same as **3 × 2 × 4**).

On disk, names are **not** one flat folder per bundle; they follow historical experiment IDs. Use this **logical → physical** map when building the Hub dataset (recommended Hub layout: `friction/absolute/{train,test1,test2,test3}/` etc., filled from the paths below).

### 1.1 Friction — absolute

| Split | Primary location under `dataset/` |
|-------|-------------------------------------|
| train | *(often bundled with synthetic pipeline; see friction_relative train or project `data_global_map` for the exact train root used in training)* |
| test1 | `friction_absolute/4.29_friction_synthetic_test1/` |
| test2 | `friction_absolute/4.29_friction_synthetic_test2/` |
| test3 | `friction_absolute/7.18_friction_test3_rgb_seg_videos_152frames_abs_test/` |

Related zips in `dataset/friction_absolute/` and top-level `dataset/*.zip` are backups of the same material.

### 1.2 Friction — relative

| Split | Location |
|-------|----------|
| train | `friction_relative/5.24_friction_synthetic_train/` |
| test1 | `friction_relative/5.24_friction_synthetic_test1/` |
| test2 | `friction_relative/5.24_friction_synthetic_test2/` |
| test3 | `friction_relative/7.18_friction_test3_rgb_seg_videos_152frames_relative_test/` |

### 1.3 Viscosity — absolute

| Split | Location |
|-------|----------|
| train | `viscosity_absolute/5.2_viscosity_synthetic_train/` (+ matching `.zip`) |
| test1 | `viscosity_absolute/5.2_viscosity_synthetic_test1.zip` (unpack for Hub) |
| test2 | `viscosity_absolute/5.2_viscosity_synthetic_test2.zip` |
| test3 | `viscosity_absolute/7.15_new_viscosity_video_clips_16frames_merged.zip` (and related `6.2_viscosity_test3_absolute.zip` at `dataset/` root if needed) |

### 1.4 Viscosity — relative

| Split | Location |
|-------|----------|
| train | `viscosity_relative/5.24_viscosity_synthetic_train/` |
| test1 | `viscosity_relative/5.24_viscosity_synthetic_test1/` |
| test2 | `viscosity_relative/5.24_viscosity_synthetic_test2/` |
| test3 | `viscosity_relative/7.15_new_viscosity_video_clips_152frames_relative_merged_revised2.zip` |

### 1.5 Elasticity — absolute

| Split | Location |
|-------|----------|
| train | `elasticity_absolute/4.11_mpm_bounce_ball_videos_train2_sorted.zip` (unpack) |
| test1 | `elasticity_absolute/4.11_mpm_bounce_ball_videos_test1_sorted.zip` |
| test2 | `elasticity_absolute/4.11_mpm_bounce_ball_videos_test2_sorted.zip` |
| test3 | `elasticity_absolute/7.15_new_elasticity_video_clips_16frames.zip` |

### 1.6 Elasticity — relative

| Split | Location |
|-------|----------|
| train | `elasticity_relative/5.24_elasticity_synthetic_train/` (+ `.zip`) |
| test1 | `elasticity_relative/5.24_elasticity_synthetic_test1.zip` |
| test2 | `elasticity_relative/5.24_elasticity_synthetic_test2.zip` |
| test3 | `elasticity_relative/7.14_elasticity_test3.zip` |

**Note:** List JSON files used by `WebVid` / configs (e.g. under `SVD_ROOT/data/lists/`) should be uploaded next to videos or documented in the dataset card; the open-source gate uses at least `elasticity_absolute_test3` + `elasticity_relative_test3` + list JSONs (see `configs/paths.example.yaml`).

---

## 2. Checkpoints — DynamiCrafter (**E**, paper “Video Generative Model”)

Row identity follows `SVD/paper_scores_E_7.29.md`: **v120–v125** cover **three properties × abs/rel** (six version IDs).

### 2.1 Six main **7.29** runs (Table 1 core, without `test3*`)

Upload the whole run folder or pin one `.ckpt` per row per your `model_global_map.md` / eval yaml.

| Version | Property | Task | Folder under `checkpoints/` |
|--------|----------|------|-----------------------------|
| v120 | friction | absolute | `training_1024_v120_7.29/checkpoints/` |
| v121 | friction | relative | `training_1024_v121_7.29/checkpoints/` |
| v122 | viscosity | absolute | `training_1024_v122_7.29/checkpoints/` |
| v123 | viscosity | relative | `training_1024_v123_7.29/checkpoints/` |
| v124 | elasticity | absolute | `training_1024_v124_7.29/checkpoints/` |
| v125 | elasticity | relative | `training_1024_v125_7.29/checkpoints/` |

That is **6** checkpoint **directories** (paper-aligned “main” E set).

### 2.2 Friction **test3\*** / finetune (Table 1 rows with domain adaptation / finetuned test3)

| Role | Folder under `checkpoints/` |
|------|-----------------------------|
| v120 friction abs test3\* | `training_1024_v120_ft_7.29/checkpoints/` (e.g. `epoch=3299-step=3300.ckpt` per your sweep) |
| v121 friction rel test3\* | `training_1024_v121_ft_7.29/checkpoints/` |

That adds **2** more directories → **8** E-side bundles if you ship **both** main 6 and **both** friction finetunes.

**Why “6+1=7” might appear:** some spreadsheets count **6** property-task heads plus **one** shared base encoder; on disk you still have **separate** training runs. Another interpretation is **6** main + **1** finetune only → **7**; the paper table, however, lists **two** friction `test3*` rows (abs and rel), which matches **+2** finetune dirs above.

**Exact filename per row:** always take the epoch/step from the same yaml / `model_global_map.md` used when the paper numbers were frozen.

---

## 3. Checkpoints — V-JEPA-2 mirror (**B**)

From `SVD/paper_scores_B_sources_and_checkpoints.md`.

### 3.1 Six **norm** runs (one ckpt path covers test1–3 in eval configs)

| Version | Property | Task | Folder under `checkpoints/` | Example file (from index) |
|--------|----------|------|-----------------------------|---------------------------|
| v110 | viscosity | relative | `training_1024_v110_vjepa_norm/checkpoints/` | `epoch=120-step=15125.ckpt` |
| v119 | viscosity | absolute | `training_1024_v119_vjepa_norm/checkpoints/` | `epoch=21-step=13000.ckpt` |
| v120 | friction | absolute | `training_1024_v120_vjepa_norm/checkpoints/` | `epoch=116-step=14000.ckpt` |
| v121 | friction | relative | `training_1024_v121_vjepa_norm/checkpoints/` | `epoch=120-step=15125.ckpt` |
| v124 | elasticity | absolute | `training_1024_v124_vjepa_norm/checkpoints/` | `epoch=21-step=8375.ckpt` |
| v125 | elasticity | relative | `training_1024_v125_vjepa_norm/checkpoints/` | `epoch=126-step=15875.ckpt` |

→ **6** directories for the main B line.

### 3.2 Two **finetune** runs (friction **test3\*** rows in the B table)

| Role | Folder under `checkpoints/` | Notes |
|------|-----------------------------|--------|
| v120 friction abs test3\* | `training_1024_v120_vjepa_ft/checkpoints/` | e.g. `epoch=5499-step=5500.ckpt` |
| v121 friction rel test3\* | **Primary ckpt in index may live outside this backup** (`paper_scores_B_sources_and_checkpoints.md` §1.1, scratch path). Locally you may ship `training_1024_v121_vjepa_ft/checkpoints/` as a substitute until the exact `norm15125_ttest` file is copied into `checkpoints/`. |

→ **+2** finetune dirs → **8** B-side bundles for full parity with the B table (including `test3*`).

**“6 + 2 = 8”** matches: **6** `_vjepa_norm` + **2** friction finetunes.

---

## 4. Summary counts (what to upload for full paper parity)

| Asset class | Count | What it is |
|-------------|------:|------------|
| **E (DynamiCrafter)** | **6** | `v120`–`v125` `_7.29` runs |
| **E finetune (optional but for `test3\*`)** | **+2** | `v120_ft_7.29`, `v121_ft_7.29` |
| **B (V-JEPA norm)** | **6** | `v110`, `v119`, `v120`, `v121`, `v124`, `v125` `_vjepa_norm` |
| **B finetune (test3\*)** | **+2** | `v120_vjepa_ft`, plus v121 finetune (verify path vs §1.1) |
| **Dataset logical cells** | **24** | 3 properties × 2 tasks × 4 splits (see §1) |

The public repo’s **minimal** self-check still only **requires** E v124/v125 and B v124/v125 for **elasticity Test-3** (`scripts/verify_paper_elasticity_test3.py`); full Hub upload should follow this document for **whole Table 1**.

---

## 5. Changelog

- **2026-04-11**: Initial manifest from `paper_scores_E_7.29.md`, `paper_scores_B_sources_and_checkpoints.md`, and local `checkpoints/` + `dataset/` layout.
