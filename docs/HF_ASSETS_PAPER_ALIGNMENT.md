# Hugging Face assets — alignment with paper Table 1

This note ties **checkpoints**, **datasets**, and **eval configs** to the paper. Physical roots on the server are under  
`/work/guanqi/proj5/Inferring_dynamic_physics` (`checkpoints/`, `dataset/`).  
The symlink `backup_dynamicrafer_ckpt` → `checkpoints` matches **`CKPT_ROOT`** in the score docs.

---

## 0. Source of truth (open-source plan)

**All weight–data–config correspondence for publication and Hub packaging follows these internal docs—not ad-hoc paths.**

| Track | Authoritative index | Scoring / repro code | Notes |
|--------|---------------------|----------------------|--------|
| **E — DynamiCrafter** | **`SVD/paper_scores_paths_and_checkpoints.md` §3** (`e729_nd_*`, new-data protocol) | **`SVD_public/calculate_e729_newdata_dc_scores.py`** (same logic as `SVD/calculate_e729_newdata_dc_scores.py` for `e729_nd_*`) | **§2** of that file (`7.29_result_*` scatter protocol) is **not** the main open-source repro path; do **not** treat it as the primary Hub alignment for E. Per-row **`pretrained_checkpoint`**, **result subdirectory**, and **yaml template** are exactly the §3 table. Default config area: `SVD/configs/training_1024_v1.0/e_729_eval_newdata_202603/`. **Exception:** v120 friction absolute **test3** uses the sweep ckpt/path listed in §3 (not the default `epoch=18-step=4500` row for that cell). |
| **B — V-JEPA-2 (mirror)** | **`SVD/paper_scores_B_sources_and_checkpoints.md`** | Mirror yaml under **`SVD/configs/training_1024_v1.0/vjepa_no_red_circle/mirror_eval/`** (generated from `generate_mirror_configs.py` where applicable) | Public **gate**: elasticity **v124_vjepa** / **v125_vjepa**, Test-3, abs + rel (`SVD_public/scripts/verify_paper_elasticity_test3.py`). Full B table: follow the same B doc’s checkpoint + result columns. |

**Numeric authority:** `SVD/paper_scores_E_7.29.md` (E / newdata) and the master / B docs referenced there. If recompute disagrees with the paper table, **fix data, preprocessing, metric code, or ckpt choice**—**not** the published table (`docs/paper_table1_reference.md` remains the external reference).

**Checkpoint filenames (Version E / B paths under `CKPT_ROOT`):** also summarized in **`model_global_map.md`** (sections **E** and **B**); keep Hub layout consistent with §3 / B table, not only the map in isolation.

---

## 1. Dataset layout (24 logical bundles)

Paper convention: **3 properties × 2 task types × 4 splits = 24** (equivalently **4 × 2 × 3**).

| Factors | Values |
|--------|--------|
| Properties | friction, viscosity, elasticity |
| Task type | **absolute**, **relative** |
| Splits | **train**, **test1**, **test2**, **test3** |

**Binding E newdata rows to disk:** for each **`e729_nd_*`** row, the **yaml** named in **`SVD/paper_scores_paths_and_checkpoints.md` §3** defines **`data_dir`** and **`video_list_json`** (and friends). Those paths—plus the **`used_video_list_3.22` / `data_global_map.md` rules** cited in `paper_scores_E_7.29.md`—are the authoritative dataset wiring for **E**. Do not swap in §2 scatter directories for that purpose.

**Rough physical map under `dataset/`** (zips may need unpacking on Hub; names are historical):

- **Friction absolute:** `friction_absolute/` (`4.29_*` test1/2, `7.18_*` test3, etc.)
- **Friction relative:** `friction_relative/` (`5.24_*` train/test1/2, `7.18_*` test3)
- **Viscosity absolute / relative:** `viscosity_absolute/`, `viscosity_relative/`
- **Elasticity absolute / relative:** `elasticity_absolute/`, `elasticity_relative/`

The **minimal public self-check** still uses **`elasticity_absolute_test3`**, **`elasticity_relative_test3`**, and **`data/lists/*.json`** as in `SVD_public/configs/paths.example.yaml` (resolved by `resolve_eval_configs.py`).

---

## 2. Checkpoints — DynamiCrafter (**E**)

### 2.1 Primary (open repro + Hub): **§3 `e729_nd_*`**

Upload weights **exactly** as in **`SVD/paper_scores_paths_and_checkpoints.md` §3** column **`pretrained_checkpoint`** (relative to **`CKPT_ROOT`** = `checkpoints/`), including:

- **v120–v125** main **`_7.29`** files for most rows;
- **v120_ft_7.29** / **v121_ft_7.29** for friction **test3\*** rows;
- **v120 friction absolute test3:** the **staging / sweep** ckpt path given in §3 (not interchangeable with default v120 main ckpt for that metric).

Result JSON layout for scoring with **`calculate_e729_newdata_dc_scores.py`** is **`e729_newdata_20260327/<result subdirectory>`** as in §3, plus the **v120 test3** sweep folder documented there.

### 2.2 Secondary (not main open repro): **§2 `7.29_result_*`**

**§2** lists **`7.29_result_*`** scatter runs and **`calculate_e_729_dc_scores.py`**. Per project policy, this is **out of scope for the primary open-source reproduction path**; optional for archival upload only, clearly labeled **non-primary**.

---

## 3. Checkpoints — V-JEPA-2 mirror (**B**)

Follow **`SVD/paper_scores_B_sources_and_checkpoints.md`** §2 (full table): **checkpoint path**, **result root** (STABLE vs VLIST), and **mirror yaml** name.

- **Six `_vjepa_norm` runs:** v110, v119, v120, v121, v124, v125 (property/task as in that table).
- **Friction test3\*:**
  - v120: **`training_1024_v120_vjepa_ft`** (see B doc for epoch/file);
  - v121 test3\*: **§1.1** may point to a **scratch** path for the paper number—sync that file into `checkpoints/` for Hub or document the exact revision.

**Configs:** `SVD/configs/training_1024_v1.0/vjepa_no_red_circle/mirror_eval/*.yaml` — **`pretrained_checkpoint`** must match the B sources table when regenerating.

---

## 4. Summary counts (full Table 1 style)

| Asset class | Count | Authority |
|-------------|------:|-----------|
| **E — primary repro** | **20** logical **`e729_nd_*` rows** (§3) | Same **6** base `_7.29` dirs + **2** `*_ft_7.29` + **one alternate** v120 test3 ckpt as in §3 |
| **E — §2 scatter (optional)** | separate | **Not** primary Hub alignment |
| **B** | **6** norm + **2** ft (friction test3\*) | `paper_scores_B_sources_and_checkpoints.md` |
| **Dataset (logical)** | **24** cells | 3 × 2 × 4; **per-row** paths from §3 yamls + `data_global_map` / list JSON rules |

**Public repo gate:** **E** v124/v125 + **B** v124/v125, **elasticity Test-3** only — `scripts/verify_paper_elasticity_test3.py`.

---

## 5. Changelog

- **2026-04-11 (b)**: **Source of truth** locked to open-source plan: **E = §3 e729_nd + `calculate_e729_newdata_dc_scores.py`**; **§2 scatter excluded** from primary repro; **B = `paper_scores_B_sources_and_checkpoints.md` + mirror_eval**.
- **2026-04-11 (a)**: Initial manifest; local `checkpoints/` + `dataset/` layout.
