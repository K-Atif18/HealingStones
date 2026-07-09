# Healing Stones — Quick Start

One-page command cheat sheet for Phase 1 & 2.

**Project:** `/home/kira/Desktop/Healing_Stones`

---

## 🚀 Run Everything

```bash
cd /home/kira/Desktop/Healing_Stones

# Phase 1: Dataset Foundation
PYTHONPATH=src python3 -m dataset_foundation.pipeline --config config/default.yaml

# Phase 2: Patch Generation
PYTHONPATH=src python3 -m patch_generation.pipeline --config config/patch_generation.yaml

# Phase 3: Ground Truth Generation
PYTHONPATH=src python3 scripts/phase3_pipeline.py --config config/ground_truth.yaml

# Phase 4: Baseline Geometry (FPFH/SHOT descriptors + retrieval + registration)
PYTHONPATH=src python3 -m baseline_geometry.pipeline --config config/baseline_geometry.yaml

# Tests
python3 -m pytest tests/ -v

# Visualize
PYTHONPATH=src python3 scripts/visualize.py --interactive
PYTHONPATH=src python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode heatmap
```

---

## 📊 Check Status

```bash
cd /home/kira/Desktop/Healing_Stones

# Phase 1 status
cat dataset/metadata/dataset.json | python3 -m json.tool | grep -A 5 "reconstruction"

# Phase 2 status
cat patches/dataset.json | python3 -m json.tool | grep "overall_validation_pass"

# Phase 4 status (validation gate)
python3 -c "import json; d=json.load(open('baseline_results/baseline_report.json')); print('gate passed:', d['validation']['passed'])"
cat baseline_results/retrieval/retrieval_summary.json | python3 -m json.tool

# Count outputs
ls -1 dataset/normalized/*.ply | wc -l    # Should be 7
ls -1 patches/*/patches.npz | wc -l       # Should be 7
ls -1 baseline_results/descriptors/fpfh/*.npz | wc -l  # Should be 7
```

---

## 🎨 Visualizations

```bash
cd /home/kira/Desktop/Healing_Stones

# Phase 1: Aligned fragments
PYTHONPATH=src python3 scripts/visualize.py --interactive

# Phase 2: Patch coverage heatmap
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 --mode heatmap

# Phase 2: Random patches
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 --mode random --count 10

# Phase 2: Single patch detail
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 --mode single --patch-id 42
```

---

## 🧪 Testing

```bash
cd /home/kira/Desktop/Healing_Stones

# All tests (136 total)
python3 -m pytest tests/ -v

# Phase 1 only (85 tests)
python3 -m pytest tests/ -v -k "not patch_generation"

# Phase 2 only (51 tests)
python3 -m pytest tests/patch_generation/ -v

# Phase 4 only (14 tests)
PYTHONPATH=src python3 -m pytest tests/baseline_geometry/ -v

# Specific test
python3 -m pytest tests/test_aligner.py -v
```

---

## 🔍 Inspect Data

```bash
cd /home/kira/Desktop/Healing_Stones

# View metadata
cat dataset/metadata/dataset.json | python3 -m json.tool
cat patches/dataset.json | python3 -m json.tool

# Count points per fragment
for i in {1..7}; do
  python3 -c "
import sys; sys.path.insert(0, 'src')
from dataset_foundation.ply_io import read_point_cloud
pc = read_point_cloud('dataset/normalized/fragment_caesar_fragment_$i.ply')
print(f'Fragment $i: {len(pc.points)} points')
"
done

# Load patches in Python
python3 -c "
import sys; sys.path.insert(0, 'src')
from patch_generation.patch_record import read_patch_records
patches = read_patch_records('patches/fragment_caesar_fragment_1/patches.npz')
print(f'{len(patches)} patches loaded')
print(f'First patch: {len(patches[0].global_coords)} points')
"
```

---

## 🗑️ Clean & Rebuild

```bash
cd /home/kira/Desktop/Healing_Stones

# Clean Phase 2 only (keep Phase 1)
rm -rf patches/
PYTHONPATH=src python3 -m patch_generation.pipeline --config config/patch_generation.yaml

# Clean everything
rm -rf dataset/ patches/ .pytest_cache/ .hypothesis/

# Rebuild from scratch
PYTHONPATH=src python3 -m dataset_foundation.pipeline --config config/default.yaml
PYTHONPATH=src python3 -m patch_generation.pipeline --config config/patch_generation.yaml
python3 -m pytest tests/ -v
```

---

## ⚙️ Configuration

**Phase 1:** `config/default.yaml`
- `voxel_size_mm: 1.5` — Downsampling resolution
- `alignment_error_threshold_mm: 5.0` — Max acceptable RMSE
- `precomputed_transforms:` — Manual alignment overrides

**Phase 2:** `config/patch_generation.yaml`
- `patch_radius_mm: 8.0` — Neighborhood radius
- `max_patch_points: 128` — Size cap
- `target_center_count: 1000` — FPS centers per fragment
- `coverage_threshold: 1.0` — Require 100% coverage

---

## 📁 File Structure

```
Healing_Stones/
├── data/                        # Source data (7 fragments + full model)
├── dataset/                     # Phase 1 outputs
│   ├── fragments/              # Aligned PLY files
│   ├── transforms/             # 4×4 matrices
│   ├── normals/                # PLY with normals
│   ├── normalized/             # Density-standardized
│   └── metadata/               # JSON records
├── patches/                     # Phase 2 outputs
│   ├── dataset.json            # Dataset-level metadata
│   └── fragment_*/             # Per-fragment patches
│       ├── patches.npz         # Compressed patch data
│       └── metadata.json       # Coverage/overlap stats
├── config/                      # Configuration files
├── scripts/                     # Utilities
│   ├── visualize.py            # Phase 1 visualization
│   ├── visualize_patches.py    # Phase 2 visualization
│   └── manual_align.py         # Assisted alignment
├── src/                         # Source code
│   ├── dataset_foundation/     # Phase 1 modules
│   └── patch_generation/       # Phase 2 modules
└── tests/                       # Test suite
```

---

## 🐛 Troubleshooting

**ImportError?**
```bash
export PYTHONPATH=/home/kira/Desktop/Healing_Stones/src
# Or use inline: PYTHONPATH=src python3 ...
```

**Display not found?**
```bash
echo $DISPLAY  # Check if X11 available
# Use --save instead of --interactive for visualizations
```

**Out of memory?**
```bash
ulimit -v 8000000  # 8 GB limit
# Or increase voxel_size_mm in config
```

---

## 📋 Expected Results

### Phase 1
- **7 aligned fragments** (dataset/normalized/)
- **Reconstruction RMSE:** ~4.60 mm (< 5.0 threshold) ✅
- **Coverage:** 100% ✅
- **85 tests passing** ✅

### Phase 2
- **6,822 patches** across 7 fragments
- **Coverage:** 100% on all fragments ✅
- **Mean overlap:** ~27% ✅
- **Mean patch size:** ~99 points ✅
- **51 tests passing** ✅

### Phase 4
- **FPFH (33-D) + SHOT (352-D)** descriptors for all **6,822 patches** ✅
- **Pair separability:** FPFH ROC-AUC **0.708**, SHOT **0.536** (beat chance) ✅
- **Ranking retrieval:** FPFH mAP **0.109**, P@1 **0.13** (cross-fragment)
- **Registration:** contact-seeded recovery **90.9%** (10/11) vs **0%** control ✅
- **Global (unaided) registration fails** (0% adjacent) — expected: fragments abut
- **Validation gate PASSED (4/4)** ✅ · **14 tests passing** ✅

---

## 🔗 See Full Documentation

- **COMMANDS.md** — Complete command reference
- **RESUME.md** — Project status and next steps
- **PHASE2_COMPLETE.md** — Phase 2 completion report
- **PHASE3_COMPLETE.md** — Phase 3 completion report
- **PHASE4_COMPLETE.md** — Phase 4 completion report

---

**Status:** Phase 1 ✅ | Phase 2 ✅ | Phase 3 ✅ | Phase 4 ✅ | Phase 5 ⏳ Next
