# Healing Stones — Command Reference

 PYTHONPATH=src python3 scripts/visualize_phase3.py --mode contacts
 ❯ PYTHONPATH=src python3 scripts/visualize_phase3.py --mode positives --pair 6 7 --count 5

bash
cd /home/kira/Desktop/Healing_Stones

# All 7 windows for one fragment (change 1 → 2..7)
PYTHONPATH=src python3 scripts/explore_fragment.py --fragment 1

# Only some windows, e.g. contact + registration for fragment 5
PYTHONPATH=src python3 scripts/explore_fragment.py --fragment 5 --viz 2 5

# Use SHOT instead of FPFH for retrieval / hard-negatives
PYTHONPATH=src python3 scripts/explore_fragment.py --fragment 4 --descriptor shot


Windows open one at a time — the 5 geometry views are fully rotatable/zoomable Open3D windows, the 2 abstract plots are matplotlib windows. Close 
each window to advance to the next. Every window also prints a "WHAT TO LOOK FOR" checklist to the terminal.

## The 7 windows and how to validate each

[1] Distinctiveness — fragment coloured red/yellow/blue/gray by patch class.
→ Correct if: red (distinctive) sits on edges, ridges, irregular features; gray/blue (flat/ambiguous) fills smooth interior areas. Wrong if red is
random noise unrelated to shape.

[2] Contact regions — gray fragment with each neighbour's contact zone in a distinct colour.
→ Correct if: each coloured zone lies on a break/fracture face, not the outer sculpted surface. Big solid patch = strong interface; thin line/
speck = weak. The terminal flags tiny contacts (e.g. F5↔F7 = 5 points → "TINY / suspect").

[3] Retrieval — rows of: black query patch (from this fragment) + its 8 nearest patches by descriptor distance. Green = true assembly partner, 
red = false positive.
→ Correct/expected: mostly red, few green, green rarely first. That IS the finding — similarity ≠ assembly compatibility. Look at what the reds 
are: flat blobs matching flat blobs, curves matching curves.

[4] Hard negatives — rows of: blue patch (this fragment) + orange patch (a fragment it does not touch) that FPFH thinks are near-identical.
→ Correct if: the two patches look almost the same yet can't possibly assemble. These are the Phase 5 hard negatives.

[5] Registration — gray anchor fragment with each neighbour perturbed and recovered via its contact interface, printed with rot/trans error.
→ Correct if: large-contact neighbours click into place hugging the gray fragment (low error, "OK ✓"); thin-contact neighbours float off ("FAIL ✗"
). Cross-check the failures against their small contact size in window [2].

[6] Embedding — PCA + t-SNE, this fragment's patches in red vs all others in gray.
→ Good if: red is spread throughout the gray cloud (transferable local geometry). Warning sign: red forms its own isolated island (descriptor is 
memorising fragment identity — F6/F7 tend to do this).

[7] Adjacency graph — this fragment gold, neighbours orange, edges labelled with #positives and contact points.
→ Correct if: the orange neighbours match the contact zones you saw in window [2]. Red edge = a real adjacency too small to use (5↔7).


# Interactive 3D viewer (requires X11/display)
PYTHONPATH=src python3 scripts/visualize.py --interactive


### 2.4 Load Patches in Python

```bash
cd /home/kira/Desktop/Healing_Stones

# Load and inspect patches
python3 -c "
import sys
sys.path.insert(0, 'src')
from patch_generation.patch_record import read_patch_records

# Load patches from fragment 1
patches = read_patch_records('patches/fragment_caesar_fragment_1/patches.npz')

print(f'Loaded {len(patches)} patches')
print(f'First patch: {patches[0].patch_id}')
print(f'  Fragment: {patches[0].fragment_id}')
print(f'  Points: {len(patches[0].global_coords)}')
print(f'  Center: {patches[0].center}')
print(f'  Has normals: {len(patches[0].normals)} vectors')
print(f'  Source indices: {len(patches[0].source_indices)} mappings')
"
```

---

## Visualization

### Phase 1 Visualization

```bash
cd /home/kira/Desktop/Healing_Stones

# Interactive 3D viewer (shows all 7 fragments aligned to model)
PYTHONPATH=src python3 scripts/visualize.py --interactive

# Save as image
PYTHONPATH=src python3 scripts/visualize.py --save
# Output: dataset/visualization.png
```

### Phase 2 Visualization

All Phase 2 visualizations use: `scripts/visualize_patches.py`

#### 1. Coverage Heatmap (MOST IMPORTANT)
```bash
cd /home/kira/Desktop/Healing_Stones

# Show coverage for fragment 1
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 \
  --mode heatmap

# Colors: Blue (low coverage) → Red (high coverage)
```

#### 2. Random Patches
```bash
cd /home/kira/Desktop/Healing_Stones

# Show 10 random patches (each in different color)
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 \
  --mode random \
  --count 10
```

#### 3. Single Patch Detail
```bash
cd /home/kira/Desktop/Healing_Stones

# Show patch #42 with normals
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 \
  --mode single \
  --patch-id 42
```

#### 4. All Patch Centers
```bash
cd /home/kira/Desktop/Healing_Stones

# Show FPS-sampled center distribution
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 \
  --mode centers
```

#### 5. Patch Size Distribution
```bash
cd /home/kira/Desktop/Healing_Stones

# Show patches colored by number of points
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 \
  --mode sizes

# Colors: Blue (small) → Red (large)
```

### Batch Visualization

```bash
cd /home/kira/Desktop/Healing_Stones

# Test visualization script (text output only)
bash scripts/test_visualizations.sh

# Visualize all fragments (heatmap mode)
for i in {1..7}; do
  echo "=== Fragment $i ==="
  PYTHONPATH=src python3 scripts/visualize_patches.py \
    --fragment fragment_caesar_fragment_$i \
    --mode heatmap
done
```

---

### 4.7 Interactive per-fragment explorer (opens rotatable windows)

Opens each diagnostic in its OWN window, ONE fragment at a time. Close each
window to advance to the next. Requires a display (X11/`$DISPLAY`).

```bash
cd /home/kira/Desktop/Healing_Stones

# All 7 windows for fragment 1 (change --fragment 1..7)
PYTHONPATH=src python3 scripts/explore_fragment.py --fragment 1

# Only specific windows (e.g. contact + registration) for fragment 5
PYTHONPATH=src python3 scripts/explore_fragment.py --fragment 5 --viz 2 5

# Drive retrieval / hard-negatives with SHOT instead of FPFH
PYTHONPATH=src python3 scripts/explore_fragment.py --fragment 4 --descriptor shot
```

Window index: 1 distinctiveness · 2 contact · 3 retrieval · 4 hard-negatives ·
5 registration · 6 embedding · 7 adjacency graph. Each window prints a
"WHAT TO LOOK FOR" checklist to the terminal so you can validate it by eye.

---

## Quick Reference

### Phase 1: End-to-End

```bash
cd /home/kira/Desktop/Healing_Stones
PYTHONPATH=src python3 -m dataset_foundation.pipeline --config config/default.yaml
PYTHONPATH=src python3 scripts/visualize.py --interactive
python3 -m pytest tests/ -v -k "not patch_generation"
```

### Phase 2: End-to-End

```bash
cd /home/kira/Desktop/Healing_Stones
PYTHONPATH=src python3 -m patch_generation.pipeline --config config/patch_generation.yaml
PYTHONPATH=src python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode heatmap
python3 -m pytest tests/patch_generation/ -v
```

### Full System: Phase 1 + 2

```bash
cd /home/kira/Desktop/Healing_Stones

# Run both pipelines
PYTHONPATH=src python3 -m dataset_foundation.pipeline --config config/default.yaml
PYTHONPATH=src python3 -m patch_generation.pipeline --config config/patch_generation.yaml

# Run all tests
python3 -m pytest tests/ -v

# Visualize results
PYTHONPATH=src python3 scripts/visualize.py --interactive
PYTHONPATH=src python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode heatmap
```

### Status Check

```bash
cd /home/kira/Desktop/Healing_Stones

echo "=== PHASE 1 STATUS ==="
if [ -f "dataset/metadata/dataset.json" ]; then
  echo "✓ Phase 1 complete"
  cat dataset/metadata/dataset.json | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"  Reconstruction RMSE: {d['reconstruction']['rmse_mm']:.2f} mm\"); print(f\"  Status: {'PASSED' if d['reconstruction']['passed'] else 'FAILED'}\")"
else
  echo "✗ Phase 1 not run"
fi

echo ""
echo "=== PHASE 2 STATUS ==="
if [ -f "patches/dataset.json" ]; then
  echo "✓ Phase 2 complete"
  cat patches/dataset.json | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"  Total patches: {sum(f['patch_count'] for f in d['fragments'].values())}\"); print(f\"  Validation: {'PASSED' if d['overall_validation_pass'] else 'FAILED'}\")"
else
  echo "✗ Phase 2 not run"
fi
```

### Clean and Rebuild

```bash
cd /home/kira/Desktop/Healing_Stones

# Clean Phase 2 outputs only (keep Phase 1)
rm -rf patches/
PYTHONPATH=src python3 -m patch_generation.pipeline --config config/patch_generation.yaml

# Clean everything and rebuild from scratch
rm -rf dataset/ patches/
PYTHONPATH=src python3 -m dataset_foundation.pipeline --config config/default.yaml
PYTHONPATH=src python3 -m patch_generation.pipeline --config config/patch_generation.yaml

# Clean test cache
rm -rf .pytest_cache/ .hypothesis/
python3 -m pytest tests/ -v
```

---

## Troubleshooting

### ImportError: cannot import name 'X'

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/home/kira/Desktop/Healing_Stones/src
echo $PYTHONPATH

# Or use inline:
PYTHONPATH=src python3 your_command.py
```

### Display not found (X11)

```bash
# Check if display is available
echo $DISPLAY

# If empty, use X11 forwarding (if SSH)
ssh -X user@host

# Or save visualization instead of interactive
PYTHONPATH=src python3 scripts/visualize.py --save
```

### Out of Memory

```bash
# Increase voxel size (reduces point count)
# Edit config/default.yaml:
#   voxel_size_mm: 1.5  →  2.0

# Or run with memory limit
ulimit -v 8000000
```

### Tests Failing

```bash
# Run with verbose output
python3 -m pytest tests/ -v -s

# Run specific failing test
python3 -m pytest tests/test_aligner.py::test_specific_function -v -s

# Clear cache and retry
rm -rf .pytest_cache/ .hypothesis/
python3 -m pytest tests/ -v
```

---

## Configuration Files

### Phase 1: `config/default.yaml`

Key parameters:
- `voxel_size_mm`: 1.5 (downsampling resolution)
- `alignment_error_threshold_mm`: 5.0
- `reconstruction_error_threshold_mm`: 5.0
- `precomputed_transforms`: Manual alignment overrides

### Phase 2: `config/patch_generation.yaml`

Key parameters:
- `patch_radius_mm`: 8.0 (neighborhood size)
- `max_patch_points`: 128 (size cap)
- `target_center_count`: 1000 (FPS centers per fragment)
- `fps_seed`: 42 (reproducibility)
- `coverage_threshold`: 1.0 (require 100% coverage)

---

## Next Steps

After Phase 1 & 2 are complete, proceed to:

**Phase 3: Ground Truth Generation**
- Detect fragment adjacencies
- Identify contact regions
- Generate positive/negative patch pairs

---

**Last updated:** 2026-07-09  
**Project status:** Phase 1 ✅ | Phase 2 ✅ | Phase 3 ✅ | Phase 4 ✅ | Phase 5 ⏳ Next
