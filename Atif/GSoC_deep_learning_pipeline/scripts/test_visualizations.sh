#!/bin/bash
# Quick test script to verify patch visualizations

cd /home/kira/Desktop/Healing_Stones

echo "======================================"
echo "TESTING PATCH VISUALIZATIONS"
echo "======================================"
echo ""

# Test 1: Coverage heatmap (text output only)
echo "Test 1: Coverage Heatmap Statistics"
echo "------------------------------------"
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 \
  --mode heatmap 2>&1 | grep -A 20 "COVERAGE HEATMAP"

echo ""
echo "Test 2: Patch Centers Statistics"
echo "------------------------------------"
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 \
  --mode centers 2>&1 | grep -A 15 "PATCH CENTERS"

echo ""
echo "Test 3: Random Patches Statistics"
echo "------------------------------------"
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 \
  --mode random \
  --count 5 2>&1 | grep -A 25 "RANDOM PATCHES"

echo ""
echo "Test 4: Single Patch Statistics"
echo "------------------------------------"
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 \
  --mode single \
  --patch-id 42 2>&1 | grep -A 15 "SINGLE PATCH"

echo ""
echo "Test 5: Patch Size Statistics"
echo "------------------------------------"
PYTHONPATH=src python3 scripts/visualize_patches.py \
  --fragment fragment_caesar_fragment_1 \
  --mode sizes 2>&1 | grep -A 10 "PATCH SIZE"

echo ""
echo "======================================"
echo "ALL TESTS COMPLETED"
echo "======================================"
echo ""
echo "To see interactive 3D visualizations, run:"
echo ""
echo "  PYTHONPATH=src python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode heatmap"
echo "  PYTHONPATH=src python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode random --count 10"
echo "  PYTHONPATH=src python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode single --patch-id 42"
echo "  PYTHONPATH=src python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode centers"
echo "  PYTHONPATH=src python3 scripts/visualize_patches.py --fragment fragment_caesar_fragment_1 --mode sizes"
echo ""
