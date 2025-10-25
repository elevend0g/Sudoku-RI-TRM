# Violation Explosion Fix - Summary

## Problem Diagnosed

The RI-TRM Sudoku solver was experiencing catastrophic violation explosion:

**Before Fixes:**
```
Initial violations: 1
Final violations: 58  ← 58× WORSE!
```

The network was rewriting the entire grid at each refinement step, creating massive new violations instead of fixing existing ones.

## Root Causes Identified

### 1. **Unrestricted Grid Rewriting** (`solver.py:266-278`)
The network sampled ALL 81 cells at each step, including:
- Original puzzle clues (should never change)
- Empty cells (should be filled carefully)
- Valid cells not involved in violations (should be preserved)

### 2. **No Cell Masking** (`solver.py:229-278`)
The solver never tracked which cells should be modified vs. preserved.

### 3. **Missing Preservation Loss** (`solver.py:496-548`)
The loss function only encouraged changing violated cells, but didn't penalize changing other cells.

### 4. **Training Reinforced Bad Behavior** (`solver.py:435-480`)
During training, the network sampled entire new grids, learning to be a "grid generator" instead of a "violation fixer."

## Fixes Implemented

### ✅ Fix 1: Track Original Clues (`solver.py:141`)
```python
# Track original clues (non-zero cells) - these should NEVER be modified
self.original_clues = (grid != 0)
```

### ✅ Fix 2: Add Helper to Get Violation Cells (`solver.py:232-246`)
```python
def _get_violation_cells(self, violations: List[Violation]) -> set:
    """Get all cells involved in violations."""
    violation_cells = set()
    for violation in violations:
        for cell in violation.cells:
            violation_cells.add(cell)
    return violation_cells
```

### ✅ Fix 3: Masked Sampling in `_refine_step` (`solver.py:285-308`)
```python
# Create mask: only modify cells involved in violations that are not original clues
violation_cells = self._get_violation_cells(violations)
edit_mask = np.zeros((9, 9), dtype=bool)
for row, col in violation_cells:
    # Only allow editing if NOT an original clue
    if not self.original_clues[row, col]:
        edit_mask[row, col] = True

# Sample improved grid from logits - but ONLY for editable cells
improved_flat = grid_flat.squeeze(0).clone()  # Start with current grid

if edit_mask_flat.any():
    # Only sample cells that are editable
    probs = torch.softmax(logits.squeeze(0)[edit_mask_flat] / 1.0, dim=-1)
    sampled_values = torch.multinomial(probs, num_samples=1).squeeze(-1)
    improved_flat[edit_mask_flat] = sampled_values
```

### ✅ Fix 4: Preservation Loss (`solver.py:584-603`)
```python
# Preservation loss: penalize changing non-violated cells
preservation_loss = 0.0
preservation_count = 0

for pos in range(81):
    if pos not in violation_positions:
        # This cell should be preserved
        target_val = current_grid[0, pos]
        cell_loss = torch.nn.functional.cross_entropy(
            logits[0, pos].unsqueeze(0),
            target_val.unsqueeze(0)
        )
        preservation_loss += cell_loss
        preservation_count += 1

if preservation_count > 0:
    preservation_loss = preservation_loss / preservation_count

# Combine losses with heavy weight on preservation
total_loss = violation_loss + 10.0 * preservation_loss
```

### ✅ Fix 5: Masked Sampling in Training (`solver.py:494-513`)
Same masking approach applied during training to prevent learning bad behavior.

## Results

### Before Fixes
```
Demo 3: 2 violations → 58 violations (29× WORSE!)
```

### After Fixes
```
Demo 3: 2 violations → 2 violations (STABLE!)
```

**Key Improvements:**
- ✅ No more violation explosion
- ✅ Original clues never modified
- ✅ Only violated cells are edited
- ✅ All 163 tests still passing
- ✅ Network learns to be conservative

## What Changed in Code

| File | Lines Changed | Description |
|------|---------------|-------------|
| `src/solver.py:141` | +2 | Track original clues |
| `src/solver.py:232-246` | +15 | Helper to get violation cells |
| `src/solver.py:285-308` | +24 | Masked sampling in inference |
| `src/solver.py:450-513` | +20 | Masked sampling in training |
| `src/solver.py:552-605` | +54 | Preservation loss |
| **Total** | **~115 lines** | **5 targeted fixes** |

## Architectural Insight

The key insight is changing the network's role:

**Before:** Network as "Grid Generator"
```
Input: Grid with violations
Network: "Generate a new 9×9 grid"
Output: Random grid with new violations
```

**After:** Network as "Surgical Debugger"
```
Input: Grid with violations at cells (0,0) and (0,5)
Network: "Change cell (0,5) from 3 to 7"
Output: Same grid, 1 cell changed, violations fixed
```

## Next Steps for Improvement

While the violation explosion is fixed, the network is now too conservative (not reducing violations yet). To improve:

### 1. **Adjust Preservation Weight** (Currently 10.0)
```python
# Try lower values: 3.0, 5.0, 7.0
total_loss = violation_loss + 5.0 * preservation_loss
```

### 2. **Add Violation Delta Reward**
```python
# Reward reducing violations more than current approach
old_count = len(old_violations)
new_count = len(new_violations)
delta = old_count - new_count

if delta > 0:
    # Reduced violations - reward!
    reward_loss = -delta * 2.0
elif delta < 0:
    # Increased violations - heavy penalty!
    reward_loss = -delta * 20.0
else:
    reward_loss = 0.0

total_loss = violation_loss + preservation_loss + reward_loss
```

### 3. **Smarter Violation Cell Editing**
Instead of sampling uniformly from violated cells, prioritize cells that appear in multiple violations:

```python
def _get_violation_cell_weights(self, violations):
    """Cells in more violations should be edited first."""
    cell_counts = {}
    for v in violations:
        for cell in v.cells:
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
    return cell_counts
```

### 4. **Use Constraint Propagation**
When filling a violated cell, only allow values that don't create new violations:

```python
def _get_valid_values(self, grid, row, col):
    """Return values that won't violate constraints."""
    used = set()
    # Check row
    used.update(grid[row, :])
    # Check column
    used.update(grid[:, col])
    # Check box
    box_row, box_col = (row // 3) * 3, (col // 3) * 3
    used.update(grid[box_row:box_row+3, box_col:box_col+3].flatten())

    return [v for v in range(1, 10) if v not in used]
```

## Testing the Fixes

```bash
# Run demo to see improvements
python demo_auto.py

# Run all tests (should all pass)
pytest tests/ -v

# Train on larger dataset to see if network learns
python experiments/train_minimal.py
```

## Files Modified

- `src/solver.py` - All fixes implemented here

## Validation

- ✅ All 163 tests passing
- ✅ No violation explosion
- ✅ Original clues preserved
- ✅ Network behavior is conservative
- ⚠️ Network not yet reducing violations (needs tuning)

---

**Summary:** The critical violation explosion bug is fixed. The network now preserves the grid structure and only modifies cells involved in violations. Further tuning of the preservation weight and violation reward is needed to make the network actively reduce violations.
