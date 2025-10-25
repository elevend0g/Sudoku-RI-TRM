# RI-TRM Sudoku Solver - Progress Report & Context

**Date:** 2025-10-25
**Status:** Violation explosion FIXED, but model now too conservative (needs action-based architecture)

---

## 📊 Timeline of Issues & Fixes

### ❌ **Problem 1: Violation Explosion (FIXED)**

**Symptoms:**
```
Initial violations: 1
Final violations: 58  ← 58× WORSE!
```

**Root Causes:**
1. Network sampled ALL 81 cells at each refinement step
2. No masking to protect original clues or non-violated cells
3. Loss function didn't penalize changing valid cells
4. Training reinforced "grid generator" behavior instead of "violation fixer"

**Fixes Implemented:**
- ✅ Track original clues: `solver.py:141` - `self.original_clues = (grid != 0)`
- ✅ Cell masking in `_refine_step`: `solver.py:285-308` - Only sample violated non-clue cells
- ✅ Helper method `_get_violation_cells`: `solver.py:232-246`
- ✅ Preservation loss: `solver.py:584-620` - Heavy penalty (10.0 → 2.0) for changing non-violated cells
- ✅ Constraint-based loss: `solver.py:622-668` - Guide network toward valid values using KL divergence
- ✅ Violation reduction reward: `solver.py:522-544` - Reward progress, penalize regression

**Results:**
```
Before: 1 → 58 violations (EXPLOSION)
After:  1 → 1 violations (STABLE) ✅
```

All 163 tests still passing ✅

---

### ⚠️ **Problem 2: Model Too Conservative (CURRENT)**

**Symptoms:**
```
Step 0: 2 violations, 0 cells changed
Step 1: 2 violations, 0 cells changed
Step 2: 2 violations, 0 cells changed
...
Step 8: 2 violations, 0 cells changed
```

**Root Cause:**
The model is **literally doing nothing**. Grid remains 100% identical across all iterations.

**Suspected Issues:**

1. **Both violated cells might be original clues** → No editable cells
2. **Network predicting same values as input** → Multinomial sampling returns same
3. **Preservation weight still too high** → Even at 2.0, might be crushing action signals
4. **Fundamental architecture problem** → Grid-to-grid prediction doesn't force changes

---

## 🔧 Fixes Applied (Session 2)

### **1. Reduced Preservation Weight**
```python
# Before: 10.0 (too conservative)
# After: 2.0 (more balanced)
total_loss = violation_loss + 2.0 * preservation_loss
```
**Location:** `src/solver.py:617-619`

### **2. Added Violation Reduction Reward**
```python
violation_delta = initial_violation_count - len(new_violations)
if violation_delta > 0:
    # Reduced violations - reward
    reward = -violation_delta * 5.0
elif violation_delta < 0:
    # Increased violations - heavy penalty
    penalty = -violation_delta * 20.0
```
**Location:** `src/solver.py:522-532`

### **3. Improved Violation Loss with Constraint Guidance**
```python
# Get valid values that won't create new violations
valid_values = self._get_valid_values(current_grid_np, row, col)

if len(valid_values) > 0:
    # Create target distribution favoring valid values
    target_dist = torch.zeros(10, device=self.device)
    for v in valid_values:
        target_dist[v] = 1.0 / len(valid_values)

    # KL divergence to encourage valid value distribution
    kl_loss = torch.nn.functional.kl_div(...)
```
**Location:** `src/solver.py:637-654`

### **4. Added Constraint Checker**
```python
def _get_valid_values(self, grid, row, col):
    """Return values 1-9 that won't violate Sudoku constraints"""
    used = set()
    used.update(grid[row, :])      # Row
    used.update(grid[:, col])       # Column
    used.update(box_values)         # 3x3 box
    used.discard(0)
    return set(range(1, 10)) - used
```
**Location:** `src/solver.py:248-275`

---

## 🎯 Next Steps: Action-Based Architecture (RECOMMENDED)

The fundamental issue is the **grid-to-grid prediction paradigm**. The network can learn to copy the input perfectly (safest option), resulting in no changes.

### **Proposed Solution: Action-Based RI-TRM**

**Core Concept:**
```
CURRENT (broken):
  Input: Grid [9×9]
  Output: New Grid [9×9]
  Problem: Output ≈ Input (too similar, no change)

PROPOSED (will work):
  Input: Grid [9×9] + Violations
  Output: Action = (cell_index, new_value)
  Guarantee: Exactly ONE cell changes per step
```

### **Architecture Design**

#### **Action Network:**
```python
class ActionBasedNetwork(nn.Module):
    def forward(self, grid, violations_mask, candidate_mask):
        # Process grid through transformer
        z = self.transformer(grid)  # [batch, 81, hidden]

        # Pool to single vector
        pooled = z.mean(dim=1)  # [batch, hidden]

        # Predict which cell to change (0-80)
        cell_logits = self.cell_selector(pooled)  # [batch, 81]
        cell_logits = cell_logits.masked_fill(~candidate_mask, -1e9)

        # Predict what value to use (1-9)
        value_logits = self.value_selector(pooled)  # [batch, 9]

        return cell_probs, value_probs
```

#### **Action Solver:**
```python
class ActionBasedSolver:
    def refine_one_step(self, grid, violations):
        # Select action from network
        cell_idx, new_value = self.network.select_action(grid, violations)

        # Apply action - GUARANTEED to change grid
        new_grid = grid.clone()
        new_grid[cell_idx] = new_value

        # Assertion: Grid MUST be different
        assert not torch.equal(grid, new_grid)

        return new_grid
```

#### **Loss Function:**
```python
def compute_loss(old_violations, new_violations):
    delta = len(new_violations) - len(old_violations)

    if delta < 0:
        # Reduced violations - reward!
        loss = -delta * 2.0
    elif delta == 0:
        # No progress - small penalty
        loss = 1.0
    else:
        # Made worse - penalty
        loss = delta * 3.0

    return loss
```

### **Benefits:**
1. ✅ **Guaranteed progress** - Grid changes every step
2. ✅ **Interpretable** - "Change cell (3,5) from 3 to 7"
3. ✅ **Trainable** - Clear reward signal (violation count delta)
4. ✅ **Efficient** - Only predict 2 numbers, not 81
5. ✅ **Path memory compatible** - Action = (cell, value) is perfect for memory

---

## 📁 Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/solver.py` | ~200 lines | All fixes implemented here |
| - Line 141 | +2 | Track original clues |
| - Lines 232-246 | +15 | Get violation cells helper |
| - Lines 248-275 | +28 | Get valid values constraint checker |
| - Lines 285-308 | +24 | Masked sampling in inference |
| - Lines 450-544 | +95 | Masked training + violation reward |
| - Lines 617-621 | ~5 | Reduced preservation weight to 2.0 |
| - Lines 622-668 | +47 | Constraint-guided violation loss |

**New files created:**
- `FIXES_SUMMARY.md` - Detailed analysis of violation explosion fix
- `test_fix_demo.py` - Demonstration script
- `test_fixable_violation.py` - Validation test
- `test_diagnosis.py` - Diagnostic tool (not yet run)
- `PROGRESS_REPORT.md` - This file

---

## 🧪 Test Results

### **All Tests Passing**
```bash
pytest tests/ -v
# 163 passed in 11.65s ✅
```

### **Demo Results (Current)**
```bash
python demo_auto.py

Demo 3: Recursive Refinement Solving
  Initial violations: 1
  Final violations: 1
  Iterations: 5
  ✓ Solving attempted

# Grid is STABLE but NOT improving
```

### **What Works:**
✅ No violation explosion
✅ Original clues preserved
✅ All tests passing
✅ Safe, conservative behavior

### **What Doesn't Work:**
❌ Model doesn't reduce violations
❌ Grid remains unchanged across iterations
❌ Network too paralyzed to act

---

## 🚀 Implementation Plan for Action-Based Architecture

### **Step 1: Create New Files**

**`src/action_network.py`:**
- `ActionBasedNetwork` class
- Outputs (cell_index, value) instead of full grid
- Uses existing transformer backbone

**`src/action_solver.py`:**
- `ActionBasedSolver` class
- Applies discrete actions to grid
- Guarantees exactly 1 cell changes per step
- Includes assertion checks

**`src/action_trainer.py`:**
- `ActionBasedTrainer` class
- Simple reward: -Δviolations
- Curriculum learning (start easy)
- Epsilon-greedy exploration

### **Step 2: Update Demo**
```python
# demo_action.py
from src.action_network import ActionBasedNetwork
from src.action_solver import ActionBasedSolver
from src.action_trainer import ActionBasedTrainer

network = ActionBasedNetwork(hidden_size=512)
solver = ActionBasedSolver(network, rule_graph)
trainer = ActionBasedTrainer(solver, dataset, rule_graph)

# Train
for epoch in range(10):
    stats = trainer.train_epoch(num_tasks=20)
    print(f"Epoch {epoch}: {stats}")

# Test
result, trace, solved = solver.solve(test_grid, verbose=True)
```

### **Step 3: Expected Output**
```
Epoch 0: loss=3.2, violations=4.5→2.1, success=15%
Epoch 1: loss=2.8, violations=4.3→1.8, success=25%
...
Epoch 9: loss=1.2, violations=3.1→0.5, success=65%

Test:
  Step 0: 2 violations → changed (0,8) 5→3 → 1 violation ✓
  Step 1: 1 violation → changed (3,0) 5→2 → 0 violations ✓
  SOLVED in 2 steps!
```

---

## 🔍 Diagnostic Questions to Answer Next Session

1. **Why is grid unchanged?**
   - Are both violated cells original clues? (edit mask empty?)
   - Is network predicting same values? (check logits)
   - Is multinomial sampling broken? (check temperature)

2. **Where exactly does nothing happen?**
   - In `_refine_step` when sampling?
   - In solver loop when no cells are editable?
   - In network forward pass?

3. **Can we fix current architecture or need redesign?**
   - If edit_mask has cells → fix sampling
   - If edit_mask is empty → need better cell selection
   - If network always predicts input → need action-based

**Run this to diagnose:**
```bash
python test_diagnosis.py
# Will show exactly where the problem is
```

---

## 💾 Code Snippets for Quick Reference

### **Check if cells are editable:**
```python
violation_cells = solver._get_violation_cells(violations)
edit_mask = np.zeros((9, 9), dtype=bool)
for row, col in violation_cells:
    if not solver.original_clues[row, col]:
        edit_mask[row, col] = True

num_editable = np.sum(edit_mask)
print(f"Editable cells: {num_editable}")
```

### **Check network predictions:**
```python
grid_tensor = torch.from_numpy(grid.flatten()).long().unsqueeze(0)
violations_tensor = solver._violations_to_tensor(violations)

with torch.no_grad():
    logits, conf, _ = network(grid_tensor, violations_tensor, None)

for row, col in violation_cells:
    pos = row * 9 + col
    probs = torch.softmax(logits[0, pos], dim=0)
    predicted = torch.argmax(probs).item()
    current = grid[row, col]
    print(f"Cell ({row},{col}): current={current}, predicted={predicted}")
```

### **Force temperature in sampling:**
```python
# In _refine_step, change temperature from 1.0 to higher
probs = torch.softmax(logits.squeeze(0)[edit_mask_flat] / 2.0, dim=-1)  # temp=2.0
# Higher temp = more exploration
```

---

## 📚 Key Insights

### **Violation Explosion Fix Worked Because:**
1. We constrained which cells can be modified
2. We added preservation penalty for non-violated cells
3. We tracked original clues to protect puzzle structure

### **Model is Conservative Because:**
1. Preservation penalty might still be too high (even at 2.0)
2. Network might not have enough training to learn fixes
3. Grid-to-grid architecture allows "do nothing" as valid output
4. No guarantee of grid change in sampling process

### **Action-Based Will Work Because:**
1. Forces exactly one cell to change per step
2. Can't learn "do nothing" strategy
3. Clear reward signal (fewer violations = good)
4. Much simpler to train (predict 2 numbers not 81)

---

## 🎓 Lessons Learned

1. **Conservative > Destructive** - Better to do nothing than explode violations
2. **Architecture Matters** - Grid-to-grid allows copying, action-based forces progress
3. **Loss Balance is Hard** - Too high preservation = paralysis, too low = chaos
4. **Test All the Way** - All 163 tests passing ≠ system working on real tasks
5. **Interpretability Helps** - Action-based is naturally interpretable

---

## 📖 References for Next Session

**Key Files:**
- `src/solver.py` - All fixes implemented here
- `src/network.py` - Transformer backbone (working fine)
- `src/rules.py` - Violation checking (working fine)
- `demo_auto.py` - Shows current behavior

**Key Methods:**
- `RecursiveSolver.solve()` - Main solving loop
- `RecursiveSolver._refine_step()` - Where grid should change
- `TrainableSolver.train_step()` - Training loop
- `TrainableSolver._violation_loss()` - Loss computation

**Tests to Run:**
```bash
pytest tests/test_solver.py -v  # Should pass
python demo_auto.py              # Shows conservative behavior
python test_diagnosis.py         # Diagnoses root cause
```

---

## 🎯 Recommendation

**Implement action-based architecture.** The current grid-to-grid approach has fundamental issues that are hard to fix with just hyperparameter tuning. An action-based design:
- Guarantees progress
- Is easier to train
- Is more interpretable
- Matches the RI-TRM philosophy better (iterative refinement = sequence of actions)

The transformer backbone (`TinyRecursiveNetwork`) is fine and can be reused. We just need to change the output heads and solver logic.

---

**End of Progress Report**
