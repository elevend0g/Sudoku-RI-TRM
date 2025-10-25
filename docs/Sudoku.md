# RI-TRM Implementation Plan for Claude Code

This is a step-by-step plan to build a rule-based RI-TRM for Sudoku that trains on rules, not examples.

---

## 🎯 Project Goal

Build a minimal RI-TRM that:

- Uses explicit Sudoku rules (K_R) for zero-shot verification
- Trains by learning to fix rule violations (not match examples)
- Implements Hebbian path memory (K_P) to learn fixing strategies
- Uses TRM's recursive refinement architecture

---

## 📁 File Structure

```
ri-trm-sudoku/
├── src/
│   ├── __init__.py
│   ├── rules.py              # K_R: Explicit Sudoku rules
│   ├── path_memory.py        # K_P: Hebbian path learning
│   ├── network.py            # 2-layer recursive network
│   ├── solver.py             # Recursive refinement algorithm
│   ├── dataset.py            # Rule-based task generation
│   ├── trainer.py            # Training loop
│   └── utils.py              # Helpers
├── tests/
│   ├── test_rules.py
│   ├── test_path_memory.py
│   ├── test_network.py
│   └── test_integration.py
├── experiments/
│   ├── train_minimal.py      # Phase 1: Basic training
│   ├── train_with_memory.py  # Phase 2: With path memory
│   └── visualize.py          # Show traces & learning
├── requirements.txt
└── README.md
```

---

## 📋 Phase 1: Core Rule Verification (K_R)

### Task 1.1: Implement Sudoku Rule Graph

**File**: `src/rules.py`

```python
"""
Explicit Sudoku rules - no training needed.
This is K_R (Layer 2) in RI-TRM architecture.

Requirements:
1. Class SudokuRuleGraph with verify(grid) method
2. Returns list of Violation objects with:
   - type: str (row_duplicate, col_duplicate, box_duplicate)
   - location: tuple (row, col) or (box_num)
   - conflicting_values: list[int]
3. Must work zero-shot (no training)

Grid format: numpy array (9, 9) with values 0-9 (0 = empty)
"""

# Expected behavior:
grid = np.array([
    [1, 2, 1, ...],  # Row has duplicate 1
    ...
])
violations = rule_graph.verify(grid)
assert len(violations) > 0
assert violations[0].type == "row_duplicate"
```

**Test Criteria**:

- Test on valid grid → 0 violations
- Test on grid with row duplicate → detects it
- Test on grid with col duplicate → detects it
- Test on grid with box duplicate → detects it
- Test on empty grid → 0 violations (not filled yet)

---

### Task 1.2: Implement Rule-Based Dataset

**File**: `src/dataset.py`

```python
"""
Generate training tasks by creating grids with violations.
Train on fixing violations, NOT matching ground truth.

Requirements:
1. Class RuleBasedSudokuDataset
2. generate_task() returns:
   {
       "grid": np.array (9, 9),  # Potentially has violations
       "violations": list[Violation],  # From rule_graph.verify()
   }
3. No ground truth complete grids needed!

Task generation strategy:
- Start with empty grid
- Randomly fill 20-35 cells
- Allow violations to occur naturally
- Rules tell us what's wrong
"""

# Expected behavior:
dataset = RuleBasedSudokuDataset(rule_graph)
task = dataset.generate_task()
assert task["grid"].shape == (9, 9)
assert len(task["violations"]) >= 0  # May have violations
```

**Test Criteria**:

- Generates diverse tasks
- Can generate tasks with 0 violations (valid partial grids)
- Can generate tasks with violations
- Tasks are solvable (not impossible configurations)

---

## 📋 Phase 2: Basic TRM Network

### Task 2.1: Implement 2-Layer Recursive Network

**File**: `src/network.py`

```python
"""
Tiny 2-layer transformer network from TRM paper.

Requirements:
1. 2 transformer layers (not 4!)
2. Hidden dim: 512
3. 8 attention heads
4. RMSNorm, no bias, RoPE, SwiGLU
5. Input: (x, y, z, violations_embedding)
   - x: embedded task specification
   - y: current solution
   - z: reasoning latent
   - violations: embedded violation info
6. Output: updated z (reasoning latent)

Parameter count should be ~7M
"""

# Expected behavior:
network = TinyRecursiveNetwork(
    hidden_size=512,
    num_layers=2,
    num_heads=8,
    vocab_size=10  # digits 0-9
)
assert network.count_parameters() < 10_000_000
```

**Test Criteria**:

- Parameters < 10M
- Forward pass works with expected shapes
- Can process violation information
- Gradients flow correctly

---

### Task 2.2: Implement Recursive Solver

**File**: `src/solver.py`

```python
"""
Recursive refinement algorithm from RI-TRM paper (Algorithm 1).

Requirements:
1. Class RecursiveSolver
2. refine(grid, max_iterations=16) method
3. For each iteration:
   a. Verify current grid using rules (K_R)
   b. If no violations, return (success!)
   c. Query path memory for candidate fixes (K_P)
   d. Recursive reasoning: n=6 steps through network
   e. Generate improved grid
   f. Update path memory based on success
4. Return: (final_grid, trace, converged)

Trace should include:
- Step number
- Violations at that step
- Action taken
- Success/failure
"""

# Expected behavior:
solver = RecursiveSolver(network, rule_graph, path_memory=None)
result = solver.refine(broken_grid)
assert result.converged or result.num_steps == 16
assert len(result.violations) <= initial_violations
```

**Test Criteria**:

- Can reduce violations over iterations
- Stops early if grid becomes valid
- Trace is interpretable
- Works without path_memory (random actions)

---

## 📋 Phase 3: Rule-Based Training

### Task 3.1: Implement Training Loop

**File**: `src/trainer.py`

```python
"""
Task-based training that trains on violation reduction.
NOT on matching ground truth examples!

Requirements:
1. Class RuleBasedTrainer
2. Train on fixing violations:
   
   for task in dataset:
       grid = task["grid"]
       
       # Deep supervision: 16 iterative improvements
       for step in range(16):
           violations = rule_graph.verify(grid)
           if not violations:
               break  # Solved!
           
           # Try to fix
           grid_new = model.refine_one_step(grid, violations)
           violations_new = rule_graph.verify(grid_new)
           
           # Loss: Did we reduce violations?
           loss = violation_reduction_loss(violations, violations_new)
           loss.backward()
           
           grid = grid_new.detach()

3. Loss function based on:
   - Number of violations (want 0)
   - Whether violations decreased
   - NOT: matching ground truth grid
"""

# Expected behavior:
trainer = RuleBasedTrainer(solver, dataset, rule_graph)
metrics = trainer.train(num_epochs=100)
assert metrics["final_loss"] < metrics["initial_loss"]
```

**Test Criteria**:

- Loss decreases over epochs
- Model learns to reduce violations
- Can solve some tasks completely (0 violations)
- Doesn't need ground truth grids

---

### Task 3.2: Implement Loss Functions

**File**: `src/trainer.py` (continued)

```python
"""
Loss functions for rule-based training.

Requirements:
1. violation_count_loss(violations) -> float
   - Simply count violations
   - Goal: 0 violations
   
2. violation_reduction_loss(old_violations, new_violations) -> float
   - Reward for reducing violations
   - Penalty if violations increased
   
3. confidence_loss(predicted_confidence, actually_correct) -> float
   - Teach model to know when it's right
   
NO cross-entropy with ground truth grids!
"""
```

**Test Criteria**:

- Loss is 0 when no violations
- Loss decreases when violations reduced
- Loss increases when violations added
- Confidence calibrates over training

---

## 📋 Phase 4: Hebbian Path Memory

### Task 4.1: Implement Path Memory (K_P)

**File**: `src/path_memory.py`

```python
"""
Hebbian path memory - learns which fixes work for which violations.

Requirements:
1. Class HebbianPathMemory
2. Store paths: (violation_pattern, action, result) -> weight
3. query(violations) returns candidate actions sorted by weight
4. update(path, success):
   - If success: w = w + α(1-w)  # LTP
   - If failure: w = w * γ        # LTD
   - If heavily used: w = w * β   # Myelination
5. ε-greedy selection: explore vs exploit

Path representation:
- violation_pattern: frozenset of violation types
- action: which cell to change, what to
- result: did violations decrease?
- weight: success rate (0-1)
- usage_count: how many times used
"""

# Expected behavior:
memory = HebbianPathMemory(alpha=0.1, gamma=0.95, beta=1.1)
memory.update(path, success=True)
assert memory.paths[path].weight > initial_weight  # LTP
```

**Test Criteria**:

- Successful paths strengthen (LTP)
- Failed paths weaken (LTD)
- Heavily-used paths get myelination boost
- ε-greedy balances exploration/exploitation
- Memory grows over training

---

### Task 4.2: Integrate Path Memory into Solver

**File**: `src/solver.py` (update)

```python
"""
Update RecursiveSolver to use path memory.

Changes:
1. Query path memory for candidate fixes
2. Pass candidates to network as additional input
3. Update path memory after each refinement step
4. Track which paths are being used

Network input becomes:
- x: task embedding
- y: current solution
- z: reasoning latent
- violations: what's wrong
- candidate_paths: what might fix it (from K_P)
"""

# Expected behavior:
solver = RecursiveSolver(network, rule_graph, path_memory)
result = solver.refine(grid)
# Path memory should grow and strengthen over many tasks
```

**Test Criteria**:

- Solver queries path memory
- Network uses candidate paths
- Memory updates after each step
- Learning accelerates with memory

---

## 📋 Phase 5: Testing & Validation

### Task 5.1: Integration Tests

**File**: `tests/test_integration.py`

```python
"""
End-to-end tests:
1. Generate task with violations
2. Solver fixes it using rules
3. Path memory learns patterns
4. Next similar task solves faster
"""

def test_learns_to_fix_row_duplicates():
    # Generate 10 tasks with row duplicates
    # Train on them
    # Check that later tasks solve faster
    pass

def test_path_memory_grows():
    # Train for 100 tasks
    # Check path memory has grown
    # Check weights have updated
    pass

def test_zero_shot_verification():
    # Rules work immediately (no training)
    assert rule_graph.verify(valid_grid) == []
    pass
```

---

### Task 5.2: Visualization

**File**: `experiments/visualize.py`

```python
"""
Visualize learning:
1. Show reasoning trace for a task
2. Show path memory growth over time
3. Show which paths are strongest
4. Show violation reduction curve
"""

def visualize_reasoning_trace(task, result):
    # Show step-by-step:
    # - Grid at each step
    # - Violations detected
    # - Action taken from path memory
    # - Success/failure
    pass

def plot_learning_curve(metrics_history):
    # Violations over time
    # Path memory growth
    # Success rate
    pass
```

---

## 📋 Phase 6: Training Experiments

### Task 6.1: Minimal Training Experiment

**File**: `experiments/train_minimal.py`

```python
"""
Train RI-TRM on 100 synthetic Sudoku tasks.

Expected results:
- Loss decreases
- Can fix some violations
- Path memory grows to ~50-100 paths
- Some tasks solve completely
"""

def main():
    rule_graph = SudokuRuleGraph()
    dataset = RuleBasedSudokuDataset(rule_graph, num_tasks=100)
    network = TinyRecursiveNetwork(...)
    path_memory = HebbianPathMemory()
    solver = RecursiveSolver(network, rule_graph, path_memory)
    trainer = RuleBasedTrainer(solver, dataset, rule_graph)
    
    metrics = trainer.train(num_epochs=100)
    
    # Save results
    torch.save(network.state_dict(), "model.pt")
    save_metrics(metrics)
```

**Success Criteria**:

- Trains without crashing
- Loss decreases over epochs
- Path memory accumulates knowledge
- Can solve at least some tasks

---

### Task 6.2: Scale to 1000 Tasks

**File**: `experiments/train_with_memory.py`

```python
"""
Scale up to 1000 tasks with full path memory.

Expected results (from RI-TRM paper):
- After 100 tasks: ~500 paths, avg weight 0.45
- After 500 tasks: ~2000 paths, avg weight 0.67
- After 1000 tasks: ~5000 paths, avg weight 0.78
- Strong paths (weight > 0.9) emerge for common patterns
"""
```

**Success Criteria**:

- Path memory statistics match paper predictions
- Interpretable paths emerge
- Solving gets faster over training
- Transfer: similar tasks solve immediately

---

## 📊 Success Metrics

Track these metrics to validate the approach:

1. **Rule-Based Learning**:
    
    - % tasks that reduce violations
    - % tasks solved completely (0 violations)
    - Average violations per step
2. **Path Memory Growth**:
    
    - Number of paths over time
    - Average path weight over time
    - Top 10 strongest paths
3. **Efficiency**:
    
    - Steps needed to solve
    - Training time vs examples
    - Parameters (should be ~7M)
4. **Interpretability**:
    
    - Can read reasoning traces
    - Can see which paths are used
    - Can explain why model made decision

---

## 🚀 Execution Order for Claude Code

Execute in this order:

```bash
# Phase 1: Rules
1. Implement src/rules.py
2. Test with tests/test_rules.py
3. Implement src/dataset.py
4. Test with tests/test_dataset.py

# Phase 2: Network
5. Implement src/network.py
6. Test with tests/test_network.py

# Phase 3: Solver & Training
7. Implement src/solver.py (without path memory)
8. Implement src/trainer.py
9. Test with tests/test_integration.py (basic)

# Phase 4: Path Memory
10. Implement src/path_memory.py
11. Test with tests/test_path_memory.py
12. Update src/solver.py to use path memory
13. Retest integration

# Phase 5: Experiments
14. Run experiments/train_minimal.py (100 tasks)
15. Implement experiments/visualize.py
16. Run experiments/train_with_memory.py (1000 tasks)
17. Generate final report
```

---

## 📝 Key Implementation Notes

1. **No Ground Truth Grids**: Training never sees complete Sudoku solutions. Only rules + broken grids.
    
2. **Loss is Rule-Based**: Loss = number of violations, NOT cross-entropy with target.
    
3. **Zero-Shot Verification**: Rules work immediately, no training needed for K_R.
    
4. **Path Memory is Interpretable**: Can see exactly what the model learned (e.g., "When row 3 has duplicate 5, try changing the rightmost cell").
    
5. **Small Scale**: Only 7M parameters, trains on 1000 tasks (not billions of tokens).
    

---

