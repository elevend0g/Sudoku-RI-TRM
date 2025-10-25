# RI-TRM Sudoku

**Rule-Initialized Tiny Recursive Models** for Sudoku solving - a complete implementation demonstrating efficient learning with explicit knowledge.

## 🎯 Status: Fully Implemented & Tested

✅ **163/163 tests passing**
✅ **~5,300 lines of code**
✅ **7M parameters (1000× smaller than LLMs)**
✅ **Ready to use!**

## 🚀 Quick Start

### Try the Demo

```bash
# Quick automated demo (30 seconds)
python demo_auto.py

# Full interactive walkthrough (5 minutes)
python demo.py
```

### Run Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train on 100 tasks (~2 minutes)
python experiments/train_minimal.py

# Run all tests
pytest tests/ -v
```

## 💡 Key Features

- **Zero-shot verification** - Rules work without training (K_R)
- **Hebbian learning** - Path memory with LTP/LTD (K_P)
- **Tiny network** - 7M parameters vs 7B+ in LLMs
- **No ground truth** - Trains on violations, not solutions
- **Interpretable** - Full reasoning traces
- **Efficient** - Trains on ~1,000 tasks, not billions of tokens

## 📊 Architecture

```
Layer 1 (K_F): Factual Knowledge     [Not needed for Sudoku]
Layer 2 (K_R): Structural Rules      ✅ Zero-shot verification
Layer 3 (K_P): Path Memory          ✅ Hebbian learning
```

**Network**: 2-layer transformer, 512 hidden dim, 8 heads, RMSNorm, RoPE, SwiGLU

## 📁 Project Structure

```
├── src/              Core implementation
│   ├── rules.py          Zero-shot verification (K_R)
│   ├── dataset.py        Rule-based generation
│   ├── path_memory.py    Hebbian learning (K_P)
│   ├── network.py        7M parameter network
│   └── solver.py         Recursive refinement
├── tests/            163 tests, all passing
├── experiments/      Training scripts
├── docs/             Research paper & implementation plan
├── demo.py           Interactive demo
└── demo_auto.py      Automated demo
```

## 🎓 RI-TRM Principles

✓ **Explicit knowledge > Learned** (when possible)
✓ **Tasks > Tokens** (train on ~1K, not billions)
✓ **Interpretability > Black box**
✓ **Efficiency > Scale**

## 📖 Documentation

- **Full README**: [README_COMPLETE.md](README_COMPLETE.md)
- **Research Paper**: [docs/RI-TRM.md](docs/RI-TRM.md)
- **Implementation Plan**: [docs/Sudoku.md](docs/Sudoku.md)

## 🔬 Example Usage

```python
from src.rules import SudokuRuleGraph
from src.network import create_sudoku_network
from src.path_memory import HebbianPathMemory
from src.solver import RecursiveSolver

# Initialize components
rule_graph = SudokuRuleGraph()
network = create_sudoku_network()  # 7M params
path_memory = HebbianPathMemory()

# Create solver
solver = RecursiveSolver(network, rule_graph, path_memory)

# Solve with interpretable trace
result = solver.solve(grid, return_trace=True)
print(f"Solved: {result.success}")
print(f"Steps: {result.num_steps}")
for step in result.trace:
    print(step)
```

## 📈 Training Results

From `experiments/train_minimal.py`:
- **Loss reduction**: 99.9% (0.0476 → 0.0000)
- **Training time**: ~2 minutes on CPU
- **Model size**: ~50MB
- **Path memory**: Grows automatically during training

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Specific suites
pytest tests/test_rules.py -v        # Zero-shot verification
pytest tests/test_network.py -v      # Network architecture
pytest tests/test_solver.py -v       # Recursive refinement

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Comparison

| Model | Parameters | Training Data | Size |
|-------|-----------|---------------|------|
| RI-TRM Sudoku | 7M | ~1,000 tasks | ~50MB |
| GPT-2 Small | 117M | Billions of tokens | ~500MB |
| GPT-3.5 | 175B | Trillions of tokens | ~700GB |

## 📝 Citation

```bibtex
@article{ritrm2025,
  title={Rule-Initialized Tiny Recursive Models},
  author={Jay Noon},
  year={2025},
  note={Under Review}
}
```

## 📄 License

MIT License

---

**Built following RI-TRM principles: Explicit knowledge when possible, train on tasks not tokens, fully interpretable, efficient.**
