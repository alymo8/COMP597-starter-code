# COMP597 Codebase Overview

## What This Project Does

This project is a **research framework for measuring the energy consumption and carbon emissions of training large language models (LLMs)**. Specifically, it trains Meta's OPT-350m model (and GPT-2) on synthetic data and measures how much energy and carbon different batch sizes produce. The goal is to understand the trade-offs between computational efficiency and environmental cost.

The framework is designed to run controlled experiments on SLURM-managed GPU clusters (McGill's HPC) and produce reproducible, comparable results.

---

## High-Level Architecture

```
CLI (launch.py)
    ├── Config System          ← hierarchical, auto-discovered config via dot-notation CLI args
    ├── Data Loader            ← synthetic dataset of random tokens (simulates real LLM training)
    ├── Model (OPT / GPT-2)   ← loaded from HuggingFace, trained with AdamW + linear schedule
    ├── SimpleTrainer          ← training loop: forward → backward → optimizer step
    └── TrainerStats           ← pluggable measurement layer (timing, energy, carbon, resources)
```

The framework uses an **auto-discovery plugin pattern**: new models, datasets, and stats collectors are registered automatically by scanning subdirectories — no core files need to be modified to add new components.

---

## Training Loop

The `SimpleTrainer` runs a fixed-duration training loop (not epoch-based, for fair comparisons):

1. **Warmup period**: a configurable number of steps/seconds at the start are skipped to exclude startup noise
2. **Measurement period**: stats collection is active
3. Each step:
   - `forward()` — runs model under `torch.cuda.amp.autocast` for FP16 compute with FP32 gradients; zeros gradients
   - `backward()` — `loss.backward()`, monitors for NaN/Inf gradients
   - `optimizer_step()` — updates weights via AdamW; monitors for NaN/Inf weights
   - Stats collector hooks wrap each phase for precise timing and resource sampling
4. Loop exits when `max_duration_sec` is reached

CUDA synchronization (`torch.cuda.synchronize()`) is called around each phase to ensure timing measurements reflect actual GPU work, not just CPU submission time.

---

## OPT Experiment Protocol

The main experimental workflow is a **batch-size sweep** to study how batch size affects energy efficiency:

### Step 1 — Run Experiments (`scripts/run_opt_experiments.py`)

For each batch size in a sweep (e.g., `[512, 256, 128]`), three profiling **profiles** are run:

| Profile | `trainer_stats` | Purpose |
|---------|-----------------|---------|
| A | `noop` | End-to-end wall-clock time baseline (zero overhead) |
| B | `codecarbon` | End-to-end energy baseline (CodeCarbon only, step tracking disabled) |
| C | `composite(simple + resource)` | Fine-grained per-step timing + resource metrics |

Each combination is repeated N times (default: 3) for statistical reliability.

**Batch size determination**: either specified with `--fixed-batch-size`, or auto-detected via a probing phase that doubles the batch until OOM, then binary searches for the maximum fitting size.

### Step 2 — Aggregate Results (`scripts/aggregate_opt_experiments.py`)

Reads all run outputs, validates data integrity (missing/failed runs), computes overhead compliance, and generates:
- `reports/integrity_report.json`
- `reports/overall_summary.json`
- Per-batch/profile timelines and phase bar charts

---

## How Energy and Carbon Are Measured

Two complementary approaches are used:

### 1. CodeCarbon (`src/trainer/stats/codecarbon.py`)

Uses the **[CodeCarbon](https://github.com/mlco2/codecarbon) library** (v3.1.0) which samples GPU power via NVML and integrates over time to compute energy.

Three `OfflineEmissionsTracker` instances are created per run:

| Tracker | Scope | Output file |
|---------|-------|-------------|
| Full tracker | Entire training session | `run_{N}_cc_full_rank_{gpu}.csv` |
| Step tracker (if `track_steps=1`) | Each training step as a task | `run_{N}_cc_step_rank_{gpu}.csv` |
| Substep tracker (if `track_substeps=1`) | Each phase (forward/backward/optimizer) | `run_{N}_cc_substep_rank_{gpu}.csv` |

**Key settings**:
- **Offline mode** — no external API calls during training
- **Location**: Canada/Quebec (determines grid carbon intensity)
- CPU tracking disabled in favor of GPU-only tracking for consistency
- Carbon estimated as: `energy_kwh × carbon_intensity_gco2_per_kwh`

### 2. NVML Direct Sampling (`src/trainer/stats/resource.py`)

Uses `nvidia-ml-py` to read GPU energy counters directly from hardware at each training step:

- `gpu_power_w`: instantaneous GPU power draw (Watts)
- `gpu_energy_mj`: energy consumed during the step (millijoules, from NVML counter)
- `step_energy_kwh`: per-step energy (kWh)
- `step_carbon_gco2`: per-step carbon (gCO2e), using configurable `carbon_intensity_gco2_per_kwh` (default: 40 gCO2/kWh)
- `cumulative_gpu_energy_kwh` / `cumulative_carbon_gco2`: running totals

This gives **step-level granularity** without depending on CodeCarbon's sampling interval.

---

## Metrics Collected

### Timing (Simple Stats — Profile C)

| Metric | Unit | Description |
|--------|------|-------------|
| `step_ms` | ms | Total step duration |
| `forward_ms` | ms | Forward pass duration |
| `backward_ms` | ms | Backward pass duration |
| `optimizer_step_ms` | ms | Optimizer update duration |
| `checkpoint_ms` | ms | Checkpoint save duration |
| `loss` | float | Training loss value |

### GPU / System Resources (Resource Stats — Profile C)

| Metric | Unit | Description |
|--------|------|-------------|
| `gpu_util` | % | GPU utilization |
| `gpu_mem_used` / `gpu_mem_total` | MiB | GPU memory usage |
| `gpu_power_w` | W | GPU power draw |
| `gpu_energy_mj` | mJ | Step GPU energy |
| `step_energy_kwh` | kWh | Per-step energy |
| `step_carbon_gco2` | gCO2e | Per-step carbon emissions |
| `cumulative_gpu_energy_kwh` | kWh | Total GPU energy |
| `cumulative_carbon_gco2` | gCO2e | Total carbon emitted |
| `sys_mem_used` / `sys_mem_total` | MiB | System RAM usage |
| `sys_cpu_percent` | % | System CPU utilization |
| `proc_rss` / `proc_vms` | MiB | Process memory (RSS/VMS) |
| `proc_cpu_percent` | % | Process CPU utilization |
| `io_read` / `io_write` | bytes | Disk I/O |
| `torch_cuda_allocated_mib` | MiB | PyTorch-allocated GPU memory |
| `torch_cuda_reserved_mib` | MiB | PyTorch-reserved GPU memory |

### Energy / Carbon (CodeCarbon — Profiles B and C)

| Metric | Description |
|--------|-------------|
| `energy_consumed` | kWh for the tracked task/session |
| `emissions` | gCO2e for the tracked task/session |
| `cpu_power`, `gpu_power`, `ram_power` | Power breakdown by component |

---

## Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.9.1 | Model training, GPU operations, autocast |
| `transformers` | 4.57.1 | OPT and GPT-2 model loading (HuggingFace) |
| `datasets` | 4.4.1 | Dataset utilities |
| `codecarbon` | 3.1.0 | Energy/carbon tracking via GPU power sampling |
| `nvidia-ml-py` | 13.580.82 | Direct NVML GPU hardware queries (power, memory, utilization) |
| `psutil` | 7.1.3 | System and process resource monitoring (CPU, RAM, I/O) |
| `pandas` | 2.3.3 | Tabular data manipulation for results |
| `numpy` | 2.3.5 | Numerical operations |
| `tqdm` | 4.67.1 | Training progress bars |

---

## Data

**Synthetic data** is used (`src/data/opt/`) rather than real text corpora. Each sample is a random sequence of token IDs, with `input_ids`, `attention_mask`, and `labels` tensors. This eliminates data loading as a confound when measuring compute-side energy.

- Default sequence length: 1024 tokens
- Dataset size: ~2.5 GiB worth of synthetic sequences
- Bytes per token: 24 (three int64 tensors)

---

## Configuration System

All parameters are controlled via dot-notation CLI arguments that map to hierarchical config classes. Examples:

```bash
--data_configs.opt.batch_size 256          # batch size
--data_configs.opt.seq_len 1024            # sequence length
--trainer_configs.simple.max_duration_sec 300  # training duration
--trainer_stats_configs.codecarbon.track_steps 0  # disable step tracking
--model_configs.opt.hf_name facebook/opt-350m    # which OPT model
```

Sub-configs are auto-discovered from `src/config/{models,data,trainers,trainer_stats}/`.

---

## Output Structure

Each experiment run produces:

```
results/e2e_experiments/run-N/opt-run-YYYYMMDD-HHMMSS/
├── manifest.json                     # full experiment metadata, git commit, GPU info
├── batch_512/
│   ├── profile_A/repeat_1/           # noop (timing baseline)
│   │   ├── stdout.log
│   │   └── run_timing.json
│   ├── profile_B/repeat_1/           # CodeCarbon energy baseline
│   │   ├── run_*_cc_full_rank_0.csv
│   │   └── run_timing.json
│   └── profile_C/repeat_1/           # fine-grained metrics
│       ├── simple_steps.csv          # per-step timing
│       ├── simple_summary.json
│       ├── resource_steps.csv        # per-step resource + energy
│       ├── resource_summary.json
│       └── plots/
└── reports/
    ├── integrity_report.json
    ├── overall_summary.json
    └── plots/
```

---

## Reproducibility

- Fixed random seeds (`--seed`, default 1234)
- Git commit hash logged in `manifest.json`
- Deterministic run numbering via `--run_num`
- Warmup period excludes startup transients
- Fixed-duration runs (not epoch-based) for fair cross-configuration comparison
- Multiple repeats for variance estimation
