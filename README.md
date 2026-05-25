# Harmonic — Experiment Logs

Training and evaluation logs from H100 GPU runs for the Harmonic hierarchical SSM architecture.

## Logs

| File | Description |
|------|-------------|
| `crossover_main_6seq_h100.log` | Main crossover study: Harmonic vs Mamba vs Transformer at seq 1K–32K (enwiki8) |
| `crossover_stateful_warmeval_h100.log` | Crossover with stateful model and warm-start sequential eval |
| `crossover_wt103_h100.log` | Cross-dataset validation: same crossover protocol on WikiText-103 |
| `scaling_128_256_512_h100.log` | Scaling study: hidden=128, 256, 512 (~7M–112M params) at seq=1024 and 8192 |
| `scaling_hidden128_512_h100.log` | Scaling study: hidden=128 and 512 at seq=1024 and 8192 |
| `scaling_768_recovery_h100.log` | Scaling study: Transformer at hidden=768 seq=32768 (2-hour run, final bpt=6.482) |
| `ablation_h_mamba_flat_h100.log` | Ablation: Harmonic vs Mamba vs flat-timescale baseline (WikiText-2) |
| `training_curves_3way_h100.log` | Training curve comparison: Harmonic, Transformer, Mamba |
| `throughput_benchmark_h100.log` | Throughput benchmark (tokens/sec) at varying sequence lengths |
| `bench_h100_seq256_16384.log` | Detailed throughput benchmark seq=256–16384 |
| `transformer_baseline_30k_h100.log` | Transformer baseline: 30K steps, seq=1024 |
| `generate_120k_h100.log` | Long-run training: 120K steps, seq=1024 |
| `regen_rep_penalty_h100.log` | Text generation with repetition penalty |

## Key Results

**enwiki8 crossover (equal token budget, 28M params):**

| Seq len | Harmonic | Mamba | Transformer | H–TF gap |
|---------|----------|-------|-------------|----------|
| 1,024   | 6.571    | 6.616 | 6.662       | +1.4%    |
| 8,192   | 6.333    | 6.422 | 6.787       | +6.7%    |
| 32,768  | 6.433    | 6.549 | 7.259       | +11.4%   |
| 65,536  | 6.169    | OOM   | OOM         | —        |

**WikiText-103 crossover (same protocol):**

| Seq len | Harmonic | Mamba | Transformer | H–TF gap |
|---------|----------|-------|-------------|----------|
| 1,024   | 7.653    | 7.722 | 7.787       | +1.7%    |
| 8,192   | 7.888    | 7.988 | 8.292       | +4.9%    |
| 32,768  | 7.533    | 7.690 | 8.114       | +7.2%    |

**Scaling (~100M params, enwiki8):**

| Seq len | Harmonic | Transformer | Gap    |
|---------|----------|-------------|--------|
| 1,024   | 5.883    | 5.692       | −3.2%  |
| 8,192   | 5.712    | 6.089       | +6.6%  |
| 32,768  | 5.719    | 6.482       | +11.8% |

## Hardware

All runs on NVIDIA H100 80GB HBM3 via Modal cloud.

## Paper

See [harmonic-ai](https://github.com/Omibranch/harmonic-ai) for the main repository and paper.
