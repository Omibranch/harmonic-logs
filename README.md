# Harmonic — Experiment Logs

Training and evaluation logs from H100 GPU runs for the Harmonic hierarchical SSM architecture.

## Logs

| File | Description |
|------|-------------|
| `crossover_main_6seq_h100.log` | Main crossover experiment: Harmonic vs Transformer at seq 1K–32K (6 points) |
| `crossover_stateful_warmeval_h100.log` | Crossover with stateful model and warm-start sequential eval |
| `scaling_hidden128_512_h100.log` | Scaling study: hidden=128 and 512 at seq=1024 and 8192 |
| `ablation_h_mamba_flat_h100.log` | Ablation: Harmonic vs Mamba-style vs flat-timescale baseline |
| `training_curves_3way_h100.log` | Training curve comparison: Harmonic, Transformer, Mamba |
| `throughput_benchmark_h100.log` | Throughput benchmark (tokens/sec) at varying sequence lengths |

## Key Results

- Harmonic outperforms Transformer by +12.7% in bits-per-token at seq=32768
- Advantage grows monotonically with context: +1.5% at 1K → +12.7% at 32K
- Scaling is consistent from ~7M to ~112M parameters at long context

## Hardware

All runs on NVIDIA H100 80GB HBM3 via Modal cloud.

## Paper

See [harmonic](https://github.com/Omibranch/harmonic) for the main repository and paper.
