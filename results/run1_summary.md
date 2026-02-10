# Run 1 Resource Summary

## Overview
- Iterations: `109,226`
- Total training time: `6,319.48 s` (`1h 45m 19.48s`)
- Carbon intensity used: `40.0 gCO2/kWh`

## Energy and Carbon
- Cumulative GPU energy: `0.250720 kWh` (`250.72 Wh`)
- Cumulative carbon estimate: `10.0288 gCO2e`
- Average step energy: `2.2954245e-06 kWh` (`0.002295 Wh`)
- Average step carbon: `9.181698e-05 gCO2e`

## Findings
- The run is strongly GPU-driven: average GPU utilization is `92.32%`, with a peak of `97%`. This indicates the workload is keeping the GPU busy most of the time.
- GPU memory pressure is low relative to capacity: about `4.69 GiB` used out of `32.76 GiB` total. There is substantial headroom to increase batch size or sequence length if needed.
- Average GPU power is `177.36 W`, peaking at `185.29 W`. This is consistent with sustained, high-throughput training rather than bursty usage.
- Total runtime is about `1h 45m`, and step time is stable on average (`54.53 ms`) with occasional long-tail spikes (peak `461.65 ms`), likely from periodic host-side overhead or data pipeline variability.
- Estimated footprint for this run is modest: `250.72 Wh` energy and `10.03 gCO2e` using the configured carbon intensity (`40 gCO2/kWh`).
- Process CPU usage averages `103.71%` (roughly one core fully utilized), while system-wide CPU is low (`0.80%`), suggesting CPU is not a system bottleneck overall.
- I/O is negligible (`0 MB` read and very small writes per step), so storage throughput does not appear to be limiting training in this run.

## Average Metrics
| Metric | Value |
|---|---:|
| GPU utilization | 92.3160 % |
| GPU memory used | 4693.2436 MiB |
| GPU memory total | 32760.0000 MiB |
| GPU power | 177.3623 W |
| GPU energy per step | 8263.5283 mJ |
| Torch CUDA allocated | 2545.8872 MiB |
| Torch CUDA reserved | 3903.9936 MiB |
| System memory used | 14025.6290 MiB |
| System memory total | 515475.1289 MiB |
| System CPU utilization | 0.8047 % |
| Process RSS | 2779.5567 MiB |
| Process VMS | 19225.9911 MiB |
| Process CPU utilization | 103.7141 % |
| Process I/O read | 0.0000 MB |
| Process I/O write | 0.000234 MB |
| Step duration | 54.5345 ms |

## Peak Metrics
| Metric | Value |
|---|---:|
| GPU utilization | 97.0000 % |
| GPU memory used | 4693.2500 MiB |
| GPU power | 185.2900 W |
| Torch CUDA allocated | 2545.8872 MiB |
| Torch CUDA reserved | 3904.0000 MiB |
| System memory used | 15122.8594 MiB |
| Process RSS | 2840.8477 MiB |
| Step duration | 461.6538 ms |

## Source
- Generated from `results/run1_summary.json`
