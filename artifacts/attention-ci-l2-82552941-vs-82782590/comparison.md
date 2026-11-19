# L2 JAX cuDNN attention CI comparison

The five job pairs with complete attention results ran and passed **4,799** cases on main and **5,038** on the feature branch. Both recorded **zero attention failures or errors**. Every shared test ID kept the same outcome. The feature branch reported 323 additional cases, of which 239 passed and 84 were skipped. [Main pipeline 82552941](https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/pipelines/82552941) · [Feature pipeline 82782590](https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/pipelines/82782590) · [Module comparison](module_comparison.csv)

## Job results

The unit jobs launch the standard fused-attention, score-modifier, and Flax layer modules, plus the new MXFP8 attention module on the feature branch. The distributed jobs launch the distributed fused-attention module. Counts include only these modules. Every recorded case that executed passed. [Job comparison](job_comparison.csv)

| L2 JAX job | Main run and passed | Main skipped | Feature run and passed | Feature skipped | Main GitLab time | Feature GitLab time |
|---|---:|---:|---:|---:|---:|---:|
| Unit · A100 1 GPU | 1,844 | 3,555 | 1,893 (partial) | 3,573 (partial) | 2:39:58 | 2:01:08 |
| Unit · H100 1 GPU | 2,305 | 3,094 | 2,367 | 3,153 | 2:31:04 | 2:35:34 |
| Unit · B200 1 GPU | 2,041 | 3,490 | 2,162 | 3,490 | 3:22:02 | 3:28:41 |
| Unit · GB200 NVL 4 GPUs | 1,231 (partial) | 1,318 (partial) | 1,012 (partial) | 1,092 (partial) | 3:00:04 | 3:00:04 |
| Distributed · H100 4 GPUs | 123 | 355 | 125 | 380 | 0:34:27 | 0:34:48 |
| Distributed · B200 8 GPUs | 123 | 355 | 150 | 355 | 0:49:10 | 0:57:59 |
| Distributed · GB200 4 GPUs | 207 | 246 | 234 | 246 | 5:37:08 | 5:42:15 |
| **Five complete pairs** | **4,799** | **7,540** | **5,038** | **7,624** | **12:53:51** | **13:19:18** |

The feature A100 retry reached three attention modules before timing out in `test_mhc.py`. The MXFP8 module was still pending. Both GB200 NVL jobs timed out during `test_fused_attn.py` after spending more than two hours waiting for GPUs. Their counts come from completed cases visible in the traces. The completed H100 and B200 unit jobs each recorded two failures in `test_mhc.py` on both branches. [Job attempts and sources](jobs.csv) · [Per-module counts](modules.csv)

All seven L2 JAX job configurations that started pytest on main also started it on the feature branch. The three main-only configurations were manual and never started. They were the R100 single-GPU unit job and the VR200 NVL four-GPU unit and distributed jobs. [Job comparison](job_comparison.csv)

## Results by module across the five complete pairs

| Module | Main run and passed | Main skipped | Feature run and passed | Feature skipped |
|---|---:|---:|---:|---:|
| `test_fused_attn.py` | 3,662 | 6,584 | 3,738 | 6,596 |
| `test_fused_attn_score_mod.py` | 24 | 0 | 70 | 0 |
| `test_layer.py` | 660 | 0 | 660 | 0 |
| `test_mxfp8_attention.py` | 0 | 0 | 61 | 47 |
| `test_distributed_fused_attn.py` | 453 | 956 | 509 | 981 |

## Total job runtime

| Jobs and timing measure | Main | Feature | Feature minus main |
|---|---:|---:|---:|
| Five complete pairs, GitLab duration | 12:53:51 | 13:19:18 | +0:25:27 (+3.3%) |
| Five complete pairs, after GPU allocation waits | 7:45:13 | 8:01:59 | +0:16:46 (+3.6%) |
| All seven pairs, GitLab duration | 18:33:53 | 18:20:30 | −0:13:23 (−1.2%) |
| All seven pairs, after GPU allocation waits | 11:14:44 | 10:49:15 | −0:25:29 (−3.8%) |

The all-job totals include the interrupted attempts shown above. Times sum full job durations, including work outside attention, and use the latest attempt per configuration. The original feature B200 unit attempt had identical attention outcomes. Using its runtime instead of the retry, the five complete feature jobs took 3m 19s longer than main by GitLab duration and 3m 35s longer after GPU allocation waits. [Original B200 attempt](https://gitlab-master.nvidia.com/dl/transformerengine/transformerengine/-/jobs/553688750) · [Timing definitions](README.txt)
