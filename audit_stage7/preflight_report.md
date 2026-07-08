# Stage 7 Preflight Report

- Python executable: `<local_python_executable>`
- Torch: `2.1.0+cu121` CUDA `12.1`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 4080 SUPER`
- GPU total/free memory: `16375` / `14259` MB
- bf16 supported: `True`
- bf16 matmul test: `{'ok': True, 'shape': [512, 512], 'dtype': 'torch.bfloat16'}`
- Disk free: `718.45` GB
- Tokenizer check: `True` vocab `2000`
- Checkpoint check: `True` params `1564992`

## Recommendation

- Model tier: `50M`
- Model config: `configs/stage7/model_50m.yaml`
- Context length: `256`
- Batch size / grad accum: `4` / `8`
- Notes: Enough free memory for a 40M-50M educational run with bf16 and gradient checkpointing.

## GPU Process Snapshot

```text
Wed Jul  8 13:26:08 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 591.86                 Driver Version: 591.86         CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4080 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
|  0%   42C    P0             60W /  320W |    1789MiB /  16376MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5528    C+G   ...8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A           12572    C+G   ...indows\System32\ShellHost.exe      N/A      |
|    0   N/A  N/A           12976    C+G   ...ntrolPanel\SystemSettings.exe      N/A      |
|    0   N/A  N/A           13032    C+G   ...t\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A           13272    C+G   <windows_process_path>               N/A      |
|    0   N/A  N/A           13712    C+G   ...2txyewy\CrossDeviceResume.exe      N/A      |
|    0   N/A  N/A           15868    C+G   ..._cw5n1h2txyewy\SearchHost.exe      N/A      |
|    0   N/A  N/A           15876    C+G   ...y\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A           18764    C+G   ...App_cw5n1h2txyewy\LockApp.exe      N/A      |
|    0   N/A  N/A           20496    C+G   ....0.4022.98\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           23956    C+G   ...5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A           24864    C+G   ...ouryDevice\asus_framework.exe      N/A      |
|    0   N/A  N/A           25900    C+G   ....0.4022.98\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           26720    C+G   ...al\Programs\Notion\Notion.exe      N/A      |
|    0   N/A  N/A           27244    C+G   ...er\Application\AVGBrowser.exe      N/A      |
|    0   N/A  N/A           31904    C+G   ....0.4022.98\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           33040    C+G   ...t\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A           34228    C+G   ...__2p2nqsd0c76g0\app\Codex.exe      N/A      |
|    0   N/A  N/A           35776    C+G   ...4__p7pnf6hceqser\snipaste.exe      N/A      |
|    0   N/A  N/A           36688    C+G   ....0.4022.98\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           36884    C+G   ...em32\ApplicationFrameHost.exe      N/A      |
|    0   N/A  N/A           36948    C+G   ...xyewy\ShellExperienceHost.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

## Risks

- OOM risk increases with 50M model, DPO reference model, GRPO online generation, and larger context length.
- Long runs may be interrupted; use checkpoint last.pt and --resume for pretrain.
- Public dataset download may fail; Stage 7 dataset script falls back to local synthetic corpus.
