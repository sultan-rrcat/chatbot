# Computer Division — Lab Notes

---

## Table of Contents

1. [GPT-2 Setup](#1-gpt-2-setup)
2. [CUDA Support for PyTorch](#2-cuda-support-for-pytorch)
3. [NLTK WordNet Issue Fix](#3-nltk-wordnet-issue-fix)
4. [Quantized vs Non-Quantized Models](#4-quantized-vs-non-quantized-models)
5. [Accelerator Physics Q&A Dataset Samples](#5-accelerator-physics-qa-dataset-samples)
6. [LM Evaluation Harness](#6-lm-evaluation-harness)
7. [MMLU Task List](#7-mmlu-task-list)
8. [Memory-Constrained Inference Strategies](#8-memory-constrained-inference-strategies)
9. [Linux (RHEL) Environment Setup](#9-linux-rhel-environment-setup)
10. [Model Downloads (HuggingFace)](#10-model-downloads-huggingface)
11. [RAG Application](#11-rag-application)
12. [Accelerator Control Chatbot](#12-accelerator-control-chatbot)
13. [Virtual Environment Issues & Fixes](#13-virtual-environment-issues--fixes)
14. [Git Reference](#14-git-reference)

---

## 1. GPT-2 Setup

```bash
pip install transformers
git clone "gpt2"
cd gpt2
pip install -r requirement.txt
python download_model.py 124M
```

---

## 2. CUDA Support for PyTorch

```bash
# Option 1 — explicit version pinning
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Option 2 — most suitable command
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

---

## 3. NLTK WordNet Issue Fix

**Issue:** `wordnet` corpus was missing, causing a kernel error.

**Fix:**
1. Downloaded the WordNet zip from Kaggle.
2. Extracted it to:
   ```
   C:\Users\trainee\nltk_data\corpora\
   ```
3. The kernel auto-detects it from that path.

---

## 4. Quantized vs Non-Quantized Models

| Type | Details |
|---|---|
| **Quantized (4-bit)** | Enables running larger models on low-end hardware, but sacrifices speed and quality. |
| **Non-quantized (FP16)** | Best for quality and speed if you have enough VRAM. |

---

## 5. Accelerator Physics Q&A Dataset Samples

Sample prompt–reference pairs used for evaluation:

**Quadrupole Magnet:**
> *Prompt:* Describe the function of a quadrupole magnet in a particle accelerator.
>
> *Reference:* A quadrupole magnet is used in particle accelerators to focus charged particle beams. It consists of four magnetic poles arranged to create a magnetic field gradient. This gradient focuses the beam in one plane while defocusing it in the perpendicular plane. By alternating focusing and defocusing quadrupoles along the accelerator, the beam is kept tightly confined, preventing it from spreading and maintaining its intensity over long distances.

**Linear vs Circular Accelerators:**
> *Prompt:* What are the advantages of linear accelerators over circular accelerators?
>
> *Reference:* Linear accelerators (linacs) accelerate particles in a straight line, avoiding synchrotron radiation losses that occur in circular accelerators at high energies. This makes them more efficient for accelerating light particles like electrons to high energies. Linacs also allow higher beam quality and avoid issues with magnetic bending. However, they require more space for equivalent energy gains compared to circular designs, which reuse the same acceleration path.

**Beam Emittance:**
> *Prompt:* Explain the concept of beam emittance in accelerator physics.
>
> *Reference:* Beam emittance is a measure of the spread of particle positions and momenta in a particle beam, representing its phase space area. Low emittance indicates a beam that is tightly focused and has low divergence, which is desirable for achieving high luminosity and precision in experiments. Emittance is conserved under linear transformations but can grow due to scattering, space charge effects, or nonlinearities. Controlling emittance is crucial for maintaining beam quality in accelerators.

### FermiLab PDF — Q&A

**Q1.** What are the three major facilities at FermiLab examined for demonstrating the practical power of the perturbative method?

> **A:** The Integrable Optics Test Accelerator (IOTA), the Main Injector (MI), and the Mu2e Delivery Ring.

**Q2.** What does the paper state the Courant-Snyder invariant allows for by propagating the Twiss parameters along the reference orbit?

> **A:** It allows the reconstruction of phase space trajectories not only at each discrete turn, but also continuously along the machine's circumference, *s*.

---

## 6. LM Evaluation Harness

### Installation

```bash
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
```

### Evaluation Commands

**Windows — GPT-2 XL on MMLU (with local dataset path):**
```bash
lm-eval --model hf \
    --model_args pretrained="C:/Users/trainee/Desktop/sultan/On Job Training/ML/models/gpt2xl" \
    --tasks mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy \
    --num_fewshot 0 \
    --output_path results.json \
    --device cuda:0 \
    --batch_size 8 \
    --include_path "C:/Users/trainee/Desktop/sultan/On Job Training/ML/data/mmlu"
```

**Windows — GPT-2 on GSM8K:**
```bash
lm-eval --model hf \
    --model_args pretrained="C:/Users/trainee/Desktop/sultan/On Job Training/ML/models/gpt2" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --output_path results.json \
    --device cuda:0 \
    --batch_size 8 \
    --limit 100
```

**Linux — with model offloading:**
```bash
lm-eval --model hf \
    --model_args "pretrained=/path/to/your/model,device_map='auto'" \
    --tasks mmlu_* \
    --num_fewshot 5 \
    --batch_size <your_batch_size> \
    --device cuda:0 \
    --output_path results_offloaded.json
```

### Why `hellaswag` Didn't Work

> `hellaswag` tries to download the dataset from `https://raw.githubusercontent.com/rowanz/hellaswag/master/data/` which is blocked on the internal network.

---

## 7. MMLU Task List

### Raw Subject Names (no prefix)

```
abstract_algebra, anatomy, astronomy, auxiliary_train, business_ethics,
clinical_knowledge, college_biology, college_chemistry, college_computer_science,
college_mathematics, college_medicine, college_physics, computer_security,
conceptual_physics, econometrics, electrical_engineering, elementary_mathematics,
formal_logic, global_facts, high_school_biology, high_school_chemistry,
high_school_computer_science, high_school_european_history, high_school_geography,
high_school_government_and_politics, high_school_macroeconomics,
high_school_mathematics, high_school_microeconomics, high_school_physics,
high_school_psychology, human_aging, human_sexuality, international_law,
jurisprudence, logical_fallacies, machine_learning, management, marketing,
medical_genetics, miscellaneous, moral_disputes, moral_scenarios, nutrition,
philosophy, prehistory, professional_accounting, professional_law,
professional_medicine, professional_psychology, public_relations,
security_studies, sociology, us_foreign_policy, virology, world_religions
```

### With `mmlu_` Prefix (for `lm-eval`)

```
mmlu_abstract_algebra, mmlu_anatomy, mmlu_astronomy, mmlu_business_ethics,
mmlu_clinical_knowledge, mmlu_college_biology, mmlu_college_chemistry,
mmlu_college_computer_science, mmlu_college_mathematics, mmlu_college_medicine,
mmlu_college_physics, mmlu_computer_security, mmlu_conceptual_physics,
mmlu_econometrics, mmlu_electrical_engineering, mmlu_elementary_mathematics,
mmlu_formal_logic, mmlu_global_facts, mmlu_high_school_biology,
mmlu_high_school_chemistry, mmlu_high_school_computer_science,
mmlu_high_school_european_history, mmlu_high_school_geography,
mmlu_high_school_government_and_politics, mmlu_high_school_macroeconomics,
mmlu_high_school_mathematics, mmlu_high_school_microeconomics,
mmlu_high_school_physics, mmlu_high_school_psychology, mmlu_human_aging,
mmlu_human_sexuality, mmlu_international_law, mmlu_jurisprudence,
mmlu_logical_fallacies, mmlu_machine_learning, mmlu_management, mmlu_marketing,
mmlu_medical_genetics, mmlu_miscellaneous, mmlu_moral_disputes,
mmlu_moral_scenarios, mmlu_nutrition, mmlu_philosophy, mmlu_prehistory,
mmlu_professional_accounting, mmlu_professional_law, mmlu_professional_medicine,
mmlu_professional_psychology, mmlu_public_relations, mmlu_security_studies,
mmlu_sociology, mmlu_us_foreign_policy, mmlu_virology, mmlu_world_religions
```

> **Note:** Some group names also exist, e.g. `mmlu_continuation_social_sciences`.

---

## 8. Memory-Constrained Inference Strategies

### Techniques

1. Use CPU or `device_map=auto`
2. Use FP16 or BF16: `torch_dtype=torch.float16` or `torch_dtype=torch.bfloat16`
3. Load in 4-bit or 8-bit: `bnb_config = BitsAndBytesConfig(load_in_8bit=True)`

### Comparison Table

| Strategy | VRAM Saved | Speed | Accuracy Impact | Best Use Case |
|---|---|---|---|---|
| CPU inference | ✅ All | 🐢 Slow | ❌ None | Debugging, testing |
| FP16/BF16 | ~50% | 🚀 Fast | ✅ Same | Modern GPU, low memory |
| Quantization (8/4-bit) | ~70–85% | ⚡ Fast | 🔸 Slight drop | Production inference |
| `device_map=auto` | ✅ Flexible | ⚠️ Depends | ✅ Same | Shared CPU+GPU deployment |
| DeepSpeed/Zero-3 | 🔥 Huge | ✅ Fast | ✅ Same | Training, very large models |

> **Note:** The best way to run DeepSpeed on Windows is via WSL 2. Native Windows support is limited and many features may not work. Consider using a Linux machine or cloud-based GPU instances for full functionality.

---

## 9. Linux (RHEL) Environment Setup

### Activate Conda Environment

```bash
source /home2/testdev/miniconda3/etc/profile.d/conda.sh
conda activate gpt_env_311
```

### Set Proxy

```bash
export http_proxy="http://cdtrainee:Indore%40234%23@10.31.31.10:5128"
export https_proxy="http://cdtrainee:Indore%40234%23@10.31.31.10:5128"
```

### Uninstall All Packages

```bash
pip freeze | xargs pip uninstall -y
```

### Install PyTorch (CUDA 11.3)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

> The CUDA 11.8 build of PyTorch is compatible with a CUDA 11.2 toolkit and drivers as they are in the same major CUDA 11 release family.

### Jupyter Notebook (Remote)

```bash
# Start notebook on GPU server
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 \
    --NotebookApp.token='sultan123' --NotebookApp.password=''

# SSH tunnel (from local machine)
ssh -L 8888:localhost:8888 -J testdev@10.10.30.85 testdev@kshitij-5-gpu4

# Access in browser
# https://localhost:8888/tree
```

### RHEL Evaluation Commands

```bash
# GPT-2 on GSM8K
lm-eval --model hf \
    --model_args pretrained="/home2/testdev/sultan/llm/models/gpt2" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --output_path "results/results.json" \
    --device cuda:0

# GPT-2 XL on MMLU
lm-eval --model hf \
    --model_args pretrained="/home2/testdev/sultan/llm/models/gpt2xl" \
    --tasks mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy \
    --num_fewshot 5 \
    --device cuda:0 \
    --batch_size 16 \
    --include_path "/home2/testdev/sultan/llm/data/mmlu/"
```

### HuggingFace Login

```bash
huggingface-cli login
# Token: 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
```

### llama.cpp (GGUF Models)

```bash
# Build with compatible GCC
conda install -y -c conda-forge gcc_linux-64 gxx_linux-64

# Fix GCC version error with CUDA 11.2
conda install -c conda-forge gcc_linux-64=9.* gxx_linux-64=9.*

# Set compiler before building
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++

git submodule update --init --recursive
pip install -e . -v

# Run inference
./main -m ../models/Mistral7BITQ/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
    -p "What is quantum entanglement?" -n 128

./llama-server -m /home2/testdev/sultan/llm/models/Mistral7BITQ/mistral-7b-instruct-v0.2.Q3_K_M.gguf \
    -p "Explain Quantum Theory" -n 128
```

> **GCC Version Error:** `RuntimeError: g++ (11.2.0) is greater than the maximum required version by CUDA 11.2. Please use g++ >=5.0.0, <11.0.`

### Installing `xformers` Successfully

```bash
# Update CUDA toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Verify CUDA version
nvcc -V

# Reinstall PyTorch for cu118
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install unsloth
```

### Installing `libaio-devel` (System-level dependency)

```bash
# Required when bitsandbytes or Triton compiles CUDA kernels
sudo yum install libaio-devel
# OR for newer RHEL
sudo dnf install libaio-devel
```

> **Contact:** Prabhat — 8450

### Port Forwarding (Flask App)

```bash
ssh -L 5000:localhost:5000 -J testdev@10.10.30.85 testdev@kshitij-5-gpu4
# http://localhost:5000
```

### Miscellaneous

```bash
# Run script in background
nohup python3 llm_comparison.py > out.log 2>&1 &

# Zip app excluding models folder
zip -r rag_app.zip rag_flask_app -x "rag_flask_app/models/*"

# Disable ChromaDB telemetry
export CHROMA_TELEMETRY_SETTINGS='{"anonymized_telemetry": false}'
```

---

## 10. Model Downloads (HuggingFace)

### Model Paths on Server

```python
{
    "DeepSeek-Qwen-7B":  "/home2/testdev/sultan/llm/models/DeepSeek-Qwen-7B/",
    "GPT2":              "/home2/testdev/sultan/llm/models/GPT2/",
    "GPT2-M":            "/home2/testdev/sultan/llm/models/GPT2-M/",
    "GPT2-XL":           "/home2/testdev/sultan/llm/models/GPT2-XL/",
    "Gemma-2-27B-IT":    "/home2/testdev/sultan/llm/models/Gemma-2-27B-IT/",
    "Gemma-2-2B-IT":     "/home2/testdev/sultan/llm/models/Gemma-2-2B-IT/",
    "Gemma-2-7B-IT":     "/home2/testdev/sultan/llm/models/Gemma-2-7B-IT/",
}
```

### CLI Download Commands

```bash
# Mistral
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 --local-dir models/Mixtral8x7B-IT

# Gemma
huggingface-cli download google/gemma-7b-it              --local-dir models/Gemma-7B-IT
huggingface-cli download google/gemma-3-4b-it            --local-dir models/Gemma-3-4B-IT
huggingface-cli download google/gemma-3-4b-pt            --local-dir models/Gemma-3-4B
huggingface-cli download google/gemma-3-12b-pt           --local-dir models/Gemma-3-12B
huggingface-cli download google/gemma-3-12b-it           --local-dir models/Gemma-3-12B-IT
huggingface-cli download google/gemma-3-27b-it           --local-dir models/Gemma-3-27B-IT

# Llama
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir models/Llama31-70B-IT
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct  --local-dir models/Llama31-8B-IT

# Embedders & Domain Models
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2
huggingface-cli download thellert/accphysbert_uncased           --local-dir models/accphysbert_uncased
huggingface-cli download nomic-ai/nomic-embed-text-v1           --local-dir models/nomic-embed-text-v1
```

### Wget Download (Authenticated)

```bash
wget --header="Authorization: Bearer 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" \
     https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

---

## 11. RAG Application

### Features

1. Domain-specific Q&A via RAG (Accelerator Physics PDFs)
2. DeepSeek Qwen 7B IT model (float16)
3. SSE (Server-Sent Events) for real-time streaming responses
4. ChromaDB as vector storage
5. `all-MiniLM-L6-v2` as sentence embedder
6. `marked.js` for formatting tables and code blocks
7. Flask backend
8. Light/dark theme toggle
9. No external internet-level dependencies (CDN/APIs) — works fine on internal networks

### Resource Usage

```
top:
29674 testdev  20  0  54.8g  2.5g  253220 S  100.3  0.7  16:14.84 python

nvidia-smi:
GPU 0: 11037 MiB
GPU 1: 12727 MiB
```

### Current Limitations

1. Gradual increase in response time due to chat history (backend deque limited to last 5 turns — may lose context in long conversations)
2. Mathematical expression formatting not yet configured
3. No persistent saved chats (needs Postgres or Redis)
4. No user management or authentication
5. No external API integration beyond RAG

### Future Scope

1. Fine-tune with domain-specific data
2. Use 70B models with quantization
3. Explore MoE models or multi-model classification setups
4. Automate RAG pipeline / Dynamic Document Ingestion from client side
5. Maintain persistent chat history
6. Mode toggle (Accelerator vs Laser physics)
7. Text-to-image model integration
8. Explore domain-tailored sentence embedders for accelerator physics
9. Advanced UI/UX: copy-to-clipboard for code blocks, download options for responses, loading animations, searchable chat history
10. Explore other inference techniques (vLLM, DeepSpeed) for serving large user bases

---

## 12. Accelerator Control Chatbot

### Planned Architecture

A hybrid chatbot for the accelerator control system combining:

- **SQL querying** — for real-time and historical data from MS SQL Server
- **RAG** — for contextually rich responses using fault-related tables and documentation

**Example queries it should handle:**
- "What is the current beam energy?" *(real-time)*
- "What is the average active hour per day of the accelerator system?" *(analytical)*
- "What was the last issue in RF Cavity 1?" *(fault-related)*
- "What is synchrotron radiation?" *(general)*

### Intent Classifier

To route queries properly, an intent classifier should be built first to distinguish between SQL queries, RAG lookups, and general responses.

### SQL Server Setup

```
Server=localhost\SQLEXPRESS;Database=master;Trusted_Connection=True;
```

SQL Server install log path: `C:\Program Files\Microsoft SQL Server\160\Setup Bootstrap\Log\20250616_110107`

### Fault Book Table Schema

**Table:** `fault_bookv3`

| Column | Type | Description |
|---|---|---|
| `fault_id` | BIGINT (NOT NULL) | Unique fault ID |
| `fault_time` | DATETIME (NOT NULL) | Timestamp of fault |
| `fault_duration` | VARCHAR(25) (NOT NULL) | Duration of fault |
| `system_name` | VARCHAR(50) | High-level system (e.g. "Control System Infrastructure", "Indus-2 RF System") |
| `device_name` | VARCHAR(50) | Specific device (e.g. "Software", "Beam-filling", "Power Supply") |
| `fault_description` | VARCHAR(2000) | Detailed fault description |
| `persons_involved` | VARCHAR(200) | Names of people involved in resolution |
| `action_taken` | VARCHAR(500) | Steps taken to resolve the fault |
| `faulty_system` | VARCHAR(200) | Granular device name (mostly NULL) |
| `human_error` | VARCHAR(200) | Yes / No / Nil |
| `document` | IMAGE | Image of problem area |
| `document_name` | VARCHAR(50) | Formal document name if created |
| `fda_entry` | VARCHAR(3) (NOT NULL) | Yes / No |
| `logged_by` | VARCHAR(25) | Person who logged the data |
| `log_time` | DATETIME | Time of data logging |
| `first_observation` | VARCHAR(4000) (NOT NULL) | Description of how the failure was first noticed |
| `beam_affected` | VARCHAR(3) (NOT NULL) | Yes / No |

### Ollama Commands

```bash
ollama serve
ollama pull phi3
ollama run phi3
ollama list
ollama show phi3
```

### Windows Proxy (PowerShell)

```powershell
$env:HTTP_PROXY  = "http://cdtrainee:Indore%40234%23@10.31.31.10:5128"
$env:HTTPS_PROXY = "http://cdtrainee:Indore%40234%23@10.31.31.10:5128"
```

> **Note:** `ping` uses ICMP, which is often blocked by corporate firewalls. Git, curl, and web browsers use HTTP/HTTPS, which are typically allowed through the proxy.

### Detecting Python 2.7 Issue

**Issue:** System was always defaulting to Python 2.7.

**Fix:** Renamed the `python2.7` executable from `python` to `python2`.

### MobaXterm Master Password

```
1234567
```

---

## 13. Virtual Environment Issues & Fixes

**Issue:** After activating Python 3.12 virtual environment (`env_312`), `pip list` shows global packages or an empty list, even though packages exist in `env_312\Lib\site-packages`.

### Root Causes

1. `python` and `pip` point to Miniconda's interpreter instead of the virtual environment's.
2. Conda is in system `PATH`, overriding venv behaviour even after activation.
3. Virtual environment was created while Conda was active (wrong base interpreter).

### Immediate Fixes

```bash
# Use the venv's python directly
env_312\Scripts\python.exe -m pip list
env_312\Scripts\python.exe -m pip install <package>
```

Avoid Conda shells — use Command Prompt or PowerShell without Conda initialised.

### Permanent Fixes

**A. Remove Conda from system PATH (optional but clean)**

Remove from `Environment Variables → System PATH`:
```
C:\Users\admin\miniconda3\
C:\Users\admin\miniconda3\Scripts
```

**B. Recreate the virtual environment with the correct Python**

```bash
C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe -m venv C:\Users\admin\Documents\envs\env_312

# Activate and verify
C:\Users\admin\Documents\envs\env_312\Scripts\activate
where python
where pip
python --version
pip list
```

### Best Practices

- Always use `python -m pip ...` to ensure you're using the correct pip.
- Avoid creating venvs from within a Conda shell unless intentionally using Conda environments.
- Consider `pyenv` or `virtualenvwrapper` when working across multiple Python versions.

---

## 14. Git Reference

### Essential Commands

```bash
git init
git remote add origin "https://..."
git add .
git commit -m "message"
git push origin main
```

### Daily Workflow

```bash
# At the start of every session (office or home)
git pull --rebase origin main

# After working
git add .
git commit -m "Implemented X feature"
git push
```

### Credential Management

```powershell
# Remove stored GitHub credentials
cmdkey /delete:git:https://github.com

# Set global Git identity
git config --global user.name "sultan-rrcat"
git config --global user.email "sultan-rrcat@gmail.com"
```

### Cleanup Checklist (Before Leaving)

- Remove Chrome profile
- Remove `cdtrainee` credential from Windows Credential Manager
- Remove GitHub authorisation from Git (or uninstall Git)
