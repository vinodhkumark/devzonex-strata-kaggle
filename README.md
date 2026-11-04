# 🚀 Strata on Kaggle — by DevZoneX

[![Tests](https://github.com/iAmAjayTeli/devzonex-strata-kaggle/actions/workflows/tests.yml/badge.svg)](https://github.com/iAmAjayTeli/devzonex-strata-kaggle/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Kaggle%20GPU-20BEFF)
![Status](https://img.shields.io/badge/status-experimental-orange)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/iAmAjayTeli/devzonex-strata-kaggle/blob/main/notebooks/Strata_Kaggle_Launcher.ipynb)

**Run a Strata-supported model on a Kaggle GPU, cache the model and Strata runtime in your own private Kaggle Datasets, and connect OpenCode Desktop through a temporary authenticated HTTPS tunnel.**

This project turns our Strata + Qwen3.8-Flash-Next Gyro-S setup into a guided notebook. The default example is `gyro / gyro-s`; other choices must be supported by the installed Strata release.

## 🧠 Which model are we running, and why is it hard to run?

The default example is **Qwen3.8-Flash-Next — Gyro-S**, using Strata's `gyro / gyro-s` model family and variant. It is a large **125-billion-parameter sparse Mixture-of-Experts (MoE) model**, with about **6 billion parameters active for each token**. “6B active” does **not** mean the whole model only occupies the memory of a 6B model: its full weights and supporting data still need to be stored and accessed.

Sources: [Strata model guide](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md) · [Gyro quantization details](https://github.com/Niko1221/Strata/blob/main/docs/GYRO.md) · [NVIDIA's Qwen3.8-Flash-Next architecture overview](https://docs.nvidia.com/dynamo/dev/recipes/qwen-3-8-flash-next).

### Model size and memory at a glance

| Model / format | Download or disk size | Estimated model memory (RAM + VRAM) |
|---|---:|---:|
| **Gyro-S (our default)** | **58.5 GB** for the main model file | **27.6 GB combined** |
| Gyro-M | 92.0 GB | 35 GB combined |
| Qwen3.8-Flash-Next Q2_0 (standard Strata option) | varies by pack/source | 37.6 GB combined |
| Qwen3.8-Flash-Next IQ2_XS | varies by pack/source | 39.2 GB combined |
| Qwen3.8-Flash-Next IQ3_XXS | varies by pack/source | 47.0 GB combined |
| Qwen3.8-Flash-Next IQ3_S | varies by pack/source | 54.8 GB combined |
| Higher 4–8-bit Unsloth formats | roughly 94 GB to 136 GB of files, depending on format | roughly 72 GB to 136 GB combined, depending on format |

These are upstream estimates for the **model weights**, not a promise that every format fits every machine. Leave memory for the operating system and other processes. Actual needs depend on context length, engine/cache settings, and which quantization you select. Gyro-S is the smallest option and trades some output quality for lower memory use. See the [official Strata model guide](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md) for updated estimates and compatibility.

### What hardware are we using in this project?

Our working Kaggle setup used **2 × NVIDIA Tesla T4 GPUs (about 15 GB VRAM each, around 30 GB total VRAM exposed)** plus roughly **31 GB system RAM**. That is a specific observed Kaggle session, not a guarantee that every account will receive the same GPU or RAM.

We use **Strata + the Gyro-S quantization**, rather than trying to load a larger, less-compressed version entirely into GPU memory. Strata manages expert weights across available VRAM and system memory; Gyro-S also keeps its large n-gram/PLE table on disk and reads relevant data as needed. That is why a model with 125B total parameters can be practical on this setup even though it would not fit as a conventional full-weight GPU-only load. The trade-off is that SSD access and memory pressure can reduce speed, and the Gyro-S quantization has lower quality than larger variants.

Our model cache is about **79.94 GiB** because it includes more than the headline GGUF file: model data, expert/dense pack files, tokenizer/support assets, and a path manifest. That is why it gets its own dataset; the MTP draft layer has another (5.2 GB downloaded and packed into a ~1 GiB `mtp/rt`), and the Strata runtime cache is about **1.3 GiB**.

### Why does Kaggle still show “Downloaded” after I attach the dataset?

**Add Input makes the dataset available to the notebook; it does not teleport 80 GiB onto the new runtime.** Kaggle still has to make those attached dataset files available to the session. In the default `symlink` mode the notebook does **not** copy them into `/root/strata-local`; it links each file to the read-only `/kaggle/input` mount. The staging progress Kaggle shows is dataset transfer, not a fresh download of the original model from its source, and no notebook option can remove it.

The benefit is **persistence and reuse**: future sessions read the model from your Kaggle Dataset instead of fetching it from the original host and recompiling the engine. Local disk holds only the extracted runtime, about 1.5 GiB instead of ~80 GiB. The trade-off is that model reads go through the input mount, which can be slower; measure it with the quality gate.


> [!WARNING]
> **Experimental project.** The complete first-run and fresh-session workflow still needs validation on more Kaggle accounts and GPU configurations. Review the Strata launch plan before approving large downloads. Kaggle availability, quotas, GPU types, model licenses, and Strata compatibility vary.

## ✨ What it does

| Stage | What happens |
|---|---|
| **1 · Preflight** | Checks Kaggle Secrets, GPU visibility, Internet, attached datasets, and Strata version. |
| **2 · Choose a model** | Select a Strata-supported family/variant, context, vision option, and GPU list. |
| **3 · Restore or install** | Restores your private caches when attached; otherwise installs Strata and lets it fetch compatible files. |
| **4 · Review plan** | Streams Strata's plan output into the notebook so you can review compatibility before downloading large files. |
| **5 · Start securely** | Streams Strata launch logs into the notebook, starts Cloudflare access mode, and creates/reuses a temporary Quick Tunnel. |
| **6 · Verify** | Checks API-key enforcement, model listing, health, and a short authenticated generation. |
| **7 · Cache** | Optionally uploads model files and the Strata terminal/engine into separate private datasets you own. |
| **8 · OpenCode** | Prints the temporary HTTPS base URL, model ID, and provider configuration snippet. |
| **9 · Optional supervision** | Runs a six-hour health-monitoring cell that checks Strata and the tunnel authentication boundary. |

## 🧭 Quick links

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/iAmAjayTeli/devzonex-strata-kaggle/blob/main/notebooks/Strata_Kaggle_Launcher.ipynb)

If the badge does not open the notebook import flow, go to [Kaggle Notebooks](https://www.kaggle.com/notebooks/welcome), choose **Load from URL**, and paste:

```text
https://github.com/iAmAjayTeli/devzonex-strata-kaggle/blob/main/notebooks/Strata_Kaggle_Launcher.ipynb
```

Opening the notebook imports its code; it does **not** grant access to anyone else's private datasets or secrets.

- **[Open the Kaggle launcher notebook](notebooks/Strata_Kaggle_Launcher.ipynb)** — serves the model
- **[Open the cache builder notebook](notebooks/Strata_Cache_Builder.ipynb)** — CPU probe, dataset creation, GPU handoff
- [Kaggle setup checklist](docs/kaggle-setup.md)
- [OpenCode connection guide](docs/opencode-setup.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Security policy](SECURITY.md)
- [Strata official setup guide](https://github.com/Niko1221/Strata/blob/main/docs/AI_SETUP.md)
- [Strata model compatibility](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md)

---

## 🗺️ Visual walkthrough

![Strata on Kaggle workflow overview](assets/workflow-overview.svg)

![Summary of the manually observed first-run result](assets/observed-first-run.svg)

These are **workflow/results illustrations**, not raw UI screenshots. The screenshots shared during setup were used to ground the run summary below.

## ✅ Verification status

### Manually observed as working

- **First launch:** Strata reported `Strata is serving` for `qwen3.8-flash-next-gyro-gyro-s`; the local health endpoint reported `"loaded": true`.
- **Hardware:** the dashboard showed two Tesla T4 GPUs with approximately 29 GiB / 30 GiB VRAM allocated.
- **Remote tunnel and API security:** Cloudflare Quick Tunnel connected; the public API rejected a request without a key, and the authenticated API test succeeded.
- **Model cache upload:** Kaggle showed a private Version 1 model dataset of approximately 85.83 GB with the model and support files.
- **Runtime cache upload:** Kaggle showed a private Version 2 runtime dataset of approximately 1.36 GB with the archive, checksum, and manifest.
- **Fresh-session restore and OpenCode:** the maintainer reports that, after attaching the private model/runtime datasets in a fresh Kaggle run, Strata launched and OpenCode connected and returned responses successfully.
- **Live progress:** Copy-mode model restoration reports live bytes/GiB restored, percentage, completed files, active copies, throughput, and elapsed time. Strata launch and Kaggle upload helpers also stream command output and print periodic status when a process is quiet.

### Still to validate across more environments

The main end-to-end workflow has been manually verified for the default `gyro / gyro-s` setup. The full six-hour monitoring period has not yet been confirmed to complete uninterrupted, and other model families, GPU types, Strata versions, and Kaggle account configurations need separate tests. The project remains experimental for those broader environments.

## ✅ Before you start

You need:

- A Kaggle account, Internet access, and an available compatible GPU accelerator.
- A Kaggle API token stored as a notebook Secret.
- Your exact Kaggle username.
- Enough session time, local disk space, and dataset quota for the model and supporting files.
- A model whose license permits downloading and use. To cache it, the license must also allow a private copy.
- OpenCode Desktop if you want to use the model through OpenCode on Windows.

### Model compatibility matters

This is **not** a universal Hugging Face model runner. Strata supports particular families, variants, formats, and hardware combinations. A model being downloadable does not mean Strata can run it. Use the official Strata compatibility list and inspect the launch plan before proceeding.

The default tested example is:

```python
MODEL_FAMILY = "gyro"
MODEL_VARIANT = "gyro-s"
CONTEXT = "32k"
VISION = "off"
```

Start with text-only mode.

---

# 🧑‍💻 First-time setup — full walkthrough

## 1. Open the notebook in Kaggle

1. Open [`notebooks/Strata_Kaggle_Launcher.ipynb`](notebooks/Strata_Kaggle_Launcher.ipynb) on GitHub.
2. Download the `.ipynb` file and use Kaggle's notebook import flow, or create a blank notebook and copy the cells in order.
3. Enable **Internet** and select an available NVIDIA GPU accelerator.
4. If Kaggle requests a restart after changing settings, restart the notebook before running any cells.

GPU availability, session length, and quotas depend on Kaggle's current policies and your account.

The first code cell clones the helper repository into `/kaggle/working` and checks
that it contains the helpers this notebook calls. `REPO_URL` in that cell must
point at the repository you actually want to run. If you fork this project,
update `REPO_URL` to your fork; otherwise Kaggle clones a checkout without these
features and fails later with `AttributeError: module 'scripts.strata_kaggle'
has no attribute ...`.

## 2. Add Kaggle API credentials as a Secret

1. Create an API token using Kaggle's current account settings.
2. In the notebook, open **Add-ons → Secrets**.
3. Create a secret named exactly `KAGGLE_API_TOKEN`.
4. Paste the token into the secret value and enable access for this notebook.
5. Never put the token in a code cell, output, GitHub file, screenshot, or chat.

The notebook reads the token through Kaggle's `UserSecretsClient` and stores it in a restricted temporary config path for CLI commands.

## 3. Configure the model and your cache names

In the **USER CONFIGURATION** cell, edit at least:

```python
KAGGLE_USERNAME = "your-kaggle-username"
MODEL_FAMILY = "gyro"
MODEL_VARIANT = "gyro-s"
CONTEXT = "32k"
VISION = "off"
GPU_IDS = ""  # blank = auto-detect visible GPUs
MODEL_DATA_DIR = "/root/strata-local"

# "symlink" reads the model/packs/MTP from attached datasets; local disk holds
# only the compiled runtime (~1.3 GiB). Default: an ~80 GiB model does not fit.
# "copy" duplicates bytes locally for faster reads; needs the free disk and time.
MODEL_RESTORE_MODE = "symlink"

# The MTP draft layer has its own dataset because it is shared by every variant.
# Blank = use your own <KAGGLE_USERNAME>/strata-mtp-rt when attached.
DRAFT_LAYER_DATASET = ""
DRAFT_LAYER_DATASET_SLUG = "strata-mtp-rt"

# Optional: use an attached dataset that holds raw Strata model files
# (Dense/PLE GGUFs, experts.bin, profile) instead of downloading them.
MODEL_FILES_DATASET = ""

# One dataset per artifact.
UPLOAD_CACHES = True                 # master switch
UPLOAD_RUNTIME_CACHE = True          # ~1.3 GiB
UPLOAD_DRAFT_LAYER = True            # ~1 GiB final, shared by variants
UPLOAD_MODEL_CACHE = True            # ~80 GiB
I_CONFIRM_MODEL_LICENSE_AND_PRIVATE_STORAGE = False
FORCE_UPDATE_MODEL_CACHE = False
FORCE_UPDATE_RUNTIME_CACHE = False
FORCE_UPDATE_DRAFT_LAYER = False

# GPU-hour guards: the supervision cell stops Strata when nobody is using it.
IDLE_MINUTES = 15                    # finish after this many idle minutes; 0 disables the check
SERVE_MINUTES = 0                    # hard stop for the supervision cell; 0 = only the idle check

# Quality gate: benchmark the chat API and check tool calling.
RUN_QUALITY_GATE = False             # several extra minutes; run it once per model
QUALITY_REPEATS = 1

# Optional PEM RSA public key from a local launcher. When set, the API key is
# published encrypted and never appears in notebook output.
CLIENT_PUBLIC_KEY = ""
```

`IDLE_MINUTES` and `SERVE_MINUTES` protect your GPU quota. The supervision cell
stops Strata after the configured idle time, and returning from that cell lets
Kaggle finish the run and release the GPU instead of idling to the session limit.
Set `SERVE_MINUTES = 0` (the default) when you want only the idle check.

`MODEL_DATA_DIR` is session scratch. It sits outside `/kaggle/working`, so the
20 GB output quota does not apply to the model. Kaggle's `/tmp`, `/root` and
`/kaggle/temp` are not separate larger disks, so download once to the path you
will upload from; copying between them needs twice the space. The CPU download
is ~63.7 GB decimal, about 59.3 GiB, which is what `df` reports.

Kaggle's whole writable layer is about **70 GB**, and Kaggle staff recommend
staying under ~60 GB. `df` can show the host's multi-terabyte overlay instead;
that is not the notebook's quota. A Gyro-S one-pass download peaks near 63 GiB,
so use the two-pass flow in section 3c: upload the model, wait for **Ready**,
delete `models/` and `packs/`, then build and upload the draft layer. Do the
same rotation to try another model, because two ~58 GB models cannot coexist in
one session. Each model has its own dataset; attach only the one you are serving.

`RUN_QUALITY_GATE = True` runs a benchmark and a tool-calling check before you
connect a client; see [step 3c](#3c-optional-quality-gate).

**Replace the placeholder Kaggle username.** Dataset names are derived from your configured username and model family/variant.

By default the notebook plans to use these separate datasets, one per artifact:

- `<your-username>/strata-runtime-cache` — compiled terminal + engine (~1.3 GiB)
- `<your-username>/strata-mtp-rt` — MTP draft layer, shared by every variant (~1 GiB final; 5.2 GB downloaded)
- `<your-username>/strata-model-<family>-<variant>` — `models/` + `packs/` (~80 GiB)

These are user-owned caches, not global datasets shared by the project owner. Attach only datasets you own or have been granted access to. The public `clothespin/strata-qfn-assets-lite` draft-layer mirror is opt-in only: leave `DRAFT_LAYER_DATASET` blank to use your own dataset.

### Confirm the model license before caching

Keep `I_CONFIRM_MODEL_LICENSE_AND_PRIVATE_STORAGE = False` until you have checked that the model's license/terms permit making and storing a private copy in Kaggle. If the flag remains false, the model upload is skipped. Do not upload model weights to GitHub.

### When the accelerator is required

Build the datasets once with [`notebooks/Strata_Cache_Builder.ipynb`](notebooks/Strata_Cache_Builder.ipynb), then use the launcher for every serving session.

| Phase | Accelerator | What happens |
|---|---|---|
| Load helpers, restore the runtime, probe `strata --help` and `--plan` | **None (CPU)** | no CUDA needed; the plan is cheap and lists the real files and sizes |
| Fetch the model, build packs, fetch the draft layer | **None (CPU)** | `prepare_model_on_cpu()` imports the installed `setup.py` and calls its download and preparation helpers directly (`setup.download`, `tools/iq_pack.py`, `tools/mtp_fetch.py` + `mtp_pack.py` + `mtp_rt.py`). `strata install` / `strata launch` cannot do this: their `main()` stops on a machine with no NVIDIA/AMD card before downloading anything |
| Compile a compatible engine and serve | **GPU (T4 ×2)** | the engine is compiled for the GPU it runs on (`sm_75` for T4); the runtime dataset already provides a matching one |
| Normal serving session | **GPU (T4 ×2)** | run the launcher |

A CPU session cannot *serve* a model, but it can create all three datasets. If a
Strata build does not expose the helpers the CPU path calls, the cell says so and
section 4's GPU session is the fallback. Switching the accelerator restarts the
kernel and wipes `/root`, so run the upload cell in the same session that
produced the files. Uploads also consume GPU wall-clock when the accelerator is
attached.

Use [`notebooks/Strata_Cache_Builder.ipynb`](notebooks/Strata_Cache_Builder.ipynb) for the CPU probe, the prepare attempt and the upload; use the launcher for serving.

## 4. Run the notebook cells in order

1. **Load helpers and credentials** — clones the repository's scripts and reads your Kaggle Secret. It also validates the configuration and reports GPU, RAM and free disk before a long launch. Free disk is reported for `MODEL_DATA_DIR`, which is the filesystem the model would land on — not `/kaggle/working`.
2. **Restore caches or install Strata** — uses attached private datasets when available; installs Strata if not.
3. **Review plan** — inspect Strata's plan and confirm the selected model, engine, context, and GPU setup before downloading.
4. **Launch Strata** — fetches missing compatible model/engine files on a first run.
5. **Tunnel and health check** — checks local health and API-key protection, starts a Quick Tunnel if Strata hasn't provided a URL, and verifies the public endpoint.
6. **Authenticated generation test** — confirms chat completions work.
7. **Upload private caches** — stores the model and runtime separately, if enabled and permitted.
8. **Print OpenCode config** — provides the current URL/model/provider snippet.
9. **Supervise and release** — watches request activity and stops Strata when idle.

### 3a. Optional: use model files from an attached dataset

If an attached dataset already holds the model as raw Strata files (`*-Dense.gguf`,
`*-PLE.gguf`, `experts.bin`, a `profile-*.bin`), set:

```python
MODEL_FILES_DATASET = "your-username/the-dataset-slug"   # owner may be omitted
```

The cache cell symlinks those files into `MODEL_DATA_DIR/models/<family>-<variant>`,
and the launch passes `--model-path` there, so Strata skips downloading the model.

Two caveats, both deliberate:

- **No manifest means no verification.** These bundles carry no record of which
  family/variant they belong to, so the notebook cannot check that they match
  `MODEL_FAMILY`/`MODEL_VARIANT`. Strata's loader is what decides, and a mismatch
  fails at launch instead of serving a different model.
- **Symlinks are the default** because a bundle can be far larger than local disk.
  Reads go through the `/kaggle/input` mount, which is slower than a local copy.

This is separate from the model **cache** created by the upload cell, which does
have a `manifest.json` and is preferred when present.

### 3b. Optional: measure the drafter

`RUN_DRAFTING_AB = True` reloads the model twice, once with `DRAFTING = "off"` and
once with `"auto"`, and reports the measured difference. It is slow and opt-in.

### 3c. Optional: quality gate

`RUN_QUALITY_GATE = True` does two things before you connect a client:

- **Benchmark** — prompt and decode tokens/s plus time-to-first-token at 128, 512
  and 2048-word prompts, using the server's own timings when it reports them.
- **Tool calling** — gives the model `read_file`, `write_file` and `run_test`
  tools, asks it to fix a local file so a function returns 42, and checks that
  the edit actually happened and the test really ran.

The tool check is the one that predicts whether an agent session will work: a
model can decode quickly and still emit invalid tool calls, edit before reading,
or claim success without running the test. It is also how to compare two model
variants on something more useful than tokens/s. Expect several extra minutes.


Before the endpoint is considered ready, the checks should establish:

- `/health` reports the model loaded and API-key protection enabled.
- An unauthenticated `/v1/models` request returns `401` or `403`.
- An authenticated model listing succeeds.
- An authenticated `/v1/chat/completions` request returns text.

If an unauthenticated public request succeeds with HTTP `200`, stop the tunnel and fix authentication before continuing.

## 5. Wait for private dataset processing

| Cache dataset | What's saved | Purpose |
|---|---|---|
| Model cache (`strata-model-<family>-<variant>`) | `models/` + `packs/` plus path-and-size manifest | Avoid re-downloading model assets from their original source |
| Draft layer (`strata-mtp-rt`) | `mtp/rt/` as flat `mtp_rt__*` files plus manifest | Fetch the 5.2 GB MTP tensors once; the uploaded layer is ~1 GiB |
| Runtime cache (`strata-runtime-cache`) | Strata terminal, engine components, manifest/checksum, and optionally Cloudflare Tunnel binary | Avoid recompiling the engine (~10–15 min) and re-downloading the same compatible runtime build |

The notebook creates datasets without a public-visibility flag. Verify each dataset's visibility and file list on Kaggle. “Upload successful” means the files transferred; it does not necessarily mean Kaggle finished processing the dataset.

Large uploads may be rejected due to account limits, disk space, network issues, or model license terms.

---

# ⏱️ Supervise the session and release GPU hours

After the launch, tunnel, and authenticated API test succeed, run the
**supervision cell** and leave it as the last cell that runs. It:

- checks once that the public URL still rejects a no-key request,
- watches the server's `/v1/status` request activity,
- stops Strata after `IDLE_MINUTES` without a request, or after `SERVE_MINUTES`
  if you set one,
- prints a `STRATA_SHUTDOWN={...}` record with the reason and elapsed time.

Returning from that cell lets the notebook finish, and a finished notebook
releases the GPU allocation. That is the difference between losing 15 minutes of
quota to an abandoned session and losing hours to it.

The cell does **not** override Kaggle's session or GPU quota limits and cannot
guarantee uptime. Kaggle's published docs list up to 12 hours for CPU/GPU
notebook executions and a shorter time for TPU sessions; limits can change.

For long executions, Kaggle's **Save Version → Save & Run All** workflow can run
in the background until the platform's runtime cap. However, the full notebook
must finish or reach its supervision cell within the permitted session, and you
must be able to retrieve the emitted tunnel URL. Do not treat a free notebook as
an always-on production server.

---

# 🔁 Restore on a later Kaggle session

1. Reopen the notebook and enable Internet/GPU.
2. Use **Add Input** to attach your own runtime, model and draft-layer datasets (`strata-runtime-cache`, `strata-model-<family>-<variant>`, `strata-mtp-rt`).
3. Enable the `KAGGLE_API_TOKEN` Secret for this notebook.
4. Keep the model family/variant and username consistent with your cache names.
5. Run the notebook from top to bottom.

The model dataset uses a `manifest.json`; the runtime dataset contains `strata-runtime-cache.bin`, `strata-runtime-cache.bin.sha256`, and `runtime-manifest.json`; the draft-layer dataset uses `mtp_rt__*` files plus a `manifest.json`.

The notebook verifies the manifest and file sizes, then uses the `MODEL_RESTORE_MODE` option:

- `"symlink"` (default): creates links in `/root/strata-local` to the attached read-only Kaggle datasets, so local disk holds only the extracted runtime (~1.3 GiB). Reads from the input mount may be slower, especially for MoE expert paging.
- `"copy"`: copies the model files into `/root/strata-local`. This takes extra restore time and disk space, but may provide faster local file access. It is usually impossible for a ~80 GiB model on Kaggle's local disk.

If an earlier `copy` restore left real files behind, `symlink` mode keeps them and logs a warning with the disk they still occupy. Delete `MODEL_DATA_DIR/models` (and `packs`, `mtp`) before restoring if you want symlink-only storage.

**Why Add Input can still show a download progress bar:** Kaggle must make the selected dataset files available to the fresh runtime; this is a transfer from your Kaggle Dataset storage into that notebook session, not the same as downloading the model again from its original model host. No notebook option can eliminate this input-staging step. `symlink` avoids only the second copy to local disk.

Measure the trade-off once with `RUN_QUALITY_GATE = True`. If decode tokens/s are unacceptable in symlink mode, copy only `models/` while symlinking `packs/` and `mtp/` — but that needs the free disk the default design avoids.

Runtime caches may not be compatible after a Strata, engine, OS, CUDA, or GPU change. Review the plan again and allow Strata to fetch a compatible component when needed.

---

# 🖥️ Connect OpenCode Desktop

1. Copy the **current** HTTPS URL printed by the notebook.
2. In the Kaggle shell, retrieve the API key privately with `strata launch --show-key`. Do not include it in notebook output or chat.
3. On Windows, save it locally as `%USERPROFILE%\.config\opencode\strata-key.txt`.
4. Merge the printed provider snippet into the existing top-level `providers` object in `opencode.jsonc`. Don't overwrite other provider or MCP settings.
5. Set the model to the provider/model ID printed by the notebook, restart OpenCode, and test in a fresh session.

Keep the Kaggle runtime and tunnel running while using OpenCode. Quick Tunnel URLs are temporary, public URLs, and can change on restart. Anyone who obtains a URL can reach the endpoint, so API-key enforcement is required. Update the OpenCode base URL after every tunnel change.

---

## 🔐 Security and licensing

- Never commit API tokens, local OpenCode keys, Kaggle credentials, model weights, runtime archives, or saved notebook outputs containing secrets.
- Keep model/runtime datasets private and check their visibility in the Kaggle UI.
- Verify API-key protection before using a public tunnel; stop the tunnel if the no-key request isn't rejected.
- Stop Strata and Cloudflare when you're finished. This is temporary experimental hosting, not production infrastructure.
- Check the model terms before making a private cache copy.
- The repository's MIT license applies only to the original project code/docs; third-party software and model files retain their own licenses.

## 🛠️ Troubleshooting

| Problem | What to check |
|---|---|
| No GPU | Enable the accelerator and restart if requested; check `nvidia-smi`. |
| Secret missing | Confirm `KAGGLE_API_TOKEN` spelling and notebook access. Never print the value. |
| Dataset missing | Attach it through Add Input and verify username/slug. |
| Kaggle API 403 | Check token/account/ownership and whether dataset processing has completed; don't immediately repeat a 1+ GB upload. |
| Model unsupported | Choose a family/variant supported by the installed Strata release; review the launch plan. |
| Launch cell shows only the first status line | The launch helper used by an already-running cell may buffer output until Strata exits. The latest helper streams logs and progress live; the notebook's helper-load cell now pulls the latest repository version and reloads the module. Rerun that cell before starting a new launch. 
| Cache upload shows only “Preparing … files” | The currently running notebook may still have the older helper imported, which buffers Kaggle CLI output. Don't interrupt an active upload. The latest helper streams Kaggle upload progress and emits a 30-second heartbeat if the CLI is silent; rerun the helper-load cell before another cache upload. |
| Model download slow | Check local disk space, network, model size, and remaining session time. |
| Cache upload skipped | Check the explicit model-license confirmation and upload logs. |
| Runtime cache fails with `Text file busy` | A Strata Python executable is still in use while restoring. Rerun the helper-loading cell to get the latest version; it reuses a matching installed runtime or stops Strata and extracts to staging before replacing files. Then rerun the restore cell. |
| Tunnel unavailable | Confirm local health, `cloudflared` is running, and the URL is current. |
| OpenCode 401 | Check the local key file and provider config without printing the key. |
| OpenCode hangs | Test authenticated `/v1/models` and `/v1/chat/completions`, inspect logs, and check model context limits. |

## 🧪 Development and tests

Run the helper tests locally:

```bash
python -m pip install --upgrade pytest
python -m compileall -q scripts tests
python -m pytest -q
```

Unit tests do not replace a clean Kaggle integration test. Validate first run, private uploads, fresh-session restore, no-key rejection, authenticated generation, six-hour monitoring, and OpenCode connectivity.

## 🤝 Contributing

Please report the Strata version, GPU, whether it was first run or cache restore, and the error with secrets removed. Never attach model files, credentials, or sensitive notebook output to issues.

## 📄 License

Original project code and documentation: [MIT](LICENSE). Third-party components and models retain their respective licenses.
