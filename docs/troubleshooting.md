# Troubleshooting

## `AttributeError: module 'scripts.strata_kaggle' has no attribute ...`

The notebook cloned a different checkout than the one that contains these
helpers. The first code cell prints nothing about this directly; the failure
appears in a later cell, for example:

```
AttributeError: module 'scripts.strata_kaggle' has no attribute 'validate_config'
```

Cause: `REPO_URL` in **1. Load helper scripts and Kaggle credentials** points at
another repository, usually the original upstream project rather than the fork
with these features.

Fix:

1. Edit `REPO_URL` in that cell to the repository you intend to run, for example
   `https://github.com/<you>/devzonex-strata-kaggle.git`.
2. Rerun the cell. It replaces a stale checkout instead of failing on `git pull`.
3. If the checkout still reports missing helpers, the cell now names them and the
   URL it used, so no later cell fails with a bare `AttributeError`.

The cell verifies `validate_config`, `environment_report`, `announce_session`,
`serve_until_idle`, `run_quality_gate` and `publish_connection` before using them.

## Model cache reports "not attached" but the dataset exists

"It exists in **Your Datasets**" and "attached to this notebook" are different
things. Only datasets added to the notebook's **Input** section are mounted.

The cache cell now prints what is actually mounted. Two outcomes:

- `Attached datasets: ...` — the dataset is mounted, but under a different name
  or without the expected marker. Compare the printed paths with the slug the
  notebook derived from `KAGGLE_USERNAME`, `MODEL_FAMILY` and `MODEL_VARIANT`.
- `No dataset is attached to this session` — nothing is mounted. Add it with
  **Add Input → Your Datasets**, then **restart the session**: Kaggle mounts
  inputs at session start, so attaching during a running session often does not
  take effect.

To inspect the mount directly in a scratch cell:

```python
import scripts.strata_kaggle as sk
print(sk.dataset_mount_report())
```

The report lists the roots that were searched and every dataset directory found,
without reading the dataset's files. A model cache also needs `manifest.json` at
the dataset's top level; a runtime cache needs `strata-runtime-cache.bin`.

## The caches are in a differently named dataset

The notebook derives the model dataset name from `MODEL_FAMILY` and
`MODEL_VARIANT`, so `gyro`/`gyro-s` means `strata-model-gyro-gyro-s`. A dataset
mounted under any other name is still used when it contains a valid
`manifest.json` with `"format": "devzonex.strata-model-cache.v1"`, either at its
top level or one directory down. The log then names the dataset it used.

To see what each mounted dataset holds:

```python
import scripts.strata_kaggle as sk
for entry in sk.describe_inputs():
    print(entry)
```

Each entry reports the top-level names, plus `model_cache` (family, variant, file
count, GiB) when a model manifest is present and `runtime_cache_gib` when the
runtime archive is present. Only names and manifests are read, so running it on a
28 GiB mount is fast.

If no entry shows `model_cache`, the dataset does not contain a model cache in
this format — the weights may have been uploaded as plain files instead. Attach a
dataset produced by the notebook's upload cell, or let Strata download the model
once and save it with `I_CONFIRM_MODEL_LICENSE_AND_PRIVATE_STORAGE = True`.

## The weights are attached as plain files, without a manifest

A dataset can hold a model as its raw Strata files (`*-Dense.gguf`, `*-PLE.gguf`,
`experts.bin`, `profile-*.bin`) with no `manifest.json`. The notebook's model
cache requires that manifest, so those files are ignored by default.

Point the launch at them instead:

```python
MODEL_FILES_DATASET = "your-username/the-dataset-slug"   # owner may be omitted
```

The cache cell symlinks the files into `MODEL_DATA_DIR/models/<family>-<variant>`
and the launch uses `--model-path`. If the launch then rejects the model, the
bundle does not match `MODEL_FAMILY`/`MODEL_VARIANT` — clear the setting and let
Strata download the correct files.

## Model family or variant is unsupported

Run the notebook's Strata plan cell and choose a family/variant accepted by the installed Strata release. Not every Hugging Face model is compatible with Strata. Review the plan before approving large downloads.

## Runtime cache doesn't restore

Attach your private runtime dataset through **Add Input** and verify that it contains:

- `strata-runtime-cache.bin`
- `strata-runtime-cache.bin.sha256`
- `runtime-manifest.json`

The notebook verifies the checksum before extracting the archive. The runtime archive filename ends in `.bin` intentionally so Kaggle doesn't automatically unpack the tar archive during dataset creation.

The current helper safely rewrites absolute symlinks recorded inside the Strata runtime archive (such as `terminal/current`) into relative links during extraction. It still rejects links that point outside the Strata runtime root or cross between `terminal/`, `components/`, and `tools/`. If an existing notebook still raises `Unsafe absolute link in runtime archive`, rerun **Load helper scripts and Kaggle credentials** so it reloads the latest helper from GitHub.

If the cached Strata/engine build doesn't match the runtime, the helper may skip or replace the cached restore and install/fetch a compatible version. Don't assume an older engine is compatible with every Strata release or GPU.

## Model files are missing or the model appears to download again

Attach your private model dataset. It should contain `manifest.json` and numbered flattened files. The notebook uses the manifest to reconstruct the model paths.

- `MODEL_RESTORE_MODE = "symlink"` (default) links to the mounted, read-only Kaggle input files, so local disk holds only the extracted runtime. Direct reads may be slower, especially for MoE expert paging.
- `MODEL_RESTORE_MODE = "copy"` copies files into `/root/strata-local`. It takes extra restore time/disk space but may improve local model reads. An ~80 GiB model usually does not fit.

**Why Add Input shows a download percentage:** Kaggle stages the attached dataset into the new runtime. For an 80-GiB model it can show progress such as `Downloaded 14/16 files: 4.5 GiB / 79.94 GiB`. This is a transfer from your Kaggle Dataset storage, not necessarily a download from the original model host. Wait until staging and the restore cell finish. If the model cache restore returns `False`, verify that the dataset slug, username and manifest match your configuration before running Strata.

## Local disk fills up even though symlink mode is set

Symlink mode keeps a file that already exists locally as a real copy instead of
replacing it, so a previous `copy` restore can silently keep tens of GiB on disk.
The log warns with the count and GiB that were kept. Inspect and clear it:

```bash
!du -sh /root/strata-local/*
!rm -rf /root/strata-local/models /root/strata-local/packs
```

Then rerun the cache cell. `sk.cache_status(MODEL_ROOT)` prints file counts,
bytes and symlink counts per root, so you can see which parts are real files.

## A CPU session cannot create the model dataset

Only the runtime restore, `strata --help` / `--plan`, and uploading files that
already exist are CPU-safe. The model, `packs/` and the MTP layer appear only
after `strata launch`, and the engine is compiled for the GPU it runs on
(`sm_75` for T4). Use `notebooks/Strata_Cache_Builder.ipynb`:

1. On CPU, restore the runtime and run the probe. If `strata` exposes a
   fetch/prepare command that works without CUDA, use it.
2. Otherwise switch to **T4 ×2**, rerun the load/restore cells (the accelerator
   change restarts the kernel and wipes `/root`), then create and upload the
   datasets in that same session.

Changing the accelerator never preserves `/root`, so upload before ending the
session. Uploads with the accelerator attached consume GPU wall-clock.

## The draft layer is restored from the wrong dataset

The MTP layer lives in its own dataset, `<your-username>/strata-mtp-rt`, and is
shared by every Qwen3.8-Flash-Next variant. `DRAFT_LAYER_DATASET = ""` resolves
to your own slug; it does not disable the restore. If that dataset is not
attached, the helper logs that Strata will fetch the layer itself (~5.2 GB).

Set `DRAFT_LAYER_DATASET = "owner/slug"` only to read another copy. The public
`clothespin/strata-qfn-assets-lite` mirror is opt-in, not a default. A layer from
a different Strata build logs a size note; the engine validates it at start.


## Cache upload appears frozen

The notebook helper now streams Strata and Kaggle CLI output and prints periodic status while long commands run. If a cell only shows `Preparing ... files`, it may be using a helper that was imported before the latest update.

1. Do not interrupt an upload that is still transferring files.
2. After the current attempt has finished (or if it failed), rerun **Load helper scripts and Kaggle credentials** to pull and reload the latest helper code.
3. Rerun only the cache-upload cell if needed. Check Kaggle's dataset page/version before trying another upload.

A 79.94-GiB model upload can take a long time. An output line saying `Dataset is being created` means the transfer was submitted; it does not prove dataset processing has completed.

## Kaggle dataset slug already exists

The helper attempts to create a dataset first. If the requested slug already exists, it should upload a new dataset version rather than create a duplicate. Wait for processing to finish and confirm the intended Version page in the Kaggle UI before repeating the command.

## Dataset upload permissions fail

Check that the Kaggle Secret is named exactly `KAGGLE_API_TOKEN`, notebook access is enabled, and `KAGGLE_USERNAME` matches the account that owns the dataset. Dataset processing can continue after all files have been uploaded. Verify dataset visibility and ownership in Kaggle.

## API key test fails

The public `/v1/models` endpoint must reject unauthenticated requests with HTTP `401` or `403`. Do not continue using a public tunnel if the unauthenticated request succeeds. Check Strata's access mode and use `--access cloudflare`.

Never paste the API key into notebook output, GitHub, issues, or chat.

## Cloudflare URL times out or changes

A Quick Tunnel exists only while its process/session is active. Use the latest HTTPS URL printed by the notebook, check the Cloudflare log if needed, and update OpenCode's `baseURL`. Keep the Kaggle runtime and tunnel process running while using the model.

## OpenCode returns 401 or hangs

- For `401`, check the local key file and OpenCode provider settings without printing the key.
- Test the authenticated `/v1/models` and `/v1/chat/completions` endpoints independently before troubleshooting OpenCode.
- Confirm the configured context/output limits match the running Strata model. After changing compaction/provider settings, start a fresh OpenCode session.

## Supervision cell exits or never stops

The supervision cell stops Strata for one of two reasons, printed as
`STRATA_SHUTDOWN={"reason": ...}`:

- `idle_timeout` — no API request arrived for `IDLE_MINUTES`. Raise the value, or
  send a request from your client to reset the timer.
- `serve_deadline` — `SERVE_MINUTES` elapsed.

If it raises instead, the server did not report `/v1/status` activity and
`SERVE_MINUTES` was `0`, so there was no way to detect idleness. Set
`SERVE_MINUTES` to a bounded window. The cell does not override Kaggle session
limits, quotas, network outages, or runtime shutdown.

## Quality gate fails or is very slow

The tool check asks the model to read, edit and test a local file, and it fails
loudly rather than reporting a false pass. Common causes:

- **Invalid tool call, or an edit before reading** — the model skipped
  `read_file`. That usually means the prompt or template is too small a model for
  agent use; try a larger variant.
- **Gave up within N turns** — the model kept answering in prose instead of
  calling tools. The same signal applies.
- **It takes minutes** — expected. A dual T4 prefills slowly, so a 12-turn task
  can take a while. Run it once per model variant, not on every session.

A `FAIL` here predicts an agent client that loops or stalls even though the
speed numbers look fine.

## OpenCode waits forever on the first token

The notebook's generated provider entry sets `headerTimeout` and `chunkTimeout`
to 15 minutes because a T4 can spend minutes loading or paging experts. If you
wrote your own config, raise those values; the defaults are far too short. Also
confirm `compatibility.reasoningField` is `reasoning_content`, otherwise thinking
tokens appear as answer text.

## Model restore progress is quiet

Copy-mode restoration reports byte-level progress, percentage, completed files, active copies, throughput, and elapsed time. If the line appears stationary briefly, a filesystem read or write may be blocked; wait for the next update unless an exception appears. Do not rerun the restore cell while it is actively copying.
