# Kaggle setup — full checklist

The quickest entry point is the **Open in Kaggle** badge at the top of the [repository README](../README.md). It imports the public notebook into Kaggle; it cannot grant access to private datasets or secrets.

## First run: create the datasets

Dataset creation runs on **CPU**: section 3c calls the installed `setup.py`'s
download and preparation helpers directly, so no NVIDIA check is involved. Use
`notebooks/Strata_Cache_Builder.ipynb`:

- [ ] Open `notebooks/Strata_Cache_Builder.ipynb` in Kaggle using **New Notebook → Load from URL** or import the file.
- [ ] Accelerator **None**, Internet on, `KAGGLE_API_TOKEN` secret enabled, `strata-runtime-cache` attached if it already exists.
- [ ] Run sections 1–3b. The probe prints `strata --help`, `strata launch --help` and the launch plan, and records the real file list and sizes.
- [ ] Set `CPU_PREPARE_MODEL = True` in section 3c and run it. Expect ~63.7 GB of downloads (58.5 GB model + 5.2 GB MTP tensors); the uploaded draft layer is ~1 GiB and the run is resumable.
- [ ] Confirm `models/`, `packs/` and `mtp/` have real bytes in `cache_status()`. If the cell reports a missing helper, switch the accelerator to **T4 ×2**, rerun the load/restore cells, and use section 4 as the fallback.
- [ ] Run section 5 in the same session that produced the files. Confirm `strata-runtime-cache`, `strata-mtp-rt` and `strata-model-<family>-<variant>` all show **Ready** before ending the session.

You can also run the launcher itself once on GPU: it launches Strata and its save cell now uploads the three datasets in order (runtime → draft layer → model).

The uploader sends model data only when the explicit license confirmation is true. The model, the draft layer and the runtime are three separate private datasets.

## Returning run: restore caches

- [ ] Start a fresh session using `notebooks/Strata_Kaggle_Launcher.ipynb`.
- [ ] Enable Internet/GPU and grant the notebook access to `KAGGLE_API_TOKEN`.
- [ ] Use **Add Input** to attach `strata-runtime-cache`, `strata-mtp-rt` and `strata-model-<family>-<variant>`. The cache names are printed by the notebook configuration cell.
- [ ] Keep the Kaggle username, model family and variant consistent with the attached cache names.
- [ ] Leave `MODEL_RESTORE_MODE = "symlink"` and `DRAFT_LAYER_DATASET = ""` (meaning your own `strata-mtp-rt`).
- [ ] Run the notebook from top to bottom.
- [ ] Confirm `Runtime cache restored: True` and `Model cache restored: True` before launching Strata.
- [ ] Confirm the runtime checksum, health check and authenticated public API test all pass.

## Why does Add Input show a download?

Attaching a dataset makes files available to a new Kaggle runtime. Kaggle may show a notification such as `Downloaded 14/16 files: ... GiB / ... GiB`; that's the input data being staged from your Kaggle Dataset storage into the session, not a download from the original model host.

The notebook defaults to `MODEL_RESTORE_MODE = "symlink"`: model, `packs/` and `mtp/` files are linked from `/kaggle/input`, so local disk holds only the extracted runtime (~1.5 GiB). Reads from the input mount may be slower; measure decode tokens/s with the quality gate. Set `"copy"` only when the free disk and restore time allow it — an ~80 GiB model usually does not fit.

Kaggle still stages the attached dataset into a new session in either mode.

No restore mode eliminates Kaggle's input staging or guarantees instant startup. The cache prevents the original-source model download when the attached files match the requested model; it does not turn the dataset into a permanently attached disk across all sessions.

## Manual account steps

Kaggle controls GPU selection, Internet access, Secrets, dataset visibility, and dataset sharing in its UI. The notebook does not create a secret for you or attach datasets on your behalf. Private datasets are only available to their owner and users with access granted by that owner.

## Six-hour monitoring

After launch and successful API verification, run the optional supervision cell. It checks the model health and tunnel/auth boundary for up to six hours. This does not override Kaggle's execution limits, quotas, inactivity behavior, or service failures.

In `copy` mode, restoration prints live progress (GiB, percentage, completed files, current copy count, throughput, and elapsed time) rather than waiting for several complete files before displaying an update.
