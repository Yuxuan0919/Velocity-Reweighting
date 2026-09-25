# Velocity-Reweighting TensorBoard snapshots

This orphan branch contains point-in-time TensorBoard snapshots with image
summaries removed. It intentionally contains no training source, checkpoints,
ordinary logs, or generated images.

## Current snapshot

- The branch contains 66 stable TensorBoard runs (26 Geneval, 38 PickScore, 2 OCR overall).
- The FlowAWR-withDelta and Step 1 A6000 PickScore runs were frozen at `2026-09-25T12:03:55Z`. Each active source was copied at a fixed byte boundary and verified against the live file by inode, size, and prefix SHA256.
- FlowAWR-withDelta consolidates its original shard with the checkpoint resume shard. The later shard wins 22 overlapping summary tag/step values at step 5, and one duplicate file-version record is removed.
- Step 1 A6000 was frozen from one active event file at train step 445 and eval step 440. Filtering removed 0 protobuf image values from the two runs; all retained scalar and tensor/text summaries were preserved.
- The four Alpha PickScore runs were frozen at `2026-09-21T05:31:06Z` from one active event file each. Every source was copied at a fixed byte boundary and verified against the live file by inode, size, and prefix SHA256.
- Filtering removed 0 protobuf image values from the four Alpha runs; all retained scalar and tensor/text summaries were preserved.
- The AWR experiment was frozen at `2026-09-18T06:35:04.141084Z` from two immutable event
  shards: the original segment through step 841 and its checkpoint-841 resume
  through step 1542. Both source files were copied only
  after exact byte-size and SHA256 verification.
- The two AWR source shards contain 52,784
  CRC/protobuf-valid records. At the resume boundary, the later shard replaces
  the original config and all 21 original scalar tag/step groups at step 841;
  the extra file-version record is also consolidated. The stable output has
  52,761 selected records, with the resume values winning
  exactly where the two trajectories overlap.
- Neither source contains image, audio, or histogram summaries, so every
  selected record is copied byte-for-byte. The canonical output contains
  52,759 scalar values and one config text tensor, and its
  selected scalar/tensor sequence is verified directly against the source
  records.
- Filtering is based only on protobuf image value/plugin type; all non-image
  scalar and tensor/text summaries are retained without tag-name filtering.
- TensorBoard loaded all 66 runs with zero image, audio, or histogram
  tags; the branch contains 1,278,676 validated records in total.
- For resumed AWR runs, semicolon-delimited `jobset`, `image_digest`,
  `code_revision`, launch, and trainer provenance fields in `MANIFEST.tsv`
  align one-for-one with the chronologically ordered `source_event` shards.

| Task | Run key | Latest train | Latest eval | Snapshot result |
|---|---|---:|---:|---|
| Geneval | `sd35-geneval-nft-ours-1singleloss-alpha-old-kl1e-4-beta1-t10of10-r1-20260828` | 721 | 720 | Geneval `0.967401` |
| Geneval | `sd35-geneval-nft-ours-1singleloss-alpha-theta-kl1e-4-beta1-t10of10-r1-20260828` | 722 | 720 | Geneval `0.943383` |
| Geneval | `vr_noalpha_geneval_20260830_091741_PDT` | 199 | 200 | Geneval `0.675429` |
| PickScore | `sd35-pickscore-nft-ours-1singleloss-alpha-old-kl1e-4-beta1-t10of10-r1-20260828` | 2255 | 2250 | PickScore `0.916593` |
| PickScore | `sd35-pickscore-nft-ours-1singleloss-alpha-theta-kl1e-4-beta1-t10of10-r1-20260828` | 2254 | 2250 | PickScore `0.917415` |
| Geneval | `sd35-geneval-nft-ours-4selection-baseline-rho1-kl1e-4-beta1-fulltime-r1-20260908` | 516 | 510 | Geneval `0.959982` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-1of2-kl1e-4-beta1-fulltime-r1-20260908` | 171 | 170 | Geneval `0.841727` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-1of3-kl1e-4-beta1-fulltime-r1-20260908` | 179 | 170 | Geneval `0.722047` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-2of3-kl1e-4-beta1-fulltime-r1-20260908` | 175 | 170 | Geneval `0.875262` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-1of2-kl1e-4-beta1-fulltime-r1-20260908` | 169 | 160 | Geneval `0.911796` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-1of3-kl1e-4-beta1-fulltime-r1-20260908` | 163 | 160 | Geneval `0.903814` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-2of3-kl1e-4-beta1-fulltime-r1-20260908` | 169 | 170 | Geneval `0.921463` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-1of2-kl1e-4-beta0p2-fulltime-r1-20260909` | 218 | 210 | Geneval `0.897482` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-1of3-kl1e-4-beta0p2-fulltime-r1-20260909` | 217 | 210 | Geneval `0.812350` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-2of3-kl1e-4-beta0p2-fulltime-r1-20260909` | 214 | 210 | Geneval `0.899430` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-1of2-kl1e-4-beta0p2-fulltime-r1-20260909` | 213 | 210 | Geneval `0.904114` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-1of3-kl1e-4-beta0p2-fulltime-r1-20260909` | 219 | 210 | Geneval `0.910484` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-2of3-kl1e-4-beta0p2-fulltime-r1-20260909` | 215 | 210 | Geneval `0.887215` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-baseline-rho1-kl1e-4-beta2-fulltime-r1-20260910` | 89 | 80 | Geneval `0.923186` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-baseline-rho1-kl1e-4-beta5-fulltime-r1-20260910` | 88 | 80 | Geneval `0.903365` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of2-kl1e-4-beta1-fulltime-r1-20260910` | 89 | 80 | Geneval `0.914269` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of2-kl1e-4-beta2-fulltime-r1-20260910` | 88 | 80 | Geneval `0.914119` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of2-kl1e-4-beta5-fulltime-r1-20260910` | 88 | 80 | Geneval `0.881183` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of3-kl1e-4-beta1-fulltime-r1-20260910` | 78 | 70 | Geneval `0.914306` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of3-kl1e-4-beta2-fulltime-r1-20260910` | 78 | 70 | Geneval `0.887740` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of3-kl1e-4-beta5-fulltime-r1-20260910` | 74 | 70 | Geneval `0.887702` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta0.5-fulltime-r1-20260912` | 124 | 120 | PickScore `0.880332` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta1-fulltime-r1-20260912` | 451 | 450 | PickScore `0.896793` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta2-fulltime-r1-20260912` | 450 | 450 | PickScore `0.905159` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta0.5-fulltime-r1-20260912` | 122 | 120 | PickScore `0.878230` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta1-fulltime-r1-20260912` | 447 | 440 | PickScore `0.899872` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta2-fulltime-r1-20260912` | 450 | 450 | PickScore `0.907561` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta0.5-fulltime-r1-20260912` | 123 | 120 | PickScore `0.881298` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta1-fulltime-r1-20260912` | 449 | 440 | PickScore `0.897855` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta2-fulltime-r1-20260912` | 448 | 440 | PickScore `0.907606` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta5-fulltime-r1-20260913` | 459 | 450 | PickScore `0.906916` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta5-fulltime-r1-20260913` | 459 | 460 | PickScore `0.911907` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta5-fulltime-r1-20260913` | 461 | 460 | PickScore `0.916062` |
| PickScore | `sd35_pickscore_4090_16gpu_flowawr_kl1e-4` | 1542 | 1540 | PickScore `0.913639` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-b-rho-1of3-kl1e-4-beta5-fulltime-r1-20260913` | 129 | 130 | PickScore `0.895376` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-b-rho-2of3-kl1e-4-beta5-fulltime-r1-20260913` | 129 | 120 | PickScore `0.896838` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-b-rho-1of3-kl1e-4-beta10-fulltime-r1-20260913` | 130 | 130 | PickScore `0.892454` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-b-rho-2of3-kl1e-4-beta10-fulltime-r1-20260913` | 127 | 120 | PickScore `0.895148` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta10-fulltime-r1-20260913` | 127 | 120 | PickScore `0.896735` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta10-fulltime-r1-20260913` | 128 | 120 | PickScore `0.895350` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta10-fulltime-r1-20260913` | 127 | 120 | PickScore `0.899011` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta7-fulltime-r1-20260914` | 1430 | 1430 | PickScore `0.923344` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of2-kl1e-4-beta7-fulltime-r1-20260914` | 1434 | 1430 | PickScore `0.919792` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta7-fulltime-r1-20260914` | 1433 | 1430 | PickScore `0.925796` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta7-fulltime-r1-20260914` | 1429 | 1420 | PickScore `0.926732` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of2-kl1e-4-beta5-fulltime-r1-20260914` | 1434 | 1430 | PickScore `0.930849` |
| Geneval | `sd35-geneval-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta5-fulltime-r1-20260914` | 332 | 330 | Geneval `0.007232` |
| Geneval | `sd35-geneval-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta5-fulltime-r1-20260914` | 333 | 330 | Geneval `0.013714` |
| OCR | `sd35-ocr-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta5-fulltime-r1-20260914` | 739 | 730 | OCR `0.000000` |
| OCR | `sd35-ocr-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta5-fulltime-r1-20260914` | 746 | 740 | OCR `0.000000` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-3of4-kl1e-4-beta5-fulltime-r1-20260914` | 1436 | 1430 | PickScore `0.922498` |
| PickScore | `sd35_pickscore_4090_16gpu_flowawr_kl1e-4-r2-20260916` | 898 | 890 | PickScore `0.912943` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of2-kl1e-4-beta10-fulltime-r2-20260916` | 526 | 520 | PickScore `0.916346` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta10-fulltime-r2-20260916` | 522 | 520 | PickScore `0.916176` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta10-fulltime-r2-20260916` | 523 | 520 | PickScore `0.914760` |
| PickScore | `sd35_pickscore_nft_ours-alpha-old-KL1e-4-beta1.0-fulltime_r1_A6000` | 1015 | 1010 | PickScore `0.904726` |
| PickScore | `sd35_pickscore_nft_ours-1singleloss-vpred-alpha-old-KL1e-4-beta1.0-fulltime_r1_A6000` | 1026 | 1020 | PickScore `0.905094` |
| PickScore | `sd35_pickscore_nft_ours-1singleloss-xpred-alpha-old-KL1e-4-beta1.0-fulltime_r1_A6000` | 993 | 990 | PickScore `0.904936` |
| PickScore | `sd35_pickscore_nft_baseline-KL1e-4-beta1.0-fulltime_r1_A6000` | 881 | 880 | PickScore `0.904326` |
| PickScore | `sd35_pickscore_4090_16gpu_flowawr_withDelta_kl1e-4` | 484 | 480 | PickScore `0.911092` |
| PickScore | `sd35_pickscore_a6000_16gpu_linear_kl1e-4` | 445 | 440 | PickScore `0.878433` |

See `MANIFEST.tsv` for exact source paths and frozen byte boundaries, hashes,
launch-script provenance, and filtering statistics. For consolidated runs, the
`source_event` field lists every source shard in chronological order.

## View

```bash
git clone --depth 1 --single-branch --branch tensorboard-results \
  git@github-new:Yuxuan0919/Velocity-Reweighting.git velocity-reweighting-tb
cd velocity-reweighting-tb
sha256sum -c SHA256SUMS
python3 -m pip install -r requirements.txt
python3 -m tensorboard.main --logdir runs --host 0.0.0.0 --port 6006
```

Then open `http://127.0.0.1:6006`, or use the devbox port-forwarded URL.

## Update semantics

The stable key is `runs/<task>/<run_key>/`. Publishing a newer snapshot of an
existing run replaces that entire directory after validation, so stale rolled
event shards cannot remain. Runs not selected for an update are retained.

This branch is maintained as a single orphan snapshot commit. Publishers update
it with an explicit `--force-with-lease` against the previously observed remote
SHA; bare `--force` is not used. Existing checkouts should refresh with:

```bash
git fetch --depth=1 origin \
  +refs/heads/tensorboard-results:refs/remotes/origin/tensorboard-results
git reset --hard origin/tensorboard-results
```

The replacement policy keeps growing binary event history out of the main Git
repository while making the branch head an atomic, self-contained snapshot.
