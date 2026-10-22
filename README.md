# Velocity-Reweighting TensorBoard snapshots

This orphan branch contains point-in-time TensorBoard snapshots with image
summaries removed. It intentionally contains no training source, checkpoints,
ordinary logs, or generated images.

## Current snapshot

- This update refreshes the same three official FlowAWR replicas in place. `MANIFEST.tsv` still lists 103 stable runs: 26 Geneval, 75 PickScore and 2 OCR. The other 100 registered runs and every existing `logs/public/` artifact are retained byte-for-byte from remote commit `81e426bb234497dcf1ac93834be6df99e5f9a225`, including its two latest alpha TensorBoard updates. Every previously published source prefix for the refreshed runs is unchanged. No old run is removed.
- These runs use `Fuzm7/FlowAWR` main commit `5ae282ef34fd664d9828fe62817f1960534f1cce`, entry `scripts/train_awr_sd3.py`, configuration `config/awr.py:sd3_pickscore`, original SD3.5-Medium initialization, and seed 42. Each replica uses 2 nodes / 16 GPUs, 48 prompt groups of 24 images (1,152 images) and one actual optimizer update per round. They are separate runs, not continuations of the older Velocity-Reweighting AWR variants.
- Frozen event-file prefixes were read between `2026-10-22T03:28:45.334702+00:00` and `2026-10-22T03:28:45.366939+00:00` through a temporary zero-GPU reader with the result volume mounted read-only. The exported attempt remains `fresh-20261020T164111Z`; earlier interrupted setup attempts are not merged into these curves.
- At collection, all three training JobSets reported `Completed` and their training Pods were absent. Their transitions were at `2026-10-21T22:32:28Z` through `2026-10-21T22:32:31Z`. This is the Kubernetes lifecycle state, not a claim that the originally configured 100,000 training rounds finished. The last completed rounds agreed across all 16 rank audits in each run: replica 01 at 1273, replica 02 at 1276, and replica 03 at 1279. This export does not resume, stop, or change training.
- The refreshed snapshots contain 173,813 validated TFRecords and no image summaries or image-plugin values. All non-image values, original tag names and step numbers are preserved; the three outputs are byte-identical to their complete source prefixes. No training source, checkpoints, ordinary logs, Dockerfiles or occupation-workload results are added.
- Official scalar tags are `reward_pickscore` and `eval_reward_pickscore`, rather than the slash-separated reward tags used by older runs. The latest-train column is the latest sampling reward step; completed rounds are separately recorded under `audit/executed_optimizer_updates`. All exported audit values report 1,152 global samples, one executed update, and zero AMP-skipped updates per completed round.
- Runtime changes remain documented in shared storage at `/mnt/shared_storage/usr/kevinliu/flowawr/official-main-5ae282e-20261020/deployment-final-20261020/`. The manifest records the exact launcher and instrumented trainer hashes; the source revision identifies the upstream base, not an unmodified source tree. This results-only branch does not include those code/deployment files.

| Replica | Run key under `runs/pickscore/` | Latest train reward step | Latest eval step | Eval PickScore |
|---|---|---:|---:|---:|
| 01 | `sd35_pickscore_16gpu_official_flowawr_main-5ae282e_r01_seed42__fresh-20261020T164111Z` | 1273 | 1270 | 0.927611 |
| 02 | `sd35_pickscore_16gpu_official_flowawr_main-5ae282e_r02_seed42__fresh-20261020T164111Z` | 1275 | 1270 | 0.925785 |
| 03 | `sd35_pickscore_16gpu_official_flowawr_main-5ae282e_r03_seed42__fresh-20261020T164111Z` | 1278 | 1270 | 0.925081 |

## Historical catalog (100 existing runs, unchanged)

The following notes and table describe the earlier 2026-10-19 collection, not
current training status. They are retained for historical provenance; the
current export scope and base commit are stated above.

- `MANIFEST.tsv` lists 100 stable TensorBoard runs under `runs/`: 26 Geneval, 72 PickScore and 2 OCR. This update refreshes 5 existing runs; all other 95 registered runs and every existing `logs/public/` artifact are retained byte-for-byte from `c1c664aa8e80c8bdd5832251a28b58a7e47d080b`. No old run is removed.
- The changed master-c10ecf9 RTX two-sided clipping runs and master-3960907 H200 runs are refreshed in place. New resume shards are merged into their existing stable run directories; later shards replace overlapping tag/step values. Every previously published raw source prefix is unchanged, and train/eval steps do not regress. Historical stopped master/Clipping and devbox snapshots remain unchanged.
- At the process audit, 5 refreshed runs were training and 0 were interrupted with remaining pods running ResNet occupation. Interrupted runs: none. All available results, including those written after the previous resume, are collected. Status reflects actual training processes, not merely Pod Running status. This export does not restart or modify training jobs.
- The unchanged RTX two-sided clipping run with epsplus1.0 / epsminus0.5 has no remaining pods at this audit. Its earlier snapshot and resumed-attempt explanation are retained; its manifest status remains the historical status at its own snapshot time.
- Remote commit `c1c664aa8e80c8bdd5832251a28b58a7e47d080b` also contains updated external H200 samplewise/shared curves under `logs/public/`; those event files are preserved byte-for-byte, and their checksum entries are refreshed.
- Source prefixes were frozen between `2026-10-19T12:45:31.726554Z` and `2026-10-19T12:46:03.542607Z` through a temporary read-only, zero-GPU reader. Inodes, fixed-byte lengths and SHA256 hashes were checked before and after copying; only an incomplete trailing TFRecord, if present, is excluded at the snapshot boundary.
- Image summaries and image-plugin metadata are removed. All selected non-image scalar, tensor/text and other event values are preserved, including recorded NaNs and diagnostics. For resumed source shards, the later shard wins at overlapping tag/step pairs. These are immutable snapshots, not live training status.
- In `sd35_pickscore_4090_16gpu_two-sided-samplewise-clip_c1.0_epsplus1.0_epsminus0.5_kl0.0001`, the newest resume segment reached train step 223 / eval step 220 before interruption. Earlier-attempt history is retained, so the table still reports its historical maximum train/eval steps 238/230. These higher historical steps are not progress of the latest resumed attempt.
- Refreshed outputs are split only between complete TFRecords into files of at most 48 MiB. The `output_event`, `output_bytes` and `output_sha256` columns are semicolon-separated, aligned lists when a run has multiple output shards. TensorBoard reads all shards from the same stable run directory.
- The fresh master AWR run is stored as `sd35_pickscore_4090_16gpu_flowawr_kl1e-4__master-3960907`; it is not merged with the older AWR run of the same launcher name. Its original source name and path remain in the manifest. Existing external H20/H200 results under `logs/public/` are likewise kept separate from these cluster runs.
- The 21 previously excluded August directories and the old header-only failed 8-GPU startup have no changes since the previous collection and are not backfilled in this update. The previously published devbox snapshot is retained. No training source, checkpoints, images, ordinary logs, Dockerfiles or occupation-workload results are added.
- Registered outputs contain 13,734,665 validated records; 0 image values were removed from the 5 refreshed runs. Non-image event sequences were compared against the frozen source selection.

| Task | Run key | Latest train | Latest eval | Snapshot result |
|---|---|---:|---:|---|
| Geneval | `vr_noalpha_geneval_20260830_091741_PDT` | 199 | 200 | Geneval `0.675429` |
| PickScore | `sd35_pickscore_4090_16gpu_flowawr_kl1e-4` | 1542 | 1540 | PickScore `0.913639` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan12_square_positive_uniform_negative_scale20_kl1e-4` | 981 | 980 | PickScore `0.885621` |
| PickScore | `sd35_pickscore_nft_baseline-KL1e-4-beta1.0-fulltime_r1_A6000` | 2531 | 2530 | PickScore `0.919040` |
| PickScore | `sd35_pickscore_nft_ours-1singleloss-vpred-alpha-old-KL1e-4-beta1.0-fulltime_r1_A6000` | 2725 | 2720 | PickScore `0.916519` |
| PickScore | `sd35_pickscore_nft_ours-1singleloss-xpred-alpha-old-KL1e-4-beta1.0-fulltime_r1_A6000` | 3215 | 3210 | PickScore `0.920025` |
| PickScore | `sd35_pickscore_nft_ours-alpha-old-KL1e-4-beta1.0-fulltime_r1_A6000` | 2939 | 2930 | PickScore `0.920526` |
| Geneval | `sd35-geneval-nft-ours-1singleloss-alpha-old-kl1e-4-beta1-t10of10-r1-20260828` | 752 | 750 | Geneval `0.970136` |
| Geneval | `sd35-geneval-nft-ours-1singleloss-alpha-theta-kl1e-4-beta1-t10of10-r1-20260828` | 753 | 750 | Geneval `0.931205` |
| PickScore | `sd35-pickscore-nft-ours-1singleloss-alpha-old-kl1e-4-beta1-t10of10-r1-20260828` | 2351 | 2350 | PickScore `0.917213` |
| PickScore | `sd35-pickscore-nft-ours-1singleloss-alpha-theta-kl1e-4-beta1-t10of10-r1-20260828` | 2350 | 2350 | PickScore `0.918102` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan03_linear_scale20_kl1e-4` | 999 | 1000 | PickScore `0.915341` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan09_square_positive_scale5_kl1e-4` | 999 | 1000 | PickScore `0.909622` |
| PickScore | `sd35_pickscore_a6000_16gpu_step1_1_linear_scale15_kl1e-4` | 1522 | 1520 | PickScore `0.913242` |
| PickScore | `sd35_pickscore_a6000_16gpu_linear_kl1e-4` | 3955 | 3950 | PickScore `0.894875` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme1_kl1e-4` | 171 | 170 | PickScore `0.870452` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme2_scale10_kl1e-4` | 348 | 340 | PickScore `0.904479` |
| PickScore | `sd35_pickscore_4090_16gpu_flowawr_kl1e-4-r2-20260916` | 944 | 940 | PickScore `0.914866` |
| PickScore | `sd35_pickscore_4090_16gpu_flowawr_withDelta_kl1e-4` | 3885 | 3880 | PickScore `0.936144` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-1of2-kl1e-4-beta0p2-fulltime-r1-20260909` | 337 | 330 | Geneval `0.901079` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-1of2-kl1e-4-beta1-fulltime-r1-20260908` | 299 | 290 | Geneval `0.809615` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-1of3-kl1e-4-beta0p2-fulltime-r1-20260909` | 336 | 330 | Geneval `0.841614` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-1of3-kl1e-4-beta1-fulltime-r1-20260908` | 313 | 310 | Geneval `0.016112` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-2of3-kl1e-4-beta0p2-fulltime-r1-20260909` | 331 | 330 | Geneval `0.907074` |
| Geneval | `sd35-geneval-nft-ours-4selection-b-rho-2of3-kl1e-4-beta1-fulltime-r1-20260908` | 304 | 300 | Geneval `0.901416` |
| Geneval | `sd35-geneval-nft-ours-4selection-baseline-rho1-kl1e-4-beta1-fulltime-r1-20260908` | 633 | 630 | Geneval `0.970174` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-1of2-kl1e-4-beta0p2-fulltime-r1-20260909` | 330 | 330 | Geneval `0.913107` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-1of2-kl1e-4-beta1-fulltime-r1-20260908` | 298 | 290 | Geneval `0.936601` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-1of3-kl1e-4-beta0p2-fulltime-r1-20260909` | 337 | 330 | Geneval `0.925809` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-1of3-kl1e-4-beta1-fulltime-r1-20260908` | 292 | 290 | Geneval `0.946905` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-2of3-kl1e-4-beta0p2-fulltime-r1-20260909` | 333 | 330 | Geneval `0.906775` |
| Geneval | `sd35-geneval-nft-ours-4selection-c-rho-2of3-kl1e-4-beta1-fulltime-r1-20260908` | 299 | 290 | Geneval `0.946081` |
| Geneval | `sd35-geneval-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta5-fulltime-r1-20260914` | 425 | 420 | Geneval `0.004272` |
| Geneval | `sd35-geneval-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta5-fulltime-r1-20260914` | 424 | 420 | Geneval `0.004159` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-baseline-rho1-kl1e-4-beta2-fulltime-r1-20260910` | 685 | 680 | Geneval `0.852780` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-baseline-rho1-kl1e-4-beta5-fulltime-r1-20260910` | 735 | 730 | Geneval `0.005171` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of2-kl1e-4-beta1-fulltime-r1-20260910` | 679 | 680 | Geneval `0.944544` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of2-kl1e-4-beta2-fulltime-r1-20260910` | 676 | 670 | Geneval `0.925697` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of2-kl1e-4-beta5-fulltime-r1-20260910` | 743 | 740 | Geneval `0.000300` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of3-kl1e-4-beta1-fulltime-r1-20260910` | 667 | 660 | Geneval `0.976281` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of3-kl1e-4-beta2-fulltime-r1-20260910` | 685 | 680 | Geneval `0.416329` |
| Geneval | `sd35-geneval-nft-ours-4selection-tie-aware-c-rho-1of3-kl1e-4-beta5-fulltime-r1-20260910` | 720 | 720 | Geneval `0.001611` |
| OCR | `sd35-ocr-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta5-fulltime-r1-20260914` | 933 | 930 | OCR `0.000000` |
| OCR | `sd35-ocr-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta5-fulltime-r1-20260914` | 941 | 940 | OCR `0.000000` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-b-rho-1of3-kl1e-4-beta10-fulltime-r1-20260913` | 305 | 300 | PickScore `0.900755` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-b-rho-1of3-kl1e-4-beta5-fulltime-r1-20260913` | 304 | 300 | PickScore `0.907120` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-b-rho-2of3-kl1e-4-beta10-fulltime-r1-20260913` | 299 | 300 | PickScore `0.905427` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-b-rho-2of3-kl1e-4-beta5-fulltime-r1-20260913` | 301 | 300 | PickScore `0.907631` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta0.5-fulltime-r1-20260912` | 171 | 170 | PickScore `0.883837` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta1-fulltime-r1-20260912` | 505 | 500 | PickScore `0.898402` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta10-fulltime-r1-20260913` | 300 | 300 | PickScore `0.905272` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta10-fulltime-r2-20260916` | 854 | 850 | PickScore `0.919880` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta2-fulltime-r1-20260912` | 505 | 500 | PickScore `0.906349` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta5-fulltime-r1-20260913` | 632 | 630 | PickScore `0.910648` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-baseline-rho1-kl1e-4-beta7-fulltime-r1-20260914` | 1775 | 1770 | PickScore `0.924533` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of2-kl1e-4-beta10-fulltime-r2-20260916` | 873 | 870 | PickScore `0.927128` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of2-kl1e-4-beta5-fulltime-r1-20260914` | 1779 | 1770 | PickScore `0.929100` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of2-kl1e-4-beta7-fulltime-r1-20260914` | 1465 | 1460 | PickScore `0.921892` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta0.5-fulltime-r1-20260912` | 169 | 170 | PickScore `0.880363` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta1-fulltime-r1-20260912` | 501 | 500 | PickScore `0.900595` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta10-fulltime-r1-20260913` | 301 | 300 | PickScore `0.909569` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta2-fulltime-r1-20260912` | 505 | 500 | PickScore `0.908782` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta5-fulltime-r1-20260913` | 633 | 630 | PickScore `0.916012` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-1of3-kl1e-4-beta7-fulltime-r1-20260914` | 1771 | 1770 | PickScore `0.922526` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta0.5-fulltime-r1-20260912` | 170 | 170 | PickScore `0.884505` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta1-fulltime-r1-20260912` | 503 | 500 | PickScore `0.898843` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta10-fulltime-r1-20260913` | 302 | 300 | PickScore `0.903901` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta10-fulltime-r2-20260916` | 866 | 860 | PickScore `0.917800` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta2-fulltime-r1-20260912` | 501 | 500 | PickScore `0.909032` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta5-fulltime-r1-20260913` | 636 | 630 | PickScore `0.918958` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-2of3-kl1e-4-beta7-fulltime-r1-20260914` | 1759 | 1750 | PickScore `0.932365` |
| PickScore | `sd35-pickscore-nft-ours-4selection-soft-c-rho-3of4-kl1e-4-beta5-fulltime-r1-20260914` | 1764 | 1760 | PickScore `0.910899` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan01_linear_scale5_kl1e-4` | 2010 | 2010 | PickScore `0.916420` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan02_linear_scale15_resume_kl1e-4` | 2248 | 2240 | PickScore `0.923856` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan04_linear_scale30_kl1e-4` | 2009 | 2010 | PickScore `0.917894` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan07_linear_decay15to7p5_kl1e-4` | 2009 | 2000 | PickScore `0.916581` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan08_linear_decay15to5_kl1e-4` | 2999 | 3000 | PickScore `0.922747` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan10_square_positive_scale20_kl1e-4` | 2999 | 3000 | PickScore `0.876401` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan11_square_positive_uniform_negative_scale5_kl1e-4` | 1877 | 1870 | PickScore `0.867719` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan13_linear_fork420_scale5_kl1e-4` | 2999 | 3000 | PickScore `0.920414` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme2_kl1e-4` | 2969 | 2960 | PickScore `0.898373` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme2_scale15_kl1e-4` | 4500 | 4500 | PickScore `0.927734` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme3_kl1e-4` | 832 | 830 | PickScore `0.860954` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme4_kl1e-4` | 5097 | 5090 | PickScore `0.904450` |
| PickScore | `sd35_pickscore_4090_16gpu_flowawr_kl1e-4__master-3960907` | 60 | 60 | PickScore `0.884140` |
| PickScore | `sd35_pickscore_4090_16gpu_positive-shared-clip_c1.0_epsplus9.0_kl0.0001` | 70 | 70 | PickScore `0.885762` |
| PickScore | `sd35_pickscore_4090_16gpu_shared-clip-zerosum_c1.0_epsplus9.0_epsminus1.0_kl0.0001` | 12 | 10 | PickScore `0.848582` |
| PickScore | `sd35_pickscore_h200_8gpu_positive-samplewise-clip_c1.0_epsplus15.0_kl0.0001` | 1586 | 1580 | PickScore `0.839125` |
| PickScore | `sd35_pickscore_h200_8gpu_positive-shared-clip_c1.0_epsplus15.0_kl0.0001` | 1669 | 1660 | PickScore `0.857517` |
| PickScore | `sd35_pickscore_4090_16gpu_two-sided-samplewise-clip_c1.0_epsplus1.0_epsminus0.5_kl0.0001` | 238 | 230 | PickScore `0.905652` |
| PickScore | `sd35_pickscore_4090_16gpu_two-sided-samplewise-clip_c1.0_epsplus15.0_epsminus0.5_kl0.0001` | 253 | 250 | PickScore `0.904293` |
| PickScore | `sd35_pickscore_4090_16gpu_two-sided-samplewise-clip_c1.0_epsplus9.0_epsminus0.5_kl0.0001` | 423 | 420 | PickScore `0.909190` |
| PickScore | `sd35_pickscore_4090_16gpu_two-sided-samplewise-clip_c1.0_epsplus9.0_epsminus0.7_kl0.0001` | 422 | 420 | PickScore `0.913653` |
| PickScore | `sd35_pickscore_a6000_16gpu_shared_exp_clip1_kl1e-4` | 0 | 0 | PickScore `0.761281` |
| PickScore | `sd35_pickscore_a6000_16gpu_shared_exp_clip2_kl1e-4` | 0 | 0 | PickScore `0.761281` |
| PickScore | `sd35_pickscore_a6000_16gpu_shared_exp_clip3_kl1e-4` | 1 | 0 | PickScore `0.761281` |
| PickScore | `sd35_pickscore_a6000_16gpu_shared_exp_clip4_kl1e-4` | 1 | 0 | PickScore `0.761281` |
| PickScore | `sd35_pickscore_a6000_16gpu_shared_exp_clip6_kl1e-4` | 1 | 0 | PickScore `0.761281` |
| PickScore | `sd35_pickscore_a6000_16gpu_shared_exp_clip7_kl1e-4` | 58 | 50 | PickScore `0.878835` |
| PickScore | `sd35_pickscore_a6000_16gpu_shared_exp_clip8_kl1e-4` | 57 | 50 | PickScore `0.879731` |

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
