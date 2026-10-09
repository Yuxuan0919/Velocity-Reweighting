# Velocity-Reweighting TensorBoard snapshots

This orphan branch contains point-in-time TensorBoard snapshots with image
summaries removed. It intentionally contains no training source, checkpoints,
ordinary logs, or generated images.

## Current snapshot

- `MANIFEST.tsv` lists 84 stable TensorBoard runs under `runs/` (26 Geneval, 56 PickScore, 2 OCR overall). Previously published `logs/public/` snapshots are retained unchanged and are not included in these registry counts.
- This update refreshes 12 existing runs with previously unpublished records. The full shared-storage catalog was compared against the published source paths, byte counts and SHA256 hashes; stopped runs were included. All other registered runs and historical public artifacts are unchanged.
- The 38 chronological source shards were frozen between `2026-10-09T07:34:11.238839Z` and `2026-10-09T07:34:37.976622Z`. Fixed-byte prefixes were verified by inode, size and SHA256 before and after copying through a temporary, read-only, zero-GPU reader. Incomplete trailing records, if any, are excluded only at the snapshot boundary.
- Refreshed Linear/Square runs: Plan01, Plan02, Plan04, Plan07, Plan08, Plan10, Plan11 and Plan13. Plan03 and Plan09 have no new source data. Step2 Scheme2, Scheme2 scale15, Scheme3 and Scheme4 also have new event records; unchanged maximum steps do not by themselves imply unchanged event content. The result table below reports the validated latest steps, not a live job status.
- Plan02 remains its own resumed c=15 run. Plan13 remains a separate c=5 fork of checkpoint-420; neither run is merged into its parent. Plan01–04/07–12 use revision `58c5e07c2ec9db3e6d93a1efd084b8cacfd738f1`; Plan13 uses `b5e774c0faa624afabccdc0fd120bcd458667edd`.
- Plan12 has no new formal TensorBoard records, so its previous numerical-failure snapshot is unchanged, including recorded NaNs and diagnostics. Matrix multiplication and temporary synthetic training are excluded from experiment results.
- Filtering removed 0 protobuf image values from refreshed source records. Selected non-image scalar and tensor/text values are preserved. Where resumed shards overlap at the same tag and step, the later shard wins. Existing run directories are replaced in full so obsolete event shards do not remain.
- Previously published H200 Step2 scale-5/scale-20 snapshots under `logs/public/experiment_plans/` are retained byte-for-byte from the previous remote snapshot. Historical runs outside the published registry have no new data since the prior update and are not newly backfilled. The legacy devbox Geneval source is no longer present locally; its published snapshot is preserved.
- Deleted deployment provenance remains unavailable for `kevinliu-vs4psc12b7r600-260917-111311`. Verified source paths and checksums are retained; missing code/image provenance is explicitly marked `unavailable`.
- TensorBoard loads all 84 registered runs with zero image, audio or histogram tags; `runs/` contains 4,273,766 validated records. Semicolon-delimited provenance fields align with chronological `source_event` shards.
- Statuses are point-in-time snapshots, not a live job listing. Rewards and diagnostics are preserved regardless of whether an experiment was healthy or stopped. No training source, checkpoints, ordinary logs or container files are added.

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
| PickScore | `sd35_pickscore_a6000_16gpu_plan01_linear_scale5_kl1e-4` | 2010 | 2010 | PickScore `0.916420` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan02_linear_scale15_resume_kl1e-4` | 2248 | 2240 | PickScore `0.923856` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan03_linear_scale20_kl1e-4` | 999 | 1000 | PickScore `0.915341` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan04_linear_scale30_kl1e-4` | 2009 | 2010 | PickScore `0.917894` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan07_linear_decay15to7p5_kl1e-4` | 2009 | 2000 | PickScore `0.916581` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan08_linear_decay15to5_kl1e-4` | 2999 | 3000 | PickScore `0.922747` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan09_square_positive_scale5_kl1e-4` | 999 | 1000 | PickScore `0.909622` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan10_square_positive_scale20_kl1e-4` | 2999 | 3000 | PickScore `0.876401` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan11_square_positive_uniform_negative_scale5_kl1e-4` | 1877 | 1870 | PickScore `0.867719` |
| PickScore | `sd35_pickscore_a6000_16gpu_plan13_linear_fork420_scale5_kl1e-4` | 2999 | 3000 | PickScore `0.920414` |
| PickScore | `sd35_pickscore_a6000_16gpu_step1_1_linear_scale15_kl1e-4` | 1522 | 1520 | PickScore `0.913242` |
| PickScore | `sd35_pickscore_a6000_16gpu_linear_kl1e-4` | 3955 | 3950 | PickScore `0.894875` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme1_kl1e-4` | 171 | 170 | PickScore `0.870452` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme2_kl1e-4` | 2969 | 2960 | PickScore `0.898373` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme2_scale10_kl1e-4` | 348 | 340 | PickScore `0.904479` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme2_scale15_kl1e-4` | 4500 | 4500 | PickScore `0.927734` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme3_kl1e-4` | 832 | 830 | PickScore `0.860954` |
| PickScore | `sd35_pickscore_a6000_16gpu_step2_scheme4_kl1e-4` | 5097 | 5090 | PickScore `0.904450` |
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
