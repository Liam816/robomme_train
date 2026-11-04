# MUSA E2E performance profile, 2026-11-04

The original 80k job was stopped at update3403 after the user explicitly accepted rollback to the only saved E2E update3 checkpoint. Four-rank restoration from3 through7, checkpoint save/reload, fixed-input identity and frozen vision checks passed. The latest verified recovery point is:

/data/heping/robomme_musa/e2e/runs/ckpts/mme_vla_suite/percmem-v1-e2e-lerobot-s42/profile_resume_e2e_20261104/step_000007

Original train/model/E2E replay sources and experiment YAML are unchanged. The long job remains stopped. GPUs0-3 were not stopped; GPUs4-7 were released after all bounded diagnostics.

## Results

At global batch64, fixed real resident inputs, same four-rank FSDP/AdamW/EMA, one warmup plus two measured trials:

| micro/rank | accumulation | seconds/update | maximum allocated GiB across ranks |
|---:|---:|---:|---:|
|2|8|13.476|30.522|
|4|4|8.573|30.877|
|8|2|6.116|33.917|
|16|1|3.963|30.911|

End-to-end micro16 diagnostics (decode/augmentation/H2D included): warmup10.724s, then5.492/5.771s, slowest-rank mean5.632s; allocated30.930GiB, reserved38.432GiB. Original production last100 mean15.177s. This short sample is not a long-run guarantee or a strict paired throughput comparison.

Original micro2 trace:296 AllGather kernels,5.467s cumulative;19 ReduceScatter kernels,.789s. Aggregate GPU kernel times can overlap and include rendezvous waits. Serial full-history differentiable replay and rank history imbalance add cost. Scorer remains zero-gradient in the original run; performance profiling did not fix that issue.

## Tools and evidence

- scripts/profile_musa_e2e.sh launches a bounded copy of the original loop instrumented in memory. Set ROBOMME_PROFILE_DIR to a new output directory; normal train arguments follow.
- ROBOMME_PROFILE_TRACE=0 avoids another large Kineto trace.
- ROBOMME_PROFILE_ONLY=1 exits after timing without writing a new checkpoint; otherwise original checkpoint/reload validation runs.
- ROBOMME_PROFILE_ALLOW_MICRO_CHANGE=1 is diagnostic-only compatibility for micro/accum changes at global64/four ranks. All other resolved values must agree; train.py itself remains strict.
- scripts/benchmark_musa_batch.sh runs the fixed-input bounded sweep. Its default output directory is profiles/batch_sweep_20261104; use with care because it is a recorded one-off experiment.

Evidence roots under /data/heping/robomme_musa/e2e/profiles/: resume_e2e_20261104/, batch_sweep_20261104/, end_to_end_micro16_20261104/.

The uniform32_same_reader ablation changes selection and keeps this variant's reader; it is NOT a FrameSamp+Modul implementation or benchmark. The other server's native FrameSamp run was not reprofiled.
