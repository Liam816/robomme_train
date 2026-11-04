# Step7 inference and FP16-intermediate gradient repair

The model and original E2E experiment YAML are unchanged. Real step8/rank0 batch16 FM loss at restored step7 weights is bitwise equal across all precision diagnostics. With unscaled FP16 storage, selected-memory gradients are entirely zero, although reader/condition outputs have nonzero gradients. A quantized-forward, FP32-gradient diagnostic oracle gives a maximum selected-memory gradient1.738e-11 and scorer norm3.549e-8.

A fixed backward loss scale2**20 recovers scorer norm3.553e-8 (relative L2 error.0022694, cosine.9999982 versus the same-forward oracle). Scale2**16 has17.95% relative gradient error and was not selected. FP16 feature storage, hard forward selection, full-history ST derivatives, FM loss, LR, optimizer/epsilon, initialization and EMA are retained. Gradients are divided back before clipping, logging and AdamW. Fixed scaling has no dynamic scaler state; its value is persisted in the resolved profile/checkpoint metadata. Non-finite norms still stop before optimizer.step.

Resume compatibility remains strict for model, optimizer, schedule, assets, dataset, source profile, global64 and four-rank layout. Only micro/accumulation, backward scale and overlay identity metadata may change.

## Inference measurements

One unsharded MTT S5000, restored step7, eval/inference_mode, inputs resident. Five warmups then20 timings for totals and a separate20 instrumented forwards for module attribution; instrumentation output identity checked. Histories32/41/80/142 with padding320: mean velocity latencies173.82/208.18/357.54/592.41ms. Batch1/history41: front37.91ms, wrist37.96, feature projection8.98, replay65.08 including scorer frame1.69 and9 context calls24.29, reader4.81, condition adapter.216, joint VLM/action18 layers52.30. Nested values overlap. Full10-denoising-step action sampling is681.60ms; vision/memory run once, transformer10 times. Real batch16 velocity is749.89ms. No decoding/H2D or FSDP communication is included.

Evidence: /data/heping/robomme_musa/e2e/profiles/inference20_gradrepair_20261104/{inference20.json,gradient_diagnostic.json,checkpoint_consolidation.json}. CPU gradient/optimizer/EMA equivalence and strict resume checks: /data/heping/robomme_musa/e2e/numerics_repair_tests.json.

## Resume launcher

Run scripts/launch_musa_e2e_resume.py --run-name UNIQUE_NAME from the overlay Python in the activated quicforge environment. It first restores step7 for a three-update four-rank preflight, requiring nonzero scorer gradient/parameter update plus checkpoint reload/frozen checks, then starts the full continuation from exactly step7 with micro16/accum1/global64 on physical4-7 until global80000.

To checkpoint and pause safely, create checkpoint_and_stop.request inside the active run directory. Alternatively send SIGUSR1 to rank0_pid recorded in its status.json. The request is reduced across ranks at the optimizer boundary; all ranks save model/AdamW/EMA/RNG/sampler state before exit. The supervisor records paused and does not restart. Do not send a terminating signal to torchrun when a fresh checkpoint is required.
