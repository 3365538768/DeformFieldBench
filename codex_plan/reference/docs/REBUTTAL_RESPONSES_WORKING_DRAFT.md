# Reviewer responses: evidence-gated working draft

Date: 2026-11-09. **Not submission-ready.** This draft separates verified source facts from missing experimental evidence. Do not submit bracketed placeholders or convert planned experiments into completed claims. Manuscript/table/checkpoint identity remains to be verified.

## Common opening

We thank the reviewers for identifying three different questions that our presentation did not sufficiently separate: whether the intermediate fields are predictable, whether a trained regressor uses them, and whether learning with field feedback improves material inference relative to a retrained, otherwise matched control. Inference-time masking addresses the second question; it does not establish the third. We will report each question separately and restrict our claims to the evidence actually supported.

**Before submission:** Replace forward-looking language with a precise statement of completed changes and results, or acknowledge an unresolved limitation. Do not imply that the following missing experiments have been completed.

## Reviewer a1nG: material learning and quantitative benefit

The manuscript available locally reports lower per-parameter median errors for the full system, but not uniformly better sample-wise error. Its unfiltered SPE median is 0.299 versus 0.304 for the supervised control, while its mean SPE is worse, 0.334 versus 0.314. We should therefore avoid a blanket claim of superiority. The mean/median difference warrants an explicit analysis of the error tail.

**Required evidence before final response:**

- Verify the manuscript version and reproduce these values from the associated predictions and checkpoints.
- Provide unfiltered predicted-versus-target plots with common axes, exact parameter transformations, and global/per-family constant controls learned only from training data.
- Report within-family association, prediction variance, per-parameter error, object-level uncertainty, and all failures. Do not use correlation across material families alone as evidence of continuous parameter recovery.
- Insert the matched C/D retraining result, including the effect size and its uncertainty. If independent feedback benefit is not established, state that limitation directly.

**Contribution statement after evidence review:** Describe aligned observations, field supervision and object-isolated evaluation as dataset capabilities. Claim a performance benefit for the architecture only for the specific control, metric and setting that supports it. Better field visualizations alone do not establish better material estimates.

## Reviewer AKYF: physical fidelity, external methods and exact feedback ablation

### Physical fidelity

The difference in rebound behavior indicates a substantial configuration/domain mismatch; it does not by itself establish that either simulator is physically correct. Same-generator replay measures agreement with that generator, not independent validation of its physical accuracy. We should not equate normalized projected stress response with calibrated volumetric stress or claim fracture recovery.

**Required evidence:** Original-generator free-fall/material-response/release or drop tests; physical-time-consistent damping settings; time-step and resolution studies; assumptions, reference solutions and failure regions. New FEM/engine fixture calibration cannot validate the original MPM data by substitution.

### Independent methods and real data

PIXIE and VoMP predict spatial material properties from representations of a 3D asset. OmniPhysGS optimizes constitutive representations using a text-conditioned generative video prior; its native objective differs from identifying parameters from a captured target video. These differences require explicit input and compute accounting rather than omitting the methods.

**Required evidence:** Completed results for available independent methods, checkpoint versions, initial geometry access, allowed observations, text information, adaptation data, supported output dimensions, total runtime and failures. Separate geometry-assisted comparisons from a strict RGB-only setting. Do not fill unsupported yield outputs with ground truth.

The local manuscript contains five real-sequence replay measurements, but these are behavior-level measurements without independently measured real material parameters. Their numerical existence does not resolve the request for adequate qualitative evidence, controls or absolute parameter validation.

**Required evidence:** Display input/predicted response on common visible regions; document geometry/action alignment and hand occlusion; compare under common conditions. Without independently measured parameters, explicitly limit the real-data claim to effective simulator behavior. A fit on the evaluated frames must not be described as held-out future prediction.

### Feedback-only ablation and numerical configuration

**Required result:** Retrain C with the same physics bottleneck, field heads, field supervision, preprocessing, stages, budget and selection rule as D, removing only the predicted-field inputs to the material head. Disclose any alternate feedback paths and preserve the bottleneck. Report multiple seeds and per-object errors. A stop-gradient operation alone is not a feedback-removal control.

**Required configuration receipt:** Exact frame indices/timestamps, views, backbone weights, parameter transformations, masks, optimizer and defaults resolved against the actual software version, stage lengths and trainable modules, coefficients, effective batch, precision, schedule and checkpoint-selection rule. Published default configuration values cannot be substituted for missing actual-run records.

## Reviewer Nxg5: regularizer assumptions and experiment interpretation

### Equations 19–21

The reviewer is correct that positive stress–motion correlation and higher response within actuator contact regions are not universal mechanical constraints. Elastic recovery can have high stress at low velocity, and stress concentration need not coincide with actuator contact. Calling these terms weak priors does not establish their validity across release and recovery phases.

For the pinned public implementation, channels are reduced using absolute-value averaging and normalization/correlation jointly aggregate the view, time and spatial axes per sample. This is not a per-time mechanical balance, and the magnitude of an encoded RGB response is not necessarily a physical stress or velocity magnitude.

**Unresolved implementation/manuscript discrepancy:** The published default configuration disables this regularizer (`use_phys_loss=false`, `lambda_phys=0`). This does not establish whether the reported manuscript experiment used it. Verify the checkpoint-associated runtime configuration before writing the final answer.

- If the actual main experiment did not use these terms, correct the training-objective description and identify the terms as unused candidates; do not invent an on/off result.
- If it did, report a matched retrained loss-off control and phase-stratified performance, especially release/recovery. Removing this regularizer is distinct from removing GT field supervision.

### Separation of evidence and capacity controls

Present three separate tables or clearly separated arguments: field prediction quality; fixed-model intervention/dependence; retrained controls for the benefit of feedback. Provide a retained/removed component matrix for each ablation. Count both nominal and actively trained parameters, and report actual updates and compute. Keeping unused parameters matches storage, not necessarily effective capacity.

### VLM reproducibility

The public launch wrapper defaults to `qwen3.5-plus`, while its registry includes other choices. It is not sufficient to identify the model used in the reported table. Recover the actual model/version, input frames, multi-view layout, prompts, adaptation status, numerical parsing, unit conversions, retry policy and invalid-output rate from sanitized request/output records.

If such records cannot be recovered, label a newly run baseline by its actual version and protocol rather than retrospectively assigning that identity to the old table. A zero-shot VLM versus task-supervised estimator comparison cannot establish the independent value of field feedback.

## Final evidence check

Before submission, every empirical sentence must link to a completed receipt, table or artifact. If matched retraining does not establish a practically useful improvement, report this immediately to the user, narrow the V1 claim, preserve the negative evidence and begin the authorized V2 construction. Continue the response with verified facts and limitations; do not delay notification until a replacement model succeeds.
