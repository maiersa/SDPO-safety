# Project Report Experiment Log

This document is the working memory for the project report. Its job is to collect the experiments, hypotheses, evidence, and open gaps before writing the introduction and results chapters.

## Core Question

The project asks:

> When does OPSD start to work?

This mirrors the question in `RL Excursions during Pretraining`, where the authors study when direct GRPO begins improving reasoning benchmarks during pretraining. Their key observation is that on-policy RL can become useful surprisingly early, especially on easier reasoning benchmarks such as GSM8K, while harder benchmarks such as MATH expose more brittleness and a stronger dependence on model capability.

Reference:

- `RL Excursions during Pretraining: How Early Is Too Early for On-policy Learning?`: https://rl-excursions.github.io/

Our version of the question studies OPSD/SDPO on OLMo 3 checkpoints, where many intermediate checkpoints are available across training stages. The central hypothesis is that OPSD becomes useful only after the base policy crosses a minimum capability threshold: too early, the model does not yet produce enough useful reasoning structure for the OPSD signal to reinforce; after the threshold, the same objective can become net-positive.

## Thesis-Shaped Claim

Draft thesis statement:

> OPSD begins to improve reasoning only after the base model reaches a task-dependent capability threshold, appearing earlier on GSM8K-like problems than on harder mathematical reasoning tasks, and the same mechanism may also provide a useful pathway for safety alignment when privileged feedback is safety-oriented.

This should be treated as a draft until the final benchmark tables are filled in.

## Main Experiment Families

### 1. Early OPSD on Math Reasoning

Purpose:

- Test whether OPSD improves reasoning benchmarks when applied to intermediate OLMo 3 checkpoints.
- Compare the first useful checkpoint across datasets.
- Use GRPO as a reference point inspired by `RL Excursions during Pretraining`.

Benchmarks:

- `GSM8K`
- `MATH`
- `MATH-500` / `math500` where relevant
- `OpenMathInstruct` for training in RL-Excursions-style runs

Primary metric:

- `delta = metric(post-train) - metric(base checkpoint)`

Report separately:

- `pass@1`
- `pass@8`
- `pass@16` or `pass@32`, depending on the available evaluation run
- Validation mean/best metrics only when clearly labeled as training validation rather than benchmark pass@k

Key files:

- `experiments/stage1/README.md`
- `experiments/stage1/lab_presentation_draft.md`
- `PRETRAIN_OPSD_COMMANDS.txt`
- `docs/pretrain_benchmark_eval.md`
- `experiments/pretrain/run_opsd_sequential_sweep.sh`
- `experiments/pretrain/run_grpo_sequential_sweep.sh`
- `experiments/pretrain/run_pretrain_benchmark_eval.sh`
- `scripts/eval_pretrain_benchmarks.py`
- `scripts/plot_pretrain_benchmarks.py`

### 2. OLMo 3 Checkpoint Sweep

Purpose:

- Use OLMo 3's checkpoint availability to ask where OPSD becomes net-positive during training.
- Separate base capability from post-training improvement.

Checkpoint families observed in the current experiment commands:

- Stage 1:
  - `stage1-step500000`
  - `stage1-step656000`
  - `stage1-step1413814`
  - Earlier planned grid: `stage1-step1000`, `stage1-step4000`, `stage1-step16000`, `stage1-step64000`, `stage1-step128000`, `stage1-step256000`, `stage1-step656000`
- Stage 2:
  - `stage2-step16000`
  - `stage2-step32000`
  - `stage2-step47684`
- Stage 3:
  - `stage3-step1000`
  - `stage3-step4000`
  - `stage3-step8000`
  - `main`

Important design choice:

- Materialize checkpoints as local model directories and point `actor_rollout_ref.model.path` to those directories, since the trainer path does not expose a clean Hugging Face `revision` field.

### 3. OPSD Gradient-Alignment Diagnostic

Purpose:

- Diagnose whether OPSD's privileged-context distillation gradient aligns with an empirical reward-improving gradient.
- This helps explain not only whether OPSD works, but why it starts working at a particular point.

Core question:

> Across OLMo 3 7B training stages, does the OPSD gradient increasingly align with the ideal gradient that would improve final-answer correctness?

Main quantities:

- `g_ideal`: empirical reward-improving gradient estimated by branching on candidate next tokens and measuring downstream success.
- `g_opsd`: distillation gradient induced by a privileged-context teacher.
- Alignment metric: cosine similarity between `g_ideal` and `g_opsd`.

Teacher variants:

- No privileged context control.
- Final-answer privileged context.
- Full-solution privileged context.

Key files:

- `gradient_alignment_opsd_experiment.md`
- `opsd_alignment/configs/main_experiment.yaml`
- `opsd_alignment/configs/smoke_test.yaml`
- `opsd_alignment/scripts/compute_gradients_and_alignment.py`
- `opsd_alignment/scripts/compute_teacher_student_distributions.py`
- `opsd_alignment/scripts/generate_student_rollouts.py`
- `opsd_alignment/scripts/select_nodes.py`
- `opsd_alignment/scripts/plot_results.py`

### 4. BASETEN / Constitutional Safety Experiments

Purpose:

- Explore whether OPSD-style dense distillation can improve safety behavior, inspired by Baseten-style constitutional alignment experiments.
- Compare sparse GRPO with an external constitutional judge against SDPO/OPSD with a constitutional teacher.

Dataset:

- `BeaverTails` safety subset where available.
- Fallback data paths used for plumbing tests are noted in the quickstart.

Safety setup:

- Constitution file: `data/constitution.txt`
- BeaverTails preparation: `scripts/prepare_beavertails.sh`
- GRPO judge path: `verl/utils/reward_score/feedback/constitution_judge.py`
- Constitutional teacher path: `verl/utils/reward_score/feedback/constitution_teacher.py`

Key files:

- `BASETEN_IMPLEMENTATION_SUMMARY.md`
- `BASETEN_CONSTITUTIONAL_QUICKSTART.md`
- `BASETEN_TRAINING_COMMANDS.txt`
- `verl/trainer/config/baseline_grpo_constitutional.yaml`
- `verl/trainer/config/sdpo_constitutional.yaml`
- `verl/trainer/config/sdpo_constitutional_offpolicy.yaml`
- `run_runai_grpo_constitutional.sh`
- `run_runai_sdpo_constitutional.sh`
- `run_local_grpo_constitutional.sh`
- `run_local_sdpo_constitutional.sh`

## Current Training Recipes To Record

### OPSD Math Teacher

Common settings from the current commands:

- Config: `sdpo_math_teacher`
- Rollout source: `student`
- Learning rates tested: `1e-5`, `5e-6`, `1e-6`
- Alpha values tested: `0.5`, `0.0`
- Distillation top-k:
  - `100` in earlier/smoke runs
  - `null` for full-vocabulary distillation
- Teacher update rate:
  - `0.0` frozen teacher
  - `0.2` lightly moving EMA teacher
- Pointwise KL clip:
  - `0.05` in later prompt-fix runs
- Typical training steps:
  - `100` for OpenMathInstruct sweeps
  - `150` for earlier GSM8K sweep plan

### GRPO Reference

Common settings from the current commands:

- Config: `baseline_grpo`
- Learning rate: `1e-5`
- Rollout samples per prompt: `8`
- Total training steps: `150`
- Used as a reference to compare whether OPSD turns on earlier, later, or similarly to direct RL.

### Benchmark Evaluation

For RL-Excursions-style comparison:

- Base checkpoints:
  - `prompt-mode=base`
  - usually 8-shot
- Trained checkpoints:
  - `prompt-mode=trained`
  - usually 0-shot
- Prompt style:
  - `rlx`
- Sampling:
  - `temperature=0.6`
  - `top_p=1.0`
  - `num_samples=32` where feasible
- Metrics:
  - `pass@1`
  - `pass@8`
  - `pass@32`

Important caveat:

- Training validation and benchmark evaluation are not identical. Training validation may use OpenMathInstruct rows, chat-style prompts, boxed-answer reward parsing, and mean/best group metrics. Benchmark evaluation uses held-out benchmark parquet files, RLX-style prompts, answer extraction, and pass@k from the generated sample pool.

## Result Table Template

Fill this in as soon as the final logs/summary CSVs are selected.

| Experiment | Checkpoint | Dataset | Method | Seed | Base metric | Post-train metric | Delta | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base eval | `stage1-step...` | `GSM8K` | Base | 0 | TODO | N/A | N/A | TODO |
| OPSD sweep | `stage1-step...` | `GSM8K` | OPSD | 0 | TODO | TODO | TODO | TODO |
| OPSD sweep | `stage2-step16000` | `GSM8K`/`MATH` | OPSD | 0 | TODO | TODO | TODO | promptfix/EMA0/full-vocab |
| OPSD sweep | `stage2-step32000` | `GSM8K`/`MATH` | OPSD | 0 | TODO | TODO | TODO | pointwise KL clip 0.05 |
| OPSD sweep | `stage2-step47684` | `GSM8K`/`MATH` | OPSD | 0 | TODO | TODO | TODO | pointwise KL clip 0.05 |
| OPSD sweep | `stage3-step1000` | `GSM8K`/`MATH` | OPSD | 0 | TODO | TODO | TODO | stage3 promptfix |
| OPSD sweep | `stage3-step4000` | `GSM8K`/`MATH` | OPSD | 0 | TODO | TODO | TODO | stage3 promptfix |
| OPSD sweep | `stage3-step8000` | `GSM8K`/`MATH` | OPSD | 0 | TODO | TODO | TODO | stage3 promptfix |
| OPSD sweep | `main` | `GSM8K`/`MATH` | OPSD | 0 | TODO | TODO | TODO | final stage3 anchor |
| GRPO reference | `stage...` | `GSM8K`/`MATH` | GRPO | 0 | TODO | TODO | TODO | TODO |
| Safety | TODO | BeaverTails | GRPO constitutional | 0 | TODO | TODO | TODO | external judge |
| Safety | TODO | BeaverTails | SDPO constitutional | 0 | TODO | TODO | TODO | constitutional teacher |

## Introduction Outline

The introduction can follow this structure.

1. Setting:
   Large language models increasingly acquire reasoning and alignment behavior through post-training, but most post-training is treated as a late-stage step after pretraining has already produced a capable base model.

2. Main challenge:
   It is unclear when an on-policy or self-distillation objective has enough signal to improve a model. Early checkpoints may be too weak to solve tasks, but waiting until the end may miss chances to shape the training trajectory earlier.

3. Why related work is insufficient:
   RL Excursions shows that GRPO can help surprisingly early, but it focuses on sparse RL rather than OPSD. Standard reasoning post-training studies mostly use final base models. Safety alignment work often evaluates end-state improvements rather than the training-stage threshold where the method first becomes useful.

4. Approach:
   Use OLMo 3 because it provides many checkpoints across training stages. Apply OPSD/SDPO and GRPO-style comparisons to reasoning benchmarks, especially GSM8K and MATH, and measure improvement relative to the same base checkpoint.

5. Why this approach is needed:
   Measuring deltas over the base checkpoint isolates post-training improvement from raw pretraining progress. Comparing datasets tests whether the OPSD threshold is task-dependent. Comparing against GRPO links the project to the early-RL finding while testing a denser teacher-guided objective.

6. Thesis statement:
   OPSD starts to work only after the model crosses a task-dependent capability threshold, with easier math reasoning tasks showing earlier gains than harder ones, and this threshold view also helps organize preliminary safety-alignment experiments.

7. Results paragraph:
   Summarize the final observed threshold, the main GSM8K/MATH contrast, the GRPO comparison, and any safety trend once the result table is filled.

8. Contributions:
   The report contributes an OLMo 3 checkpoint-sweep framing for OPSD, a benchmark/evaluation setup for measuring deltas over base checkpoints, diagnostic evidence about privileged teacher alignment where available, and a preliminary extension from math reasoning to constitutional safety alignment.


### Personal Scratch Pad

We ask when does OPSD start to help the model learn math reasoning problems. We had the idea that GRPO is seen to be pretty useful even very early on in the training on math. So naturally, we looked into OPSD/SDPO as an alternative to GRPO as it has the extra benefit of having dense rewards, since it is on the token level, and that it is on-policy which has its own. benefits. 

We also was trying to figure out at what point the model's policy becomes strong enough to understand its Privileged Information (PI) and assign correct rewards to the tokens. 

We really focused on the math reasoning part and less on the safety part, since it seemed easier to come up with a cool result that is a direct comparison to the rl excursion post.

We went through several Olmo 3 checkpoints in the hopes to see great improvements to math reasoning compared to the base model and the GRPO trained model. This has led to an underwhelming conclusion that math is too hard to learn through OPSD for several different reasons that have come apparant through some new research papers. 

We first looked into which hyper parameters seem to perform the best. We set ourselves on the following hyper params:

1) JSD/ Forward / Reverse KL Alpha value: 0.5 -> JSD
2) EMA: 0
3) distillation top k: null -> full 
4) pointwise kl clip: 0.05

We can see the plots showing base vs OPSD vs GRPO. GRPO constantly improves the model, while the OPSD seems to follow the the base model. It does not degrade the model too much or does not improve it much neither. 

I looked into the Apple's gradient alignment methods to see if we could see if the gradient would help the model or not. There are a couple results, but we could probably run slightly more. We can decide on this to really finalize the project.

Notes for the introduction:
It feels likes we are not introducing the project well enough with some concrete paper citations and problem motivations. I would like to look more into the problem motivation and maybe give me some things to look up on internet to be able to mention in the paper. 

The question is not whether OPSD can help a sufficiently capable model.
The question is when the model becomes capable enough for OPSD's privileged signal to mean something.

RL Excursions asks when sparse on-policy reinforcement learning starts helping.
We ask when dense, privileged-information self-distillation starts helping.
If GRPO can work early because reward selects good sampled trajectories, OPSD may work early or fail early depending on whether the model can use the privileged answer/context to produce a useful token-level teacher distribution.

To give a bit of a comment on the last parts of the introduciton, we don't see a meaningful delta over the base on math reasoning for OPSD, since it has been showed that the objective of the answer given as privileged information is not a set in stone objective for the model to really learn, then the model is also taught to be more confident in its answers, but math requires some sort of unknown for the model to explore. I can cite a couple papers. 

I like the contributions that are already in the introduction, but can we talk about the gradient direction part? 


### Draft

```latex
%%%%%%%%%%%%%%%%%%%%%%
\chapter{Introduction}
%%%%%%%%%%%%%%%%%%%%%%

Post-training has become a central ingredient in building reasoning-capable language models. In the standard pipeline, a base model is pretrained first and then improved with instruction tuning, reinforcement learning, preference optimization, or other post-training objectives. For mathematical reasoning, reinforcement learning with verifiable rewards (RLVR) is especially attractive because final answers can be checked automatically. Recent work complicates the usual view of post-training as a final polishing step: \emph{RL Excursions during Pretraining} shows that GRPO can improve reasoning even when applied to intermediate pretraining checkpoints \cite{bansal2026rlexcursions,shao2024deepseekmath}. This suggests that post-training objectives may not only refine a finished model; they may also interact with the trajectory by which reasoning behavior emerges.

On-policy self-distillation (OPSD) has recently emerged as a promising alternative to reinforcement learning with verifiable rewards (RLVR) for post-training reasoning models \cite{hubotter2026sdpo}. RLVR-style methods such as GRPO score a completed rollout with a sparse verifiable reward, while OPSD aims to turn richer feedback or privileged context into token-level training signal along the sampled trajectory. This direction is also practically relevant: recent model reports and on-policy distillation (OPD) analyses describe on-policy distillation as part of post-training or capability-consolidation pipelines for models such as Qwen3, DeepSeek-V4, and MiMo-style systems \cite{qwen3technical,deepseekv4,xiao2026mimo,rethinkingopd}. In this project, we study OPSD on mathematical reasoning, where the privileged context is the correct answer or a reference solution. The central question is whether this privileged signal is meaningful at every training stage. A checkpoint that cannot use the privileged information well may produce teacher distributions that are confident without being reward-improving, so this project asks: when does OPSD start to help?

Mathematical reasoning provides a natural testbed for this question because correctness can be evaluated automatically while the learning problem remains nontrivial. Benchmarks such as GSM8K and MATH require multi-step reasoning with verifiable final answers \cite{cobbe2021training,hendrycks2021measuring}. However, a final answer does not uniquely determine a correct reasoning path. This creates a potential mismatch for OPSD: privileged answer information may sharpen the teacher distribution without necessarily producing token-level updates that improve final-answer correctness.

We investigate this issue using OLMo 3 checkpoints, which allow us to apply the same post-training procedure at different points in the model's training trajectory \cite{olmo3}. For each checkpoint, we evaluate OPSD relative to the corresponding base model, using delta over base as the primary readout. This separates improvements due to pretraining progress from improvements due to the post-training objective itself. We evaluate on GSM8K and MATH-style benchmarks and compare against GRPO-trained references, following the early-on-policy-learning motivation of RL Excursions.

Our experiments suggest that OPSD does not yield meaningful gains over the base model in this math-reasoning setting. Across the evaluated checkpoints, GRPO improves more consistently, while OPSD largely tracks the base checkpoint. Hyperparameter sweeps identify a relatively stable configuration using JSD-style mixing with \(\alpha=0.5\), a frozen teacher, full-vocabulary distillation, and pointwise KL clipping, but this does not close the gap to GRPO. These results indicate that dense privileged supervision is not automatically sufficient for mathematical reasoning, even when the same setting admits gains from sparse on-policy RL.

To better understand this outcome, we complement benchmark evaluation with gradient-direction diagnostics. Inspired by gradient-alignment analyses, we compare the OPSD distillation gradient with an empirical reward-improving direction estimated by branching on candidate next tokens and measuring downstream success \cite{armandpour2026unmaskingonpolicydistillationhelps}. This diagnostic directly probes whether the privileged teacher distribution points in directions that the task reward would endorse. It also reframes the central question: OPSD should help only when privileged information induces token-level gradients aligned with reward-improving updates.

The thesis of this project is therefore that OPSD is a promising but capability- and task-dependent alternative to RLVR: in mathematical reasoning, its effectiveness depends on whether privileged context yields reward-aligned token-level supervision, and in our OLMo 3 experiments this condition appears insufficiently satisfied.

Our contributions are: (1) a checkpoint-sweep evaluation of OPSD on OLMo 3 mathematical reasoning benchmarks; (2) a delta-over-base methodology for comparing OPSD to both base checkpoints and GRPO-trained references; (3) a hyperparameter study of the OPSD configuration used in our runs; (4) gradient-direction diagnostics for analyzing whether privileged teacher updates align with reward-improving token directions; and (5) an exploratory extension to Baseten-style constitutional safety experiments, where privileged context is a constitution rather than a mathematical answer \cite{kirkby2026dense,bai2022constitutional,beavertails2023}.
```

## Working Order: Problem Setup Before Background

Recommendation: draft the problem setup first as a working anchor, even if the final report later places Background and Related Work before it. The setup defines exactly what the paper studies; after that, the background section can be written to cover only the concepts the reader needs in order to understand the setup and experiments.

### Problem Setup Bullet Points

Goal of the section:

- Formalize the question: for which OLMo 3 checkpoints, if any, does OPSD improve mathematical reasoning over the corresponding base checkpoint?
- Define the checkpoint-indexed evaluation setting.
- Define pass@k as the main evaluation metric, with k=1, 8, and 32 where available.
- Clarify that this is not a final-model leaderboard comparison; it is a study of whether the post-training objective adds value at each training stage.

Suggested structure:

1. Model trajectory:
   - Let \(\theta_t\) denote an OLMo 3 checkpoint at training stage or step \(t\).
   - Each \(\theta_t\) has a base benchmark score before post-training.

2. OPSD intervention:
   - Starting from \(\theta_t\), apply OPSD to obtain a post-trained checkpoint \(\theta_t^{\mathrm{OPSD}}\).
   - OPSD uses on-policy samples and privileged mathematical context, such as the correct answer or reference solution.

3. GRPO reference:
   - Starting from the same or comparable \(\theta_t\), apply GRPO to obtain \(\theta_t^{\mathrm{GRPO}}\).
   - GRPO is the sparse RLVR baseline and connects the study to RL Excursions.

4. Evaluation metrics:
   - Evaluate each checkpoint on GSM8K and MATH/MATH-500.
   - Main metrics are pass@1, pass@8, and pass@32 where available.
   - For each method, compare these pass@k scores to the corresponding base checkpoint.
   - This change relative to base is used to judge whether OPSD or GRPO adds value beyond the checkpoint's existing capability.

5. Central empirical question:
   - Does OPSD improve pass@k over the corresponding base checkpoint for any checkpoint?
   - Does the first positive checkpoint differ between GSM8K and MATH/MATH-500?
   - How does OPSD compare to GRPO at the same stage?

6. Diagnostic question:
   - When OPSD fails to improve, is the privileged teacher signal aligned with reward-improving token directions?
   - Define this informally here; full details can go in the diagnostics section.

### Problem Setup Draft

```latex
\section{Problem Setup}

We study whether OPSD improves mathematical reasoning when applied to intermediate checkpoints of a model during training. Let a checkpoint denote a fixed base model at a particular training stage. For each checkpoint, we evaluate the base model, then apply a post-training method such as OPSD or GRPO, and evaluate the resulting model on the same benchmarks. This setup lets us ask whether the post-training method adds value beyond the capability already present in the checkpoint, rather than only comparing absolute performance across different stages of pretraining.

Our main evaluation metrics are pass@1, pass@8, and pass@32. We report these metrics on GSM8K and MATH-style benchmarks, using the same evaluation protocol for the base, OPSD-trained, and GRPO-trained models. For each checkpoint and benchmark, we compare the post-trained model's pass@k to the corresponding base checkpoint's pass@k. A method is useful at a checkpoint if it improves these scores over the base model from which it started.

This comparison is especially important for checkpoint sweeps. Later OLMo 3 checkpoints may perform better simply because pretraining has progressed. Our goal is not to show that later checkpoints are stronger than earlier checkpoints, but to determine whether OPSD provides an additional improvement at each stage. We therefore treat the base checkpoint as the natural reference point for every OPSD or GRPO run.

We use GRPO as a reference method because it represents the sparse RLVR setting studied in RL Excursions. OPSD, by contrast, uses privileged mathematical context, such as the correct answer or a reference solution, to produce token-level training signal on the model's sampled trajectories. The central empirical question is whether this denser signal leads to better pass@k than the base model, and how its effect compares to GRPO across the same training stages.

Finally, when OPSD does not improve pass@k, we ask whether the failure can be explained by the direction of the training signal. In particular, we study whether the OPSD distillation gradient aligns with token-level directions that empirically improve downstream correctness. This diagnostic connects the benchmark results to a more mechanistic question: whether privileged information actually points the model toward reward-improving reasoning behavior.
```

### Reader Knowledge After Problem Setup

After reading the problem setup, the reader should understand:

- We study checkpoints along an OLMo 3 training trajectory, not just one final model.
- The central comparison is base checkpoint vs. OPSD-trained checkpoint, with GRPO as a reference.
- The main metrics are pass@1, pass@8, and pass@32.
- A method is considered useful at a checkpoint if it improves pass@k over that same checkpoint before post-training.
- OPSD uses privileged mathematical context, while GRPO represents sparse RLVR-style learning.
- If OPSD fails to improve pass@k, we analyze whether its token-level update direction is aligned with reward-improving directions.

This list tells us what the background section must support. Anything that does not help explain one of these points can probably be omitted or moved to an appendix.

### Background And Related Work Draft

```latex
\section{Background and Related Work}

\paragraph{RLVR and GRPO for reasoning.}
Reinforcement learning with verifiable rewards (RLVR) optimizes language models using automatically checkable task outcomes. In mathematical reasoning, a sampled solution receives a reward based on final-answer correctness, and the policy is updated to increase the likelihood of higher-reward samples. GRPO is a sparse on-policy RLVR method that estimates advantages within a group of samples from the same prompt and applies a policy-gradient update without requiring a learned critic \cite{shao2024deepseekmath}. The recent \emph{RL Excursions during Pretraining} study shows that GRPO can improve reasoning even when applied to intermediate pretraining checkpoints, motivating the question of whether other on-policy post-training objectives can also become useful early \cite{bansal2026rlexcursions}.



\paragraph{OPSD as dense on-policy distillation.}
OPSD is motivated by the idea that a model can learn from privileged solutions in the same way a student can study a worked solution and internalize the reasoning. Given a problem \(x\) and reference solution or answer \(y^\star\), OPSD instantiates a student and teacher from the same model parameters but with different conditioning contexts. The student observes only the problem and samples an on-policy response \(\hat{y} \sim p_S(\cdot \mid x)\). The teacher conditions on the same problem together with privileged information \(y^\star\), and both policies evaluate the student-generated trajectory token by token:
\[
p_S(\cdot \mid x,\hat{y}_{<n}),
\qquad
p_T(\cdot \mid x,y^\star,\hat{y}_{<n}).
\]
Rather than assigning a scalar reward to the completed rollout, OPSD minimizes a trajectory-averaged divergence between these next-token distributions:
\[
\mathcal{L}_{\mathrm{OPSD}}
= \mathbb{E}_{(x,y^\star),\hat{y}\sim p_S(\cdot\mid x)}
\left[
\frac{1}{|\hat{y}|}\sum_{n=1}^{|\hat{y}|}
D\!\left(
 p_T(\cdot \mid x,y^\star,\hat{y}_{<n})
 \;\middle\|\;
 p_S(\cdot \mid x,\hat{y}_{<n})
\right)
\right],
\]
where \(D\) may be a KL divergence or a generalized Jensen-Shannon divergence. In our main OPSD runs, we use full-vocabulary distillation with JSD-style mixing, corresponding to \(\alpha=0.5\), and apply pointwise KL clipping to prevent a small number of high-divergence vocabulary entries from dominating the update. This objective gives OPSD a different credit-assignment structure from GRPO: GRPO scores a completed rollout with a sparse reward, while OPSD supplies dense token-level targets along the sampled trajectory. The key risk studied in this project is that the privileged teacher distribution may be dense without being reward-aligned: if it sharpens probability mass around tokens that do not improve final-answer correctness, the dense update can become confidently unhelpful. OPD-style methods have appeared in recent model post-training pipelines and analyses, including Qwen3, DeepSeek-V4, and MiMo-style systems \cite{qwen3technical,deepseekv4,xiao2026mimo,rethinkingopd}.

\paragraph{Checkpoint-stage post-training with OLMo 3.}
Most post-training studies evaluate a method on a fixed final base model. In contrast, this project studies post-training as a function of model training stage. OLMo 3 is useful for this because checkpoints are available across several stages of training \cite{olmo3}. We use this checkpoint structure to ask whether OPSD adds value over the corresponding base checkpoint, rather than only asking whether later checkpoints outperform earlier ones. In our experiments, the main sweep includes Stage 2 checkpoints such as \texttt{stage2-step16000}, \texttt{stage2-step32000}, and \texttt{stage2-step47684}, as well as Stage 3 checkpoints such as \texttt{stage3-step1000}, \texttt{stage3-step4000}, \texttt{stage3-step8000}, and \texttt{main}. These checkpoints let us compare base, OPSD-trained, and GRPO-trained models along the training trajectory.


\paragraph{Gradient-direction diagnostics.}
Benchmark scores reveal whether OPSD improves pass@k, but they do not explain whether the OPSD update points in a useful direction. To analyze this, we use a gradient-alignment diagnostic inspired by recent work on on-policy distillation \cite{armandpour2026unmaskingonpolicydistillationhelps}. For a generated reasoning prefix, we consider a set of candidate next tokens. For each candidate token, we estimate how useful that token is by forcing it as the next token and measuring downstream rollout success. These success estimates define an empirical reward-improving direction over the candidate token distribution. Separately, OPSD defines a distillation direction from the student next-token distribution toward the privileged teacher next-token distribution. We then compare these two directions, for example with cosine similarity, to ask whether the privileged teacher is pushing probability mass toward tokens that actually improve final-answer correctness. In this framing, OPSD should help only when the privileged-context teacher distribution is aligned with the empirical reward-improving direction.

\paragraph{Exploratory safety alignment.}
Although the main focus of this report is mathematical reasoning, we also include a small exploratory safety extension inspired by constitutional alignment. Constitutional AI studies how principles can guide harmless behavior, while BeaverTails provides safety-relevant preference data \cite{bai2022constitutional,beavertails2023}. Baseten-style dense/on-policy safety experiments motivate asking whether privileged context in the form of a constitution can provide useful supervision beyond math \cite{kirkby2026dense}. We treat these experiments as exploratory rather than as the central empirical claim of the report.
```

## Experimental Design: Main Math Experiment

### Experimental Design Draft

```latex
\section{Experimental Design}

\subsection{Overview}
The main experiment evaluates whether OPSD improves mathematical reasoning when applied to intermediate OLMo 3 checkpoints. For each selected checkpoint, we train an OPSD model and compare it to the corresponding base checkpoint. We also include GRPO-trained models as sparse RLVR references. The goal is not to maximize final benchmark performance, but to measure whether each post-training method adds value beyond the checkpoint's existing capability.

\subsection{Checkpoints}
We focus on OLMo 3 7B checkpoints from the later stages of training. The Stage 2 sweep includes \texttt{stage2-step16000}, \texttt{stage2-step32000}, and \texttt{stage2-step47684}. The Stage 3 sweep includes \texttt{stage3-step1000}, \texttt{stage3-step4000}, \texttt{stage3-step8000}, and \texttt{main}. Earlier exploratory runs also considered Stage 1 checkpoints, but the main analysis centers on Stage 2 and Stage 3 because these checkpoints are more likely to have enough mathematical capability for OPSD's privileged signal to be meaningful.

\subsection{Training Data and Objective}
OPSD training uses the OpenMathInstruct-style training data. We use the \texttt{sdpo\_math\_teacher} configuration with student-generated on-policy rollouts. The teacher receives privileged mathematical context through the math-teacher prompt, while the student is trained on the sampled trajectories. After preliminary sweeps, the main OPSD configuration uses JSD-style mixing with \(\alpha=0.5\), a frozen teacher with teacher update rate 0, full-vocabulary distillation, and pointwise KL clipping at 0.05. Training runs use 100 post-training steps, rollout batch size 8, PPO mini-batch size 64, and train batch size 64. We use seed 0 for the main sweep.

\subsection{GRPO Reference Runs}
We compare OPSD against GRPO because GRPO is the sparse RLVR method used as the reference point in RL Excursions. The GRPO runs use the same checkpoint-sweep framing: train from an intermediate checkpoint and compare the resulting model against the same base checkpoint. This allows us to distinguish failures of early post-training in general from failures specific to OPSD's privileged distillation signal.

\subsection{Benchmark Evaluation}
Training validation is not used as the final source of truth. Instead, we evaluate base and post-trained checkpoints with a separate benchmark pipeline under \texttt{outputs/pretrain\_benchmarks}. This evaluation follows an RL-Excursions-style setup: base checkpoints are evaluated in \texttt{base} prompt mode, trained checkpoints in \texttt{trained} prompt mode, with \texttt{rlx} prompt style, temperature 0.6, top-p 1.0, and 32 sampled completions where feasible. The main metrics are pass@1, pass@8, and pass@32 on GSM8K and MATH-style benchmarks. When only smaller runs are available, we report the available pass@k values and clearly mark them as such.

\subsection{Model Conversion and Evaluation Details}
Base Hugging Face checkpoints are evaluated directly. Post-trained actor checkpoints are first merged from FSDP format into Hugging Face format before benchmark evaluation. We use the same answer extraction and benchmark scripts for base, OPSD, and GRPO models to avoid conflating training-validation behavior with benchmark performance. This distinction is important because OpenMathInstruct validation uses chat-style prompts and boxed-answer reward parsing, while the benchmark path uses plain benchmark prompts and computes pass@k from a shared sample pool.

\subsection{Reporting}
For each checkpoint, method, and benchmark, we report absolute pass@k and compare it to the corresponding base checkpoint. We summarize whether OPSD improves over base, whether GRPO improves over base, and whether the relative pattern changes across training stages. The main result is the checkpoint-wise comparison between base, OPSD, and GRPO, rather than a single final-model score.
```

### Experimental Design Notes

Key choices to verify before finalizing:

- Confirm whether final benchmark table uses `MATH`, `MATH-500`, or both.
- Confirm the exact set of GRPO checkpoints with completed benchmark evaluations.
- Confirm whether all final runs have pass@32 or whether some only have pass@1/pass@8.
- Confirm if the final OPSD configuration is always `alpha=0.5`, `teacher_update_rate=0.0`, `distillation_topk=null`, `pointwise_kl_clip=0.05`, or whether any reported checkpoint uses a different setting.
- Keep training validation separate from `outputs/pretrain_benchmarks`; do not mix validation `auto_acc_mean` with pass@k in the main results table.

### Background And Related Work Bullet Points

Goal of the section:

- Keep this brief: 1 to 1.5 pages in a report, or less if space is tight.
- Explain only the pieces needed for the problem setup and experiments.

Suggested subsections:

#### RLVR And GRPO For Reasoning

- RLVR trains models using automatically verifiable rewards, often for math or code.
- GRPO is a sparse on-policy RL method used for mathematical reasoning.
- RL Excursions shows that GRPO can improve reasoning surprisingly early during pretraining.
- Papers to cite:
  - DeepSeekMath / GRPO \cite{shao2024deepseekmath}
  - RL Excursions \cite{bansal2026rlexcursions}
  - PPO only if needed for broader RL context \cite{schulman2017ppo}

#### OPSD / OPD

- OPSD is motivated as an alternative to RLVR that can convert richer feedback or privileged context into dense token-level supervision.
- The key promise is better credit assignment than sparse rollout rewards.
- The key risk, central to this project, is that the privileged teacher distribution may not align with reward-improving token directions.
- Papers to cite:
  - OPSD / On-Policy Self-Distillation paper \cite{hubotter2026sdpo}
  - OPD usage or analyses in Qwen3, DeepSeek-V4, MiMo-style systems \cite{qwen3technical,deepseekv4,xiao2026mimo,rethinkingopd}
  - Privileged-information distillation, if included in the bibliography.

#### Early Post-Training During Pretraining

- This is the conceptual bridge between RL Excursions and our work.
- Standard post-training is usually applied late; checkpoint-rich models let us ask when a method becomes useful.
- Papers to cite:
  - RL Excursions \cite{bansal2026rlexcursions}
  - OLMo 3 technical report \cite{olmo3}

#### Math Reasoning Benchmarks

- GSM8K: grade-school math word problems with verifiable numeric answers.
- MATH/MATH-500: harder competition-style math reasoning.
- Papers to cite:
  - GSM8K \cite{cobbe2021training}
  - MATH \cite{hendrycks2021measuring}

#### Gradient Alignment / Diagnostics

- Motivation: benchmark deltas tell us whether OPSD helps, but not why it fails.
- Gradient-direction diagnostics compare the OPSD update direction to an empirical reward-improving direction.
- Papers to cite:
  - Apple's gradient-alignment diagnostic paper \cite{armandpour2026unmaskingonpolicydistillationhelps}
  - Any RLVR entropy/exploration papers if used in the analysis section.

#### Safety Alignment Background, Optional Brief Paragraph

- Only include if the Baseten/constitutional section remains in the final report.
- Position it as exploratory rather than central.
- Papers to cite:
  - Constitutional AI \cite{bai2022constitutional}
  - BeaverTails \cite{beavertails2023}
  - Baseten dense/on-policy post \cite{kirkby2026dense}

### Proposed Order In The Report

1. Introduction
2. Problem Setup
3. Background and Related Work
4. Experimental Design
5. Results
6. Gradient-Direction Diagnostics
7. Exploratory Safety Experiments
8. Limitations
9. Conclusion

Alternative if the report should feel more conventional:

1. Introduction
2. Background and Related Work
3. Problem Setup
4. Experimental Design
5. Results
6. Diagnostics
7. Limitations and Conclusion

The first order is probably better for this project because it makes the report less like a survey and more like a focused empirical study.


```latex
%%%%%%%%%%%%%%%%%%%%%%
\chapter{Background and Related Work}
%%%%%%%%%%%%%%%%%%%%%%



```


## Papers To Include

Use this as a citation checklist when you send over your BibTeX. The exact keys below are placeholders; we should replace them with your saved keys.

### Essential framing

- `RL Excursions during Pretraining: How Early Is Too Early for On-policy Learning?` \cite{bansal2026rlexcursions}
  - Why: directly motivates the question "when does the post-training objective start to work?"
  - Use for: early GRPO during pretraining, pass@1/pass@k framing, comparison to this project.
- `Reinforcement Learning via Self-Distillation` / SDPO \cite{hubotter2026sdpo}
  - Why: anchors the dense self-distillation mechanism in this repo.
  - Use for: explaining SDPO/OPSD as dense feedback rather than only scalar reward.
- OPSD-specific paper(s), if separate from SDPO in your bibliography.
  - Why: your project uses privileged answer information, so this may be the most faithful method citation.
  - Use for: method definition and terminology.
- Qwen3 technical report, DeepSeek-V4 technical report, and MiMo technical report / OPD analysis papers \cite{qwen3technical,deepseekv4,xiao2026mimo,rethinkingopd}.
  - Why: shows OPD is becoming a practical post-training and capability-consolidation recipe in recent model pipelines.
  - Use for: motivating why studying OPD/OPSD failure modes is relevant beyond this single project.

### RL and reasoning baselines

- `DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models` \cite{shao2024deepseekmath}
  - Why: introduces GRPO in the math-reasoning setting.
  - Use for: GRPO reference baseline.
- `Proximal Policy Optimization Algorithms` \cite{schulman2017ppo}
  - Why: optional background if you explain GRPO as a critic-free relative of PPO-style policy optimization.
- `Training Verifiers to Solve Math Word Problems` \cite{cobbe2021training}
  - Why: introduces GSM8K.
  - Use for: GSM8K benchmark motivation and verifiable math reasoning.
- `Measuring Mathematical Problem Solving With the MATH Dataset` \cite{hendrycks2021measuring}
  - Why: introduces MATH.
  - Use for: harder mathematical reasoning benchmark.

### Model and evaluation infrastructure

- OLMo 3 technical report \cite{olmo3}
  - Why: justifies the checkpoint-sweep design.
  - Use for: open checkpoints across training stages, model background.
- OLMo / OLMo 2 papers, if OLMo 3 citation is unavailable or if your report needs the broader model-family context.
  - Why: supports the claim that the model family is unusually transparent/open.

### Safety exploratory section

- `Constitutional AI: Harmlessness from AI Feedback` \cite{bai2022constitutional}
  - Why: motivates constitution-as-privileged-context and RLAIF-style safety training.
- `BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset` \cite{beavertails2023}
  - Why: dataset used for the safety experiments.
- `Dense, on-policy, or both?` \cite{kirkby2026dense}
  - Why: directly motivates the Baseten-style exploratory section.
  - Use for: dense vs sparse and on-policy vs off-policy framing.

### Optional but useful

- `Training language models to follow instructions with human feedback` \cite{ouyang2022training}
  - Why: standard post-training/RLHF background.
- `Chain-of-Thought Prompting Elicits Reasoning in Large Language Models` \cite{wei2022chain}
  - Why: optional if you discuss reasoning traces and step-by-step solution generation.
- `DAgger` \cite{ross2011dagger}
  - Why: useful analogy if you later discuss why supervising on the learner's own state distribution may matter.


### Papers To Look Up Next

These are especially relevant to the newer introduction angle, where the result is that OPSD mostly tracks the base model while GRPO improves.

- `Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models` / OPSD, arXiv 2601.18734.
  - Look for: the exact assumption that the model can use privileged answers or traces to teach its non-privileged self.
  - Use for: method definition and for explaining why our math setting stresses that assumption.
- `Privileged Information Distillation for Language Models`, arXiv 2602.04942.
  - Look for: privileged-information teacher/student framing and any discussion of when PI-conditioned teachers help or fail.
  - Use for: the broader PI-distillation framing beyond the SDPO repo.
- Apple's gradient-alignment or diagnostic paper that inspired the branching-gradient method.
  - Look for: how they define the empirical reward-improving gradient and how they compare it to a training objective gradient.
  - Use for: motivating our gradient-direction contribution.
- Recent RLVR entropy/exploration papers.
  - Search terms: `RLVR entropy collapse`, `high entropy tokens reasoning`, `exploration exploitation RLVR`, `does RL really incentivize reasoning beyond base model`.
  - Use for: explaining why making the model more confident may not be enough for math reasoning, and why exploration can matter.
- Recent OPSD/RLSD/OGLS-SD follow-up work.
  - Search terms: `OPSD information leakage instability`, `self-distilled RLVR stable policy optimization`, `outcome-guided logit steering OPSD`.
  - Use for: citing work that diagnoses why OPSD can be unstable or why answer-conditioned dense supervision may need extra outcome guidance.


## Open Gaps Before Writing The Final Report

- Identify the final source of truth for results: WandB runs, `outputs/pretrain_benchmarks`, validation generation folders, or RunAI logs.
$ We will use the outputs/pretrain_benchmarks as our final source of truth
- Fill the result table with exact numbers and metric definitions.
- Decide whether the report calls the method `OPSD`, `SDPO`, or `OPSD/SDPO` throughout.
$ It seems like we are using OPSD more faithfully since they use the answer to the question as their privileged information
- Confirm which benchmarks are final: `GSM8K`, `MATH`, `MATH-500`, or a subset.
$ We will have the 'GSM8K' and 'MATH-500' as our final benchmarks, but the thing is that we have ran all the benchmarks on the 'MATH' one, so I might just keep the unofficial 'MATH' benchmark.
- Decide whether the safety experiments are a main result, an exploratory section, or future work.
$ We will use this section as an exploratory section. 
- Add exact citation keys for RL Excursions, OLMo 3, OPSD/SDPO, GRPO, GSM8K, MATH, and BeaverTails.
$ The citations will be included in the bibliogrpahy.
