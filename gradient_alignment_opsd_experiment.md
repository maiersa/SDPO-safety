# Gradient-Alignment Diagnostic for OPSD Across OLMo 3 Training Stages

## 1. Goal

Implement an experiment to test whether OPSD becomes useful only after a model has enough task/context understanding for privileged-context distillation gradients to align with reward-improving gradients.

The core research question is:

> Across OLMo 3 7B training stages, does the OPSD gradient increasingly align with the ideal gradient that would improve final-answer correctness?

The main metric is cosine similarity between:

- `g_ideal`: an empirical reward-improving gradient estimated by branching on candidate next tokens and measuring downstream success.
- `g_opsd`: the distillation gradient induced by a privileged-context teacher, usually the same model evaluated with extra information such as the final answer or full reference solution.

The experiment should be compute-conscious and should not attempt to reproduce the full Apple diagnostic at full scale.

---

## 2. Models / Checkpoints

Use three OLMo 3 7B checkpoints as the minimal experiment backbone:

1. `olmo3_7b_stage1_final`
   - Last checkpoint of Stage 1 pretraining.
   - Represents mostly general pretraining.

2. `olmo3_7b_stage2_final`
   - Last checkpoint of Stage 2 / math + code midtraining.
   - Represents a model with improved math/code reasoning exposure.

3. `olmo3_7b_think_final`
   - Final thinking model checkpoint, preferably the final RL or Think model if available.
   - Represents explicit reasoning-oriented post-training.

Optional extension:

- Add early Stage 2 and Think-SFT / Think-DPO checkpoints if available.
- This would allow identifying whether the transition happens during Stage 2 or during thinking post-training.

---

## 3. Teacher Context Variants

For each student checkpoint, construct teachers from the same base checkpoint but evaluated under different context conditions.

### Teacher A: No privileged context / control

Use the same prompt as the student.

Purpose:

- Sanity check.
- Distillation signal should be weak, zero, or uninformative if teacher and student are identical.

### Teacher B: Final-answer privileged context

Prompt the teacher with the question and the correct final answer.

Example template:

```text
Question:
{question}

The correct final answer is:
{answer}

The student solution so far is:
{student_prefix}

Continue the solution in a way that leads to the correct answer.
```

Purpose:

- Tests whether knowing only the final answer is enough to produce useful token-level guidance.

### Teacher C: Full-solution privileged context

Prompt the teacher with the question and the reference solution.

Example template:

```text
Question:
{question}

Reference solution:
{reference_solution}

The student solution so far is:
{student_prefix}

Continue or correct the reasoning in a way that follows the reference solution.
```

Purpose:

- Tests whether richer privileged context produces better OPSD gradients.

Main comparison:

- Answer-only teacher vs full-solution teacher.

---

## 4. Dataset

Use math/reasoning questions with automatically checkable final answers.

Recommended main experiment dataset:

- 30 GSM8K-style grade-school math questions.
- 15 easy/intermediate MATH questions, preferably algebra/counting/number theory with short final answers.
- 5 synthetic arithmetic/logic questions.

Total: 50 questions.

Difficulty split:

- 30 easy questions where Stage 1 has nonzero success probability.
- 15 medium questions where Stage 2 should improve.
- 5 harder but still tractable questions where Think should separate itself.

Avoid starting with only very hard MATH questions, because early checkpoints may have near-zero success across all branches, making the ideal-gradient estimate noisy and uninformative.

Each dataset example should contain:

```json
{
  "id": "unique_question_id",
  "source": "gsm8k|math|synthetic",
  "difficulty": "easy|medium|hard",
  "question": "...",
  "answer": "...",
  "reference_solution": "..."
}
```

---

## 5. Generation Settings

Use the same generation settings across checkpoints unless there is a strong reason not to.

Recommended student rollout settings:

```yaml
temperature: 0.7
top_p: 0.95
max_new_tokens: 512
num_student_rollouts_per_question: 2
```

For the Think model, optionally allow:

```yaml
max_new_tokens: 1024
```

But keep this controlled and record token lengths.

---

## 6. Pipeline Overview

For each model checkpoint and each question:

1. Generate student rollouts.
2. Grade each rollout as correct/incorrect.
3. Select a small number of diagnostic token positions/nodes from each rollout.
4. At each selected node:
   - Compute student next-token distribution.
   - Compute teacher next-token distribution under each teacher context.
   - Select candidate next tokens from top-k student and teacher tokens.
   - Estimate downstream success probability for each candidate token by forced-token rollouts.
   - Compute candidate-set ideal gradient.
   - Compute candidate-set OPSD/distillation gradient.
   - Compute cosine alignment.
5. Aggregate alignment metrics by checkpoint, teacher context, correctness, entropy, KL, and dataset type.

---

## 7. Student Rollouts

For each `(model_checkpoint, question)` pair:

- Generate `N = 2` student rollouts.
- Store full generated text and token IDs.
- Extract final answer using a parser.
- Mark the rollout as correct or incorrect.

Store:

```json
{
  "question_id": "...",
  "checkpoint": "...",
  "rollout_id": "...",
  "prompt": "...",
  "generated_text": "...",
  "generated_token_ids": [...],
  "parsed_answer": "...",
  "is_correct": true,
  "num_generated_tokens": 123
}
```

---

## 8. Node / Token Position Selection

Do not analyze every token position. Select only a few diagnostic nodes per rollout.

Recommended:

```yaml
nodes_per_rollout: 3
```

Node selection rule:

- Select 2 high teacher-student KL positions.
- Select 1 high student entropy position.

For each token position `t` in the student trajectory:

- Let `prefix_t` be the prompt plus generated tokens up to but not including token `t`.
- Compute student distribution `pi_s(. | prefix_t)`.
- Compute teacher distribution `pi_t(. | teacher_context(prefix_t))`.
- Compute:
  - student entropy `H(pi_s)`
  - teacher-student KL, preferably `KL(pi_s || pi_t)`

Candidate node filters:

- Ignore very early positions such as the first 3 generated tokens.
- Ignore special tokens unless they are semantically meaningful.
- Optionally prioritize numeric/operator/final-answer-region tokens.
- Optionally ignore positions after the model has already emitted a final answer marker.

Store selected nodes with metadata:

```json
{
  "question_id": "...",
  "checkpoint": "...",
  "rollout_id": "...",
  "node_id": "...",
  "token_position": 42,
  "prefix_token_ids": [...],
  "prefix_text": "...",
  "selection_reason": "high_kl|high_entropy|heuristic",
  "student_entropy": 2.31,
  "student_teacher_kl": 0.84,
  "student_rollout_correct": false
}
```

---

## 9. Candidate Token Selection

At each selected node and teacher context, define the candidate token set:

```text
K = top_k_student_tokens ∪ top_k_teacher_tokens
```

Recommended:

```yaml
top_k_student: 5
top_k_teacher: 5
```

Deduplicate candidate tokens.

Store for each candidate:

```json
{
  "token_id": 1234,
  "token_str": "...",
  "student_logprob": -1.23,
  "teacher_logprob": -0.95,
  "in_student_topk": true,
  "in_teacher_topk": true
}
```

Use candidate-set renormalized probabilities for gradient calculations:

```text
p_s_k = softmax(student_logits over K)
p_t_k = softmax(teacher_logits over K)
```

---

## 10. Estimating Downstream Success Probabilities

For each selected node and each candidate token `k`:

1. Force the prefix:

```text
prompt + generated_tokens_before_node + candidate_token_k
```

2. Generate `R` continuations from the student model.

Recommended:

```yaml
forced_rollouts_per_candidate: 4
```

3. Parse and grade each continuation.

4. Estimate:

```text
P_success[k] = num_correct_continuations_after_k / R
```

Store:

```json
{
  "question_id": "...",
  "checkpoint": "...",
  "rollout_id": "...",
  "node_id": "...",
  "teacher_context": "answer_only|full_solution",
  "candidate_token_id": 1234,
  "forced_rollouts": [
    {
      "continuation_text": "...",
      "parsed_answer": "...",
      "is_correct": false
    }
  ],
  "p_success": 0.25
}
```

---

## 11. Ideal Gradient Computation

For candidate set `K`, use renormalized student probabilities:

```text
p_s[k] = P_s(k) / sum_{j in K} P_s(j)
```

Estimate expected success under the candidate-set student distribution:

```text
baseline_success = sum_{j in K} p_s[j] * P_success[j]
```

Then compute the candidate-set ideal gradient:

```text
g_ideal[k] = p_s[k] * (P_success[k] - baseline_success)
```

Interpretation:

- Tokens that lead to above-average downstream success receive positive gradient.
- Tokens that lead to below-average downstream success receive negative gradient.
- The update is scaled by the student probability.

---

## 12. OPSD / Distillation Gradient Computation

Use a candidate-set KL-style distillation gradient.

For forward KL:

```text
KL(pi_s || pi_t) = sum_k p_s[k] * (log p_s[k] - log p_t[k])
```

Let:

```text
ell[k] = log p_s[k] - log p_t[k]
ell_bar = sum_j p_s[j] * ell[j]
```

The descent direction for minimizing forward KL is:

```text
g_opsd[k] = - p_s[k] * (ell[k] - ell_bar)
```

This is the candidate-set approximation of the OPSD gradient.

Important:

- Use log probabilities from the same tokenization and exact same prefix.
- Use `epsilon` clipping or numerical stabilization for extremely small probabilities.
- Keep track of whether the gradient is computed over full vocabulary or candidate set. The recommended implementation is candidate-set only.

---

## 13. Alignment Metric

Compute cosine similarity:

```text
alignment = dot(g_ideal, g_opsd) / (norm(g_ideal) * norm(g_opsd))
```

If either gradient norm is near zero, mark alignment as `null` or `nan` and track separately.

Recommended threshold:

```yaml
min_gradient_norm: 1e-8
```

Store:

```json
{
  "question_id": "...",
  "checkpoint": "...",
  "rollout_id": "...",
  "node_id": "...",
  "teacher_context": "answer_only|full_solution",
  "student_rollout_correct": false,
  "student_entropy": 2.31,
  "student_teacher_kl": 0.84,
  "candidate_token_ids": [...],
  "p_student": [...],
  "p_teacher": [...],
  "p_success": [...],
  "g_ideal": [...],
  "g_opsd": [...],
  "alignment": 0.27
}
```

---

## 14. Evaluation / Grading

Implement answer extraction and grading for each dataset type.

For GSM8K:

- Extract final numeric answer.
- Normalize commas, currency symbols, percentages, and whitespace.
- Compare numerically where possible.

For MATH:

- Prefer examples with short numeric or symbolic answers.
- Normalize LaTeX formatting if needed.
- Avoid examples where grading requires complex equivalence checking in the first version.

For synthetic:

- Use exact or numeric comparison.

Store both raw and normalized answers.

---

## 15. Main Aggregations

Compute summary statistics grouped by:

- checkpoint
- teacher context
- dataset source
- difficulty
- student rollout correctness
- node selection reason

For each group, report:

```text
count_nodes
mean_alignment
median_alignment
std_alignment
standard_error
fraction_positive_alignment
mean_student_teacher_kl
mean_student_entropy
mean_student_success_rate
```

Important comparisons:

1. Alignment by checkpoint:

```text
Stage 1 vs Stage 2 vs Think
```

2. Alignment by teacher context:

```text
answer-only vs full-solution
```

3. Alignment on incorrect vs correct student trajectories.

4. Alignment vs KL.

5. Alignment vs base success rate.

---

## 16. Required Plots

Generate at least the following plots:

### Plot 1: Mean alignment by checkpoint and teacher context

- x-axis: checkpoint
- y-axis: mean cosine alignment
- hue/color: teacher context
- include error bars

### Plot 2: Alignment by correctness

- x-axis: checkpoint
- y-axis: mean alignment
- hue/color: correct vs incorrect student rollout

### Plot 3: Alignment vs KL

- x-axis: `KL(pi_s || pi_t)`
- y-axis: cosine alignment
- color: checkpoint
- optionally facet by teacher context

### Plot 4: Alignment vs student success rate

- x-axis: checkpoint-level base pass@1 or empirical success rate
- y-axis: mean alignment

### Plot 5: Distribution of alignment values

- histogram or violin plot
- grouped by checkpoint and teacher context

---

## 17. Expected Hypotheses

The experiment should test the following hypotheses:

### H1: Alignment increases with training stage

Expected:

```text
alignment(Stage 1) < alignment(Stage 2) < alignment(Think)
```

Interpretation:

- OPSD becomes more useful as the model gains task/context understanding.

### H2: Full-solution context aligns better than answer-only context

Expected:

```text
alignment(full_solution_teacher) > alignment(answer_only_teacher)
```

Interpretation:

- Richer privileged context gives more useful token-level guidance.

### H3: Alignment is more meaningful on incorrect-but-recoverable trajectories

Expected:

- Incorrect but coherent trajectories should show useful positive alignment if OPSD can correct the model.
- Fully correct trajectories may have weaker or noisier alignment.
- Completely incoherent trajectories may have near-zero or negative alignment.

### H4: KL magnitude alone is not enough

Expected:

- Stage 1 may have high teacher-student KL but low alignment.
- Later checkpoints may have more useful KL, meaning disagreement points in reward-improving directions.

Interpretation:

- Teacher-student disagreement is not sufficient. The disagreement must be aligned with success-improving directions.

---

## 18. Recommended Experiment Sizes

### Smoke Test

Use this first to validate the implementation.

```yaml
models: 3
questions: 20
student_rollouts_per_question: 2
nodes_per_rollout: 2
candidate_tokens: top3_student_union_top3_teacher
forced_rollouts_per_candidate: 4
teacher_contexts:
  - answer_only
  - full_solution
```

Estimated forced continuations:

```text
3 * 20 * 2 * 2 * 6 * 4 * 2 teacher_contexts ≈ 11,520
```

Note: If candidate sets overlap strongly, this will be lower.

### Main Experiment

```yaml
models: 3
questions: 50
student_rollouts_per_question: 2
nodes_per_rollout: 3
candidate_tokens: top5_student_union_top5_teacher
forced_rollouts_per_candidate: 4
teacher_contexts:
  - answer_only
  - full_solution
```

Estimated forced continuations:

```text
3 * 50 * 2 * 3 * 8 * 4 * 2 teacher_contexts ≈ 57,600
```

This is the recommended first serious run.

### Larger Experiment

```yaml
models: 5
questions: 100
student_rollouts_per_question: 4
nodes_per_rollout: 5
candidate_tokens: top5_student_union_top5_teacher
forced_rollouts_per_candidate: 8
teacher_contexts:
  - answer_only
  - full_solution
  - correction_style
```

This can become very expensive and should only be run after the smaller experiment is validated.

---

## 19. Implementation Structure

Suggested project structure:

```text
opsd_alignment/
  configs/
    smoke_test.yaml
    main_experiment.yaml
  data/
    questions.jsonl
  scripts/
    generate_student_rollouts.py
    select_nodes.py
    compute_teacher_student_distributions.py
    estimate_success_branches.py
    compute_gradients_and_alignment.py
    aggregate_results.py
    plot_results.py
  src/
    models.py
    prompts.py
    grading.py
    node_selection.py
    candidate_selection.py
    rollouts.py
    gradients.py
    metrics.py
    storage.py
  outputs/
    rollouts/
    nodes/
    branches/
    alignments/
    summaries/
    plots/
```

---

## 20. Implementation Notes

- Use batching aggressively for log-prob computation and forced rollouts.
- Cache all model outputs and logits/logprobs.
- Make the pipeline resumable: every step should write JSONL or Parquet outputs.
- Never recompute rollouts if the output file already exists unless explicitly requested.
- Store random seeds for every rollout.
- Log exact model checkpoint, tokenizer, generation config, and prompt template.
- Use deterministic grading where possible.
- Track invalid parses separately rather than silently marking them wrong.
- Keep raw text for manual inspection.

---

## 21. Success Criteria

The experiment is successful if it produces:

1. A reliable table of mean alignment by checkpoint and teacher context.
2. Plots showing whether alignment increases from Stage 1 to Stage 2 to Think.
3. Evidence about whether answer-only or full-solution OPSD context is more useful.
4. Evidence about whether KL magnitude predicts usefulness or whether alignment is a better signal.
5. A small set of qualitative examples showing positive, near-zero, and negative alignment cases.

A strong result would show:

```text
Stage 1: high KL but low/negative alignment
Stage 2: moderate positive alignment
Think: clearly positive alignment, especially on incorrect but recoverable trajectories
```

This would support the thesis that OPSD is useful only once the student has sufficient context/task understanding for privileged-context gradients to point in success-improving directions.
