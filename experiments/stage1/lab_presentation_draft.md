# Lab Presentation Draft: Early OPSD/SDPO and RL During Training

This draft turns the Stage 1 README into a short 3-slide story for a lab presentation.

Assumption:
- I am treating your mention of `OPSD` as referring to your `SDPO`-style post-training method in this repo, and I use `OPSD/SDPO` on the slides so you can match the wording you prefer in the talk.

## Slide 1

### Title

`What Is OPSD/SDPO, and Why Try It Early?`

### Goal of the slide

Give the room a fast mental model:
- we start from an intermediate pretrained checkpoint
- we sample model answers on reasoning tasks
- we score them with a verifiable outcome
- we update the policy so the model reinforces useful behaviors from its own attempts

### Suggested visual

Use the custom figure:
- [opsd_sdpo_pipeline.svg](/dlabscratch1/samaier/projects/SDPO-safety/experiments/stage1/presentation_assets/opsd_sdpo_pipeline.svg)

Optional second image on the right:
- a cropped screenshot of Figure 1 from the RL Excursions page, which illustrates RL across intermediate pretraining checkpoints
- source: https://rl-excursions.github.io/

### On-slide bullets

- `OPSD/SDPO` asks whether a model can improve from its own sampled reasoning traces, not only from curated demonstrations.
- The key question is not just `does it work?`, but `how early does it start to work?`
- If it turns on early, we may be able to reinforce useful reasoning behavior before the model is fully pretrained.

### Speaker notes

“The first idea I want to motivate is very simple. Instead of waiting until the end of training, we take an intermediate checkpoint and ask whether a self-improvement style objective like OPSD or SDPO is already useful there. The model generates several candidate solutions, we evaluate them with a verifiable reward, and then we push the policy toward the more successful behaviors. Conceptually, this is attractive because it could let us reinforce reasoning-related structure much earlier than the standard pipeline would.”

## Slide 2

### Title

`GRPO Seems Useful Surprisingly Early`

### Goal of the slide

Anchor your intuition with an external result:
- RL can already help at relatively early checkpoints
- gains depend on task difficulty
- early training is promising, but brittle

### Suggested visuals

Use one of these combinations:

Option A:
- screenshot or redraw of Figure 2 from RL Excursions for GSM8K
- screenshot or redraw of Figure 3 for MATH

Option B:
- Figure 1 for the overview
- your custom summary figure [early_rl_window.svg](/dlabscratch1/samaier/projects/SDPO-safety/experiments/stage1/presentation_assets/early_rl_window.svg)

### On-slide bullets

- In `RL Excursions during Pretraining`, direct `GRPO` on base checkpoints improved reasoning surprisingly early.
- On `GSM8K`, direct RL showed meaningful gains as early as about `4B` pretraining tokens.
- On `MATH`, RL still helped over base, but did not catch up to the stronger `SFT -> RL` pipeline.
- Early-stage RL is promising, but seed sensitivity and sparse rewards remain real concerns.

### Concrete talking points from the study

Use these exact claims in paraphrased form:
- The authors report that on-policy learning is effective “starting very early during standard pretraining.”
- Their `RL-only` pipeline improves both `pass@1` and `pass@k`, which they interpret as evidence of expansion rather than only sharpening.
- They also note brittleness: between roughly `4B` and `10B` tokens, outcomes can vary substantially across seeds even when training rewards look similar.

### Speaker notes

“The second point is that this general idea is no longer purely speculative. A recent study on GRPO across intermediate checkpoints found that RL can help much earlier than many of us would have expected. On GSM8K, they see meaningful improvements already around 4 billion pretraining tokens. On harder MATH problems, the story is more mixed: RL still helps compared with the base checkpoint, but it does not fully close the gap with the standard SFT-then-RL recipe. So the encouraging part is that early on-policy learning can work; the cautionary part is that the gains are task-dependent and somewhat brittle.”

## Slide 3

### Title

`What We Are Testing: The SDPO Feasibility Threshold`

### Goal of the slide

Move from external motivation to your project:
- define your central question
- show the Stage 1 experimental design
- end with the lab-discussion questions

### Suggested visual

Use the custom figure:
- [stage1_threshold_hypothesis.svg](/dlabscratch1/samaier/projects/SDPO-safety/experiments/stage1/presentation_assets/stage1_threshold_hypothesis.svg)

Optional small inset:
- a simplified version of the Stage 1 wave design from the README

### On-slide bullets

- Stage 1 asks: `when does SDPO first become net-positive over the base model during pretraining?`
- We measure `delta = metric(post-train) - metric(base checkpoint)` across checkpoint progress.
- Main hypothesis: SDPO becomes effective only after a minimum capability threshold.
- We expect the threshold to appear earlier on `GSM8K` than on `MATH`.
- We compare against a `GRPO` reference subset to ask whether SDPO turns on earlier, later, or at the same point.

### Questions to put at the bottom of the slide

- `How early can OPSD/SDPO help?`
- `Is the threshold dataset-dependent?`
- `What should we prioritize next: denser checkpoint sweeps, more seeds, or mixing RL/SDPO earlier into training?`

### Speaker notes

“This is where our project comes in. We are not trying to solve all of alignment or all of RL pretraining at once. Stage 1 is deliberately narrow: we want to identify the first checkpoint where SDPO is actually net-positive over the base model, and whether that threshold differs between GSM8K and MATH. The main hypothesis is that there is a capability threshold below which SDPO does not help much, and above which it starts to reinforce something real. We also expect that threshold to arrive earlier on easier reasoning tasks like GSM8K than on MATH. The reason I’m bringing this to the lab is that the next-step decision is not obvious: should we invest first in denser checkpoint coverage, more seeds near the threshold, or comparisons with GRPO to understand whether SDPO turns on earlier or later?”

## Recommended Flow For The Talk

If you want this to feel smooth and discussion-oriented, keep the main narrative as:

1. `Why this is interesting`
   Early self-improvement or RL-style signals might be useful before full pretraining is complete.
2. `Why this is plausible`
   GRPO evidence suggests on-policy learning can help surprisingly early.
3. `What we are concretely testing`
   We want to measure the earliest positive SDPO checkpoint and understand what controls that threshold.
4. `What the lab can help with`
   Experimental priorities, possible confounds, and alternative hypotheses.

## Figures To Use

### Figure 1: OPSD/SDPO method schematic

File:
- [opsd_sdpo_pipeline.svg](/dlabscratch1/samaier/projects/SDPO-safety/experiments/stage1/presentation_assets/opsd_sdpo_pipeline.svg)

Why it works:
- clean overview of the training loop
- visually separates pretraining checkpoint, sampled reasoning traces, verifiable reward, and policy update
- useful for an audience that may not know the acronym yet

### Figure 2: Early-RL opportunity window

File:
- [early_rl_window.svg](/dlabscratch1/samaier/projects/SDPO-safety/experiments/stage1/presentation_assets/early_rl_window.svg)

Why it works:
- communicates the key takeaway without requiring people to parse a dense paper figure
- highlights `too early`, `promising window`, and `late-stage baseline`
- gives you a natural segue into seed sensitivity and task difficulty

### Figure 3: Stage 1 threshold hypothesis

File:
- [stage1_threshold_hypothesis.svg](/dlabscratch1/samaier/projects/SDPO-safety/experiments/stage1/presentation_assets/stage1_threshold_hypothesis.svg)

Why it works:
- maps directly onto your README
- makes the central experiment legible in one glance
- shows the expected offset between `GSM8K` and `MATH`

## Full Short Speech

“Today I want to get feedback on one main question: how early in training can a self-improvement style objective like OPSD or SDPO start helping a model learn useful reasoning behavior?

The motivation is that we usually think of post-training methods as something we apply after pretraining is mostly done. But if these objectives can already reinforce useful structure earlier, then maybe we should not treat them purely as a final-stage tool.

At a high level, the method is straightforward. We take an intermediate checkpoint, have the model generate several candidate reasoning traces, score them with a verifiable signal, and then update the policy to reinforce the more successful behavior. So the core issue is not whether this can ever work on a strong model, but when it first becomes net-positive during training.

There is already evidence pointing in this direction. A recent GRPO study looked at RL directly on intermediate pretraining checkpoints and found that on-policy learning can help surprisingly early, especially on GSM8K-like reasoning tasks. They report meaningful gains well before the model is fully pretrained. At the same time, the results are more mixed on harder tasks like MATH, and the early regime is brittle across seeds. So the evidence is encouraging, but it also suggests there may be a real capability threshold and a real dependence on task difficulty.

That is exactly what we are trying to probe in Stage 1. Our question is: when does SDPO first become net-positive over the base checkpoint during pretraining, and does that threshold differ between GSM8K and MATH? Operationally, we measure delta as post-training performance minus base checkpoint performance, and we care about the first positive and repeatable delta.

Our current hypothesis is that SDPO only becomes effective once the base policy crosses a minimum capability threshold, that this threshold appears earlier on GSM8K than on MATH, and that the first successful checkpoint may actually be earlier than we would expect from looking only at final-model behavior. We also suspect this threshold region will be especially seed-sensitive.

So the main reason I’m presenting this to the lab is to get advice on the next experimental move. Should we spend effort on a denser checkpoint sweep around the transition region? Should we prioritize more seeds to distinguish real gains from noise? Or should we focus more on comparisons with GRPO, to understand whether SDPO turns on earlier, later, or at roughly the same point?

The broader question behind all of this is whether early OPSD or SDPO can help the model learn better, not just at the end of training, but as part of shaping the trajectory itself. That is the main thing I’d love feedback on.”

## Very Short Version If You Only Have 2 Minutes

“I want feedback on whether OPSD or SDPO can already help at intermediate checkpoints rather than only after pretraining is mostly complete. A recent GRPO study suggests early on-policy learning can work surprisingly early, especially on GSM8K, though it becomes less clean on harder tasks like MATH and is brittle across seeds. Our Stage 1 experiment is to find the earliest checkpoint where SDPO becomes net-positive over the base model, compare that threshold on GSM8K versus MATH, and use GRPO as a reference point. The main question for the lab is what we should prioritize next to learn the most: denser checkpoint sweeps, more seeds near the threshold, or stronger algorithm comparisons.”

## Sources

- Stage 1 project brief: [README.md](/dlabscratch1/samaier/projects/SDPO-safety/experiments/stage1/README.md)
- RL Excursions project page: https://rl-excursions.github.io/
