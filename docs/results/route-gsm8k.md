# Does the local model know when it is wrong?

Made by `scripts/route_eval.py`. Do not edit by hand.

Qwen3 0.6B running in Ember answered 1319 GSM8K test questions and got 62.1% right. For each answer, Ember scored how unsure the model was from its own logits. A good signal scores the wrong answers as more unsure.

| Signal | AUROC (95% CI) | Wrong answers caught by escalating 30% (95% CI) |
|---|---|---|
| mean entropy | 0.740 [0.711, 0.768] | 52.6% [49.4%, 55.8%] |
| max entropy | 0.740 [0.712, 0.768] | 51.0% [47.8%, 54.5%] |
| lowest margin | 0.592 [0.558, 0.622] | 36.2% [33.2%, 39.9%] |
| share of tokens with margin under 0.5 | 0.698 [0.668, 0.727] | 47.4% [44.0%, 50.5%] |
| mean entropy, first 32 tokens only | 0.646 [0.614, 0.676] | 41.6% [37.9%, 44.8%] |
| question length (baseline) | 0.687 [0.658, 0.717] | 45.6% [42.6%, 49.2%] |
| random (baseline) | 0.478 [0.446, 0.510] | 27.6% [24.5%, 30.7%] |

AUROC is the chance that a wrong answer scores as more unsure than a right one; 0.5 is no better than chance. Escalating 30% at random catches about 30% of the wrong answers.

## Better than question length?

Longer questions are harder, so question length alone is a real baseline. The best of the 5 logit signals, max entropy, picked on this same data, differs from it by +0.053 AUROC (95% CI [+0.023, +0.094], paired bootstrap that picks the best signal again in every resample). The whole interval is above zero, so the logits find mistakes better than length alone does.

## Held out

The signal (max entropy) and its threshold were picked on the first 659 questions to escalate 30% (it escalated 30.0% there), then used unchanged on the other 660. There it escalated 28.9% and caught 51.2% [45.0%, 57.5%] of the local model's wrong answers, against about 29% for escalating at random. Accuracy on the 71.1% kept local was 73.3% [69.3%, 77.3%], against 61.2% for keeping everything local.

Only max entropy and lowest margin were candidates here, because they never go down as an answer goes on, so `ember route` can stop the moment one crosses the threshold and still make the same call as the full answer. The threshold sits halfway between two logged scores, so rounding in the log cannot flip a question. To run it:

```sh
ember route models/qwen3-0.6b --signal "max entropy" --threshold 2.272105 "your question"
```

## Limits

- No cloud model was called, so this measures whether the signal finds the local model's mistakes, not what a combined system would score.
- One model, one benchmark, greedy decoding. Short math answers may make uncertainty easier to read than open-ended chat. GSM8K is public, so it may be in the model's training data.
- 121 answers hit the 384 token cap and 250 never said "The answer is"; those were scored on their last number. Both count toward the local accuracy above.
- f32 weights. With --q8 the entropies shift a little, so this threshold is for f32.
- Speed: 17.29 s per question on average with 6 threads, 232.0 tokens per answer, on a machine that was not kept idle, so this is a rough figure, not a benchmark.
