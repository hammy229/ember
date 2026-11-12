# Demo Script

Target: about 2 minutes for the demo video. Beats marked (optional) can be cut to
hold that. Each beat says what to do on screen and what to say. The lines are a
guide, so say them in your own words.

Everything is recorded on the live site, [ember-rosy-three.vercel.app](https://ember-rosy-three.vercel.app).
Nothing on screen is edited. Every number you say is in the README results
tables, so don't round them differently or add new ones.

## 0. Setup (before recording)

- Chrome, window 1600 px wide, zoom 100%, every other tab closed, notifications
  off (macOS Focus mode).
- Open the site once and press **Load the model (32 MB)** in the writing
  section, then reload the page. The model is now cached, so it loads in about a
  second on camera instead of downloading.
- Quit anything heavy (Docker, other browsers). Training and writing speed are
  real and slow down when the CPU is busy.
- Record with QuickTime (File > New Screen Recording, pick the Chrome window)
  and a separate voice track, or narrate live. Do a 10 second test first and
  play it back to check the mic.
- Training takes about 2 to 3 minutes for all 1000 steps. Start it early (beat
  5) and cut back to it, or press it before recording and film it near the end.

## 1. The hook (10s)

**Screen:** the home page. Move the mouse slowly over a few coals in the grid so
their numbers show.

**Say:** "This is Ember, an LLM engine I wrote from scratch in C++, with no ML
framework. It compiles to WebAssembly, so everything here is running in this
browser tab. This grid isn't an animation. Every coal is a real attention weight
the engine is computing right now."

## 2. It writes, and shows where it was unsure (25s)

**Screen:** click **Watch it write** (or scroll to the writing section). Press
**Load the model**, then **Write**. Let the story fill in.

**Say:** "This is a small 15 million parameter Llama, running on my machine with
no server. Each token is shaded by how unsure the model was. Pale means it was
sure, dark means a lot of next words looked about as good."

**Screen:** hover one dark token so its numbers show, then drag the **Hand off at
entropy** slider so the hand-off marker moves.

**Say:** "That signal is free when you own the engine, and it's what a
local-first router needs. Past this threshold, a router would stop here and hand
the request to a bigger model."

## 3. The routing result (20s)

**Screen:** click the routing section in the side rail ("It can tell when it is
likely wrong"). Hold on the bars and the accuracy numbers.

**Say:** "I tested that idea for real on Qwen3 0.6B, on all 1319 GSM8K math
questions. The highest per-token entropy in an answer spots the model's wrong
answers with an AUROC of 0.740, against 0.687 for just using question length.
With the threshold set on one half and tested on the other, escalating 28.9% of
questions caught about half the mistakes, and accuracy on the rest went from
61.2% to 73.3%."

**Be clear:** no cloud model was called. Say "escalating would catch", not "the
system scored 73%".

## 4. It matches Hugging Face, and here is where it stands (20s)

**Screen:** the Qwen section ("A 0.6B model, next to llama.cpp"). Show the
Hugging Face match, then the llama.cpp bars.

**Say:** "Getting the math exactly right came first. On Qwen3, Ember's logits are
within 1e-4 of Hugging Face's, with the same next token at every position I
tested. And it's not faster than llama.cpp. With 8-bit weights it runs at 46% of
llama.cpp's speed writing, on CPU, and I publish that number as measured."

## 5. It can learn, not just run (20s)

**Screen:** the training section. Press **Train from scratch (1.1 MB of text)**.
Show the sample text as random letters, then cut to later when the loss line has
dropped and the sample looks like Shakespeare.

**Say:** "Ember also has its own autograd. This is a 111 thousand parameter
transformer starting from random weights, learning Shakespeare one character at
a time, in this tab. The held-out line is text it never trains on, so it falling
means it's learning the language, not memorizing."

## 6. Look inside (optional, 15s)

**Screen:** the "Look inside" section. Press **Look inside**, click a few layer
and head buttons so the heatmap changes.

**Say:** "It can also record what it's doing on the way: every attention head's
weights, and what the model would predict if it stopped after each layer."

## 7. Try to break it (optional, 10s)

**Screen:** the softmax section. Press **Add 1000 to every score**. The chances
don't change.

**Say:** "A naive softmax overflows here. Ember subtracts each row's largest
score first, so the chances hold."

## 8. The proof (15s)

**Screen:** switch to the GitHub repo
([github.com/hammy229/ember](https://github.com/hammy229/ember)). Scroll the
README results table slowly, and point at the "How it is checked" column.

**Say:** "Every number I said links to the script or test that produced it, with
the machine and commit. There are over 400 tests, fuzzing on every push, and
sanitizers in CI. The site and code are both public."

## 9. Wrap-up (5s)

**Screen:** back to the home page grid.

**Say:** "I'm Arham Doshi. The link to try it is in the description."

## Things not to say

- Don't say it's faster than llama.cpp or NumPy. It isn't, and the site says so.
- Don't call the 73.3% a system score. No bigger model was called.
- Don't say the tiny story model is smart. It's small on purpose. The engine is
  the point.

## After recording

- Watch it once end to end with sound before posting.
- Upload to YouTube as unlisted first, then add the link to the README and the
  launch post.
