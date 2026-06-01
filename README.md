# fsrs-autoresearch
Improving FSRS-7 with Claude. This repo was inspired by AlphaEvolve and [Andrej Karpathy's "autoresearch" repo](https://github.com/karpathy/autoresearch). Huge thanks to [1DWalker](https://github.com/1DWalker), without [his code](https://github.com/open-spaced-repetition/fsrs-gpu-benchmark) this wouldn't be possible!

Below is a graph that shows how log loss (average across 3000 Anki users) changed as Claude kept working on it.

(graph.png)

You can find more details in history.md and history.jsonl. Some notes:
1) Not all proposals came from Claude, for example, using 4 different values of learning rate was my idea. Though I discarded it later.
2) What counts for the complexity score has been changed multiple times, so that number is mostly worthless.
3) This repo also contains code that is used to find optimal default parameters for FSRS-7, both optimal user-facing default parameters and optimal "starting point for Adam" parameters (previously, the same parameters were used for both). It also contains an automated hyperparameter tuner.
