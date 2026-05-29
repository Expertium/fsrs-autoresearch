You are participating in an AlphaEvolve/Karpathy's "autoresearch"-style automated improvement loop. More specifically, you are working on improving FSRS-7, a spaced repetition algorithm.
Your task is to propose code changes to improve average log loss on a dataset of 3000 Anki users. Your task is proposal-generation, not writing the patch; a smaller model will implement your proposals as a git patches and they will be applied and benchmarked.
You will be shown train.py, which contains both the FSRS-7 model class and the training loop; as well as the history of your latest proposals and some diagnostics to help you better understand how optimization is going with the current train.py code.

train.py:
	

Diagnostics:
	Code complexity score (AST_node_count + 40 * cyclomatic_complexity_score): __
	Average log loss: __
	Log loss on Again (rating=1) only: 
	Log loss on Hard (rating=2) only: 
	Log loss on Good (rating=3) only: 
	Log loss on Easy (rating=4) only: 
	Log loss on delta_t<1 day reviews: 
	Log loss on delta_t>=1 day reviews: 
	
	Per-parameter diagnostics
	w[__]:
		bounds: (__, __)
		p01/median/p99: __ / __ / __
		Hit lower bound on __% of users
		Hit upper bound on __% of users
		Mean gradient across all epochs: 
		Mean gradient at the last epoch: 


Latest proposals (at most 25):
	summary, log loss before=__, log loss after=__, threshold=__, improvement=__, status: rejected; reason: improvement<threshold; code complexity score before: __, code complexity score after: __, % change: __
	summary, log loss before=__, log loss after=__, threshold=__, improvement=__, status: rejected; reason: code complexity score increase>5%; code complexity score before: __, code complexity score after: __, % change: __
	summary, log loss before=__, log loss after=__, threshold=__, improvement=NaN, status: rejected; reason: git apply failure; code complexity score before: __, code complexity score after: NaN, % change: NaN
	summary, log loss before=__, log loss after=__, threshold=__, improvement=__, status: accepted; reason: improvement>=threshold; code complexity score before: __, code complexity score after: __, % change: __


You can make the following changes:
1) Add new parameters/constants to or remove parameters from FSRS formulas; modify parameter ranges, their default values and the sigmas for the L2 penalty
2) Add new formulas and new state variables (currently there are 2: S, memory stability, and D, difficulty)
3) Modify the training loop: learning rate, betas, lr scheduler, SGD optimizer, recency weighting, etc. (except for n_epoch)

However, there are also constraints that you MUST obey:
1) Forgetting curve must be monotonic with respect to time delta_t, for any parameter values within the allowed ranges
2) w[0] <= w[1] <= w[2] <= w[3], aka for initial stabilities the following must be true: S0(rating=1) <= S0(rating=2) <= S0(rating=3) <= S0(rating=4)
3) stability_after_review(rating=1) <= stability_after_review(rating=2) <= stability_after_review(rating=3) <= stability_after_review(rating=4)
4) Higher D = slower growth of S. Higher values of D cannot make memory stability grow faster than lower values
5) Do not change the number of epochs. While increasing them will most likely improve log loss, let's focus less on brute force and more on clever architectural changes. And do not rename n_epoch
6) Do not skip users, do not change time series splits, and do not change filters in review preprocessing
7) Anything non-deterministic needs to be seeded, to ensure reproducibility
8) Do not add data leakage from the test/evaluation set into the training set (the golden rule of ML: do not train on the test set)
9) Do not propose changes that exploit the way .jsonl files are saved, user IDs, rounding of the loss value, limitations of floating-point precision, or any other property unrelated to spaced repetition. Improvements must come from the FSRS-7 model and/or training loop
10) Training loop changes are allowed, but the evaluation metric remains average log loss. Changing training (not eval) metric to, for example, focal loss or BCE+some_auxiliary_loss is allowed
!!! ALL of your proposals must satisfy ALL of the above constraints !!!
If you change the training loss, explain why using different metrics for training and evaluation should improve evaluation log loss.
Also, code complexity is evaluated using the following formula: score = AST_node_count(train.py) + 40 * cyclomatic_complexity_score(train.py). Coefficient of 40 was chosen so that both AST node count and cyclomatic complexity contribute to the final score roughly equally. Each proposal must not increase this score by more than 5%, ideally <2%. This is something that you have to think about in advance when making proposals. 
It's entirely possible that you will underestimate how much extra complexity is being added, and your proposal will be rejected automatically based on the complexity score, so be conservative.

You should output a Python dictionary in the following format:
{1: {'summary': 'short string', 'full_description': 'long string', 'threshold': 0.0002}, 2: {...}, 3: {...}, 4:{...}, 5: {..}}

'summary' must contain a concise description of your idea, no longer than 30 words. It must convey the idea with as little loss of information as possible; a future instance of you will be provided with a list of successful and failed attempts, and that's what it will see (plus log loss values and whether the change was rejected or accepted).
Example of a bad 'summary': different lr for different groups of params
Example of a good 'summary': w[0:4]: lr=1e-2, w[4:7]: lr=4e-2, w[7:27]: lr=4e-2, w[27:]: lr=1e-2

'full_description' is a specification for another, smaller model. That model will write a patch for train.py and it will be applied via git apply.
Make it as clear and unambiguous as possible, so that the smaller model understands exactly what it must do.
The patch-writing model will see only train.py and your 'full_description'. It may not infer your intent correctly unless you specify exact parameter indices, formulas, clamps, initialization values, L2 sigmas, affected functions, etc.
Do not say “use a different recency weighting formula” without giving the exact formula. Do not say “make difficulty nonlinear” without giving the exact replacement expression.
Each 'full_description' should include:
- target function/class locations in train.py
- new parameters/new state variables, if any
- exact default values for new/changed parameters
- exact clamp ranges for new/changed parameters
- exact sigma values for L2 penalty for new/changed parameters (note: L2 penalty only slightly reduces log loss, so you don't have to be afraid of messing sigmas up too badly)
- exact hyperparameter values for new/changed hyperparameters
- tips to avoid indexing mistakes if the number of FSRS parameters has changed
- tips to implement the proposal in a way that doesn't add a lot of extra complexity aka doesn't increase the complexity score by >5%
- when adding new state variables: how to handle unpacking in places that expect 2 state variables, such as 'stabilities, difficulties = outputs[...].transpose(0, 1)'
- how to preserve all constraints
'full_description' must be no longer than 500 words.

Threshold is the difference between the current best log loss value and the new value after the change, old_log_loss - new_log_loss. If old_log_loss - new_log_loss < threshold, the proposal is rejected. Here's how to choose the value:
1) 0.0010 per every new state variable
2) 0.0002 per every new parameter
3) 0.0001 flat for every other change, regardless of how big it is
Example 1: if you are adding a new state variable (for example, moving average of ratings) and 3 more parameters, threshold=0.0010+3*0.0002=0.0016.
A new trainable scalar in 'self.w' counts as a new parameter.
A new fixed constant outside of 'self.w' does not count as a new parameter.
A new variable returned by 'step' in 'torch.stack' counts as a new state variable.
Changing the training loop without adding parameters/state variables has threshold=0.0001.
Changing a formula without adding parameters/state variables has threshold=0.0001.
The minimum value for threshold is 0.0001 for changes that do not introduce any new parameters.
Removing some parameters while simultaneously adding new parameters reduces the threshold by 0.0002 per each removed parameter, but not below 0.0002. In other words, simplifying formulas to use fewer parameters must also yield an improvement.
Example 2: if you are removing 3 parameters and adding one new parameter, threshold=max(-3*0.0002+0.0002, 0.0002)=0.0002.

Do not lump unrelated ideas into the same proposal. For example, "replace cosine annealing with a different lr scheduler" and "replace Adam with NAdam" are unrelated, so they must be 2 separate proposals. Another example: "change D formulas" and "change forgetting curve formula" should also be 2 different proposals.
You can make 5 completely different proposals, or 5 variations of the same idea. Balancing exploration and breadth of changes is up to you. Make sure to take a good look at the history so far (see above). Do not repeat past mistakes, but also do not be afraid of trying again with a slightly different variation.
Prefer proposals with good expected improvement per new parameter. Do not add new parameters unless the expected gain plausibly exceeds 0.0002 times the number of new parameters. Ideally, we want maximum log loss reduction with minimum extra complexity.
If some change worked (improvement exceeded the threshold), you can push it further and focus on squeezing everything out of that change, or explore other changes.

Before outputting the dictionary with proposals, silently verify the following for every proposal:
- Does it preserve monotonicity in the forgetting curve with respect to delta_t?
- Does it preserve w[0] <= w[1] <= w[2] <= w[3] (initial stabilities)?
- Does it preserve monotonicity of stability_after_review with respect to rating?
- Do higher values of D still make S grow slower?
- Does it avoid changing n_epoch?
- Does it avoid changing filtering/time series splits/review preprocessing?
- Does it avoid "gaming" the loss function?
- Are there no leakages from the test/eval set to the train set?
- Are all new stochastic operations seeded?
- Are all new parameters properly initialized, clipped, regularized with sigma (for L2 penalty), and included in init_w?
- Is threshold computed properly?
- Does it avoid adding too much complexity at once?

Now please output the dictionary.
!!! Your output must be parseable by ast.literal_eval in Python  !!!
Use integer keys 1 through 5.
Do not add # comments.
Escape newlines as \n if needed.
Structure of the dictionary:
- 'summary': string, <=30 words per each proposal
- 'full_description': string, <=500 words per each proposal
- 'threshold': float, >=0.0001