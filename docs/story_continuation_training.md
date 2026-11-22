# Training for story openings

The app serves a local Qwen-based Transformer fine-tuned to continue an
opening. The original character Transformer runs are historical experiments.
The starting weights already learned English during upstream pretraining.

## Data and objective

The fine-tuning script uses the pinned, deduplicated TinyStories and
SimpleStories splits documented in [expanded story training](expanded_story_training.md).
It splits a story after its first sentence and presents that sentence as the
user opening. The loss scores subsequent story text only. Each run uses
48 training examples per source, eight validation examples and six held-out
examples. Story IDs are checked for exact overlap across these sets. This
small transfer-learning experiment is distinct from full-corpus pretraining.

The base weights stay frozen. Rank-eight updates to attention projections
in the last eight decoder layers are the only trained parameters. The script
evaluates every 24 updates and saves the adapter with the lowest mean
per-example continuation cross-entropy. Six additional hand-written openings
include the two examples reported by the user. Test prompts are generated
after checkpoint selection.

## Reproduction

Install the `story` and `web` extras with the lockfile. Restore the pinned
local model files, then run `fine_tuning/train_story_continuation.py` from the project
root. The script refuses to reuse an existing run directory. Use `--larger-base`
for the separate 1.5B experiment. Each run writes its exact samples to
`runs/story-continuation-v*/results.json`. Review the examples for English
grammar, whether characters and situation carry forward, and whether the ending
follows from the preceding events. Validation loss is only a token prediction
measure.

Base weights and provenance:
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
and [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
Both cards list Apache 2.0. Registry entries pin the adapter, tokenizer,
base-file inventory and training record. Inference reads local files.

## Results and application release

Both local MPS runs finished 96 training updates. The 0.5B adapter lowered
mean validation loss from 1.9434 to 1.8518. The 1.5B adapter lowered it from
1.6766 to 1.5852 and was selected at update 96. The final 1.5B review tested
an instruction that asks the model to address the opening's key event early
and keep the characters and situation consistent. The opening text is still
sent as its own message. If generation repeats the entire opening verbatim,
the app displays the remaining continuation.

The app offers one model, `transformer-story-continuation-v2`. The actual
continuations are readable English and several retain the specific situation.
Other samples still introduce odd details, leave the central question unresolved
or end at a token limit. The sample set supports improvement over the original
character model, with quality varying by prompt and seed. Story writing is
open-ended, so the held-out source continuation is a reference for review.

Training outputs are in `runs/story-continuation-v2/` and the release evidence
is in `artifacts/story-continuation-v2/`. The previous registry was archived
before the app switched.

The live app returned three nonempty English continuations on MPS from the
sole registered checkpoint. All three reached an end marker. The fresh
library-kitten prompt retained Nina and the kitten and resolved their encounter.
The ribbon prompt continued the bird's action. The box sample read fluently
and offered a conclusion, although it left the tapping unexplained. The exact
responses are in `artifacts/story-continuation-v2/live-check.json`.
The Python regression suite passed 480 tests during development. The final
JavaScript suite passed 25 tests, and the pinned 1.5B artifact and real app
requests passed after adjusting the weight-file size bound.
