# Task 1 Failure Analysis

The examples below come from the final generated sample for run `20260925_163912` using the best checkpoint, prompt `Once upon a time`, and temperature 0.8.

## Failure 1

**Generated text:** "Her mommy said \"Wait and bury!\""

**Failure type:** Loss of coherence / incorrect word choice

**Observation:** The sentence is grammatically shaped like dialogue, but "bury" does not fit the situation described in the story. The model learned common sentence patterns but sometimes chooses a locally plausible character sequence that gives the sentence the wrong meaning.

## Failure 2

**Generated text:** "she had to do a something special."

**Failure type:** Broken grammar

**Observation:** The phrase uses both "a" and "something" together, which makes the sentence ungrammatical. This shows that the character-level model can produce readable sentences while still making small grammatical errors over longer sequences.

## Failure 3

**Generated text:** "She wanted to help the sharp stone"

**Failure type:** Loss of coherence

**Observation:** Earlier, the sharp stone is presented as something Lily encountered, but the story suddenly treats the stone as something that needs help. The sentence itself is readable, but the meaning does not follow the previous context well.
