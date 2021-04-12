####################################################
# Original Screenplay to Vectorized Character Data #
####################################################
1. Run [screenplay_to_character_dialogues.py] (just need original screenplay to fulfil a few requirements listed at the top of that script)
EXPECTS: screenplays/rawdocs/<original screenplay doc>
CREATES: screenplays/formattedtxts/<character dialogues txt>

2. Run [separate_characters_and_dialogues.py] to get characters.txt and dialogues.txt files
EXPECTS: screenplays/formattedtxts/<character dialogues txt>
CREATES: screenplays/characters_and_dialogues/<characters txt>, screenplays/characters_and_dialogues/<dialogues txt>

3. Run [Screenplay Character Labeling - Using Subtitles File.ipynb] to label subtitles in .srt file with characters where possible.
EXPECTS: screenplays/characters_and_dialogues/<characters txt>, screenplays/characters_and_dialogues/<dialogues txt>, subtitles/<.srt file>
CREATES: character_subtitles/<character subtitles txt>

3.1 (Optional) For best results, go over the character_subtitles/<character subtitles txt> file and manually replace Nones with appropriate characters/OTHER.

4. Run [Algorithm Speech-to-Text - Using Subtitles With Characters File.ipynb] to create timed words Data
EXPECTS: character_subtitles/<character subtitles txt>, movies_audio/<movie>/<segment audio files>
CREATES: movies_transcripts_with_characters/<movie>/<segment timed words with character labels files>

5. Run [Vectorize Transcripts with Characters.ipynb] to create word-level one-hot character features
EXPECTS: movies_transcripts_with_characters/<movie>/<segment timed words with character labels files>, character_subtitles/<character subtitles txt>
CREATES: movies_word_level_character_features/<movie>/<segment features npy files>

#############################
# Stacking Code and Metrics #
#############################
NOTE: Everything stacking-related is in the stacking_experiments directory
* brain_prediction_pipeline/make_fold_weights.py: Computes and stores weights per fold with ridge and 12-fold CV for a given feat_type (elmo/speaker) and subject
* brain_prediction_pipeline/make_stacked_predictions.py: Uses the fold weights from above to perform stacking and stores a dictionary containing
individual and stacked r-squared values, correlations with ground truth, and average stacking weights across all folds.