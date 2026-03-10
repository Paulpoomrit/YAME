# LING 450 | Term Project

> Under construction!!

---

Here’s the progress so far

- Under the script folder, there are two partially done scripts for:
    - Extracting features [`feature_extractor.py`]:
        - An Enum class `Feature` is defined for feature types for convenience.
        - `regex_dict`, taking the enum as the key, is defined for features that require pattern matching.
        - `ai_vocab`  is a list containing words typically associated with/ that be in overepresented by AI.
        - `count_all_words_in_list` : a utility function for counting the number of occurrences of words in a list in a piece of text.
        - `extract_group_a` : use to extract features that belong to group A (see below for more info about the three groups).
        - `extract_features`: the main function that will be called by the client.
    - Collecting data [`data_collector.py`]:
        - `process_doc` : takes in a document path and:
            - Add them to the csv file, labelled as H for human
            - Call two other utility functions to get their AI counterpart for that particular text, labelled them as A for AI.
        - The original is to use the latest model of ChatGPT; however, this might be…financially unrealistic, so I might replace this part with another open-sourced model.
        - I may or may not upload my api key there, pls don’t hack me T0T.
- More information about the two scripts/ features list can be found under the documentation folder (these are drafts that will hopefully go in the final paper!).

---
