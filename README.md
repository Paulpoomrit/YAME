# YAME(**Y**our **A**i slop bores ME) Model #
## Classifier for AI-Generated Text | LING 450 Final Project ##
---
#### YAME is a Random Forest Classifier that label text as either AI-generated (AI) or human-created (H) ####
---

## Directory Guide ##

- Under the script folder, there are two scripts for:
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
- More information about the two scripts/ features list can be found under the documentation folder.
- The main code to run the model is located at `src/yame.py`
- The random forest classifier is saved and can be loaded at `model/yame.pkl`

---

## To Run `yame.py`
- Make sure Python 3 and Java are installed (this is required for the grammar checker module).
- Make sure to set `export PYTHONPATH={Path\to\script}` pointing to the script folder that contains all the utility scripts.
- Run `src/yame.py` in the terminal
### Sample Output
<img width="678" height="589" alt="image" src="https://github.com/user-attachments/assets/5fb82644-d233-4e40-bec3-d56bdad108be" />
