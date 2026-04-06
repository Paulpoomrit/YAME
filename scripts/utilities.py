import re
import csv
import pandas as pd
from feature_extractor import extract_features

def clean_and_convert_to_tsv(csv_path: str, dest_path: str) -> None:

    with open(csv_path) as f:
        content = f.read()
        content_new = re.sub(pattern=r",AI,",
                             repl=r"\tAI\t",
                             string=content)
        content_new = re.sub(pattern=r",H,",
                             repl=r"\tH\t",
                             string=content_new)

        content_new = re.sub(pattern=r",{2,}",
                             repl="",
                             string=content_new)

        content_new = re.sub(pattern=r"text,label",
                             repl="text\tlabel",
                             string=content_new)

        with open(dest_path, 'w') as tsv:
            tsv.write(content_new)


def extract_features_and_save(data_path: str, dest_path: str) -> None:

    #df = pd.read_csv(data_path, sep='\t', index_col=False) # for tsv
    df = pd.read_csv(data_path, index_col=False)
    # print(df['text'])

    feature_vectors = []
    count = 0
    for r in df.itertuples(index=False):
        count += 1
        print(f'Extracting features for sentence: {count} / {len(df)} ')
        feature_vectors.append(extract_features(r.text))

    # print(feature_vectors)

    feature_names = ["emdash",
                     "emoji",
                     "bold",
                     "title_case",
                     "list",
                     "curly_quotation",
                     "table",
                     "template",
                     "grammar",
                     "vocab",
                     "neg_parallelism",
                     "rule_of_three",
                     "basic_copulative",
                     "eleg_variation",
                     "subject_line",
                     "summary"
                     ]
    features_df = pd.DataFrame(feature_vectors, columns=feature_names)

    df = pd.concat([df, features_df], axis=1)
    print(df)
    df.to_csv(dest_path, sep='\t')


def get_feature_vectors(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep='\t', index_col=False)
    # print(df['text'])

    feature_vectors = []
    count = 0
    for r in df.itertuples(index=False):

        print(f'Extracting features for sentence: {count} / {df.size} ')
        feature_vectors.append(extract_features(r.text))
        count += 1

    # print(feature_vectors)
    df['vector'] = feature_vectors
    return df




#clean_and_convert_to_tsv('data/test_data/test_corrupt.csv', 'data/test_data/test_not_corrupt.tsv')
extract_features_and_save('data/now_data_subset/now_data_subset.csv', 'data/now_data_subset/now_data_subset_w_features.csv')
