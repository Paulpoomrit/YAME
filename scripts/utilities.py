import re

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

        with open(dest_path, 'w') as tsv:
            tsv.write(content_new)


clean_and_convert_to_tsv('data/test_data/test_corrupt.csv', 'data/test_data/test_not_corrupt.tsv')