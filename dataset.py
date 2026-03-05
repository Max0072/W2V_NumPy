import pandas as pd

# Returns text for training in string format
def get_dataset_text():
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet',
              'test': 'data/test-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/afmck/text8/" + splits["train"])
    text = df["text"].iloc[0]
    num_words = 10000
    text = " ".join(text.split()[:num_words])
    print("--- Dataset loaded! ---")
    return text
