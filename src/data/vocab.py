"""
Vocabulary building utilities for metadata encoding.
"""


def build_vocabs(df):
    """
    Build vocabulary dictionaries for categorical metadata columns.

    Args:
        df: pandas DataFrame with 'Habitat' and 'Substrate' columns

    Returns:
        tuple: (habitat_vocab, substrate_vocab)
            - habitat_vocab: dict mapping habitat string -> index
            - substrate_vocab: dict mapping substrate string -> index

    Note:
        UNK index = len(vocab), NOT 0 (0 is a valid class)
    """
    # Handle column name variations (Habitat vs habitat)
    habitat_col = 'Habitat' if 'Habitat' in df.columns else 'habitat'
    substrate_col = 'Substrate' if 'Substrate' in df.columns else 'substrate'

    # Build habitat vocabulary from unique non-null values
    habitat_values = df[habitat_col].dropna().unique()
    habitat_vocab = {v: i for i, v in enumerate(sorted(habitat_values))}

    # Build substrate vocabulary from unique non-null values
    substrate_values = df[substrate_col].dropna().unique()
    substrate_vocab = {v: i for i, v in enumerate(sorted(substrate_values))}

    print(f"Built vocabularies:")
    print(f"  Habitat: {len(habitat_vocab)} categories")
    print(f"  Substrate: {len(substrate_vocab)} categories")

    return habitat_vocab, substrate_vocab


def get_vocab_sizes(habitat_vocab, substrate_vocab):
    """
    Get vocabulary sizes including UNK token.

    Returns:
        tuple: (habitat_size, substrate_size) including UNK
    """
    return len(habitat_vocab) + 1, len(substrate_vocab) + 1