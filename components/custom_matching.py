from components.process_text import normalize_text

exceptions = ["of", "the", "and", "I", "in", "to", "close", "i", "o", "dr",  "study"]

def custom_similarity(text, query, exceptions=exceptions):

    text_words = normalize_text(text)
    query_words = normalize_text(query)

    matching_sequences = set()
    for word in text_words:
        if word in query_words and word not in exceptions:
            matching_sequences.add(word)

    # Return the number of matching sequences
    return len(matching_sequences)
