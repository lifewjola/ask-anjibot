def normalize_text(text):
    clean_text = ''.join(char.lower() for char in text if char.isalnum() or char.isspace())

    words = clean_text.split()

    normalized_words = []
    for word in words:
        word = word.rstrip("'s") # Remove possessive apostrophe
        normalized_words.append(word)
    return set(normalized_words)
