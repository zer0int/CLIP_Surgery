# Run this after you've run the main script, and there are CLIP opinions in the "TOK" folder.
# Produces a list of record CLIP craziest words. :-)

import os

def find_long_words_in_files(directory, min_length=15):
    long_words = set()  # Use a set to avoid duplicate words

    # Walk through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Check if the file is a .txt file
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                words = content.split()  # Split content into words
                for word in words:
                    if len(word) > min_length:  # Check if the word length is more than 15 characters
                        long_words.add(word)

    return long_words

# Specify the directory to search in
directory_path = 'TOK'

# Find long words
long_words = find_long_words_in_files(directory_path)

# Save the long words to a file, sorted from longest to shortest
with open('longest_clip_word_records.txt', 'w', encoding='utf-8') as output_file:
    for word in sorted(long_words, key=len, reverse=True):  # Sort by word length, longest first
        output_file.write(word + '\n')

print(f"Found and saved {len(long_words)} words longer than 15 characters.")

