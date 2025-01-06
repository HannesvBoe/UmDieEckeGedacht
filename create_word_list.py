import pandas as pd

def read_word_list(use_long_list = False):
    if use_long_list:
        # Load the word list from a CSV file
        word_list = pd.read_csv("data/Filtered_Wordlist_German.csv", header=True)["Wort"].dropna().tolist()
    else:
        word_list_input = pd.read_csv(
            "data\\Wortliste.csv",
            delimiter=" ",
            encoding="ISO-8859-1",
            skiprows=43,
            header=None,
            names=["Wort", "Häufigkeit", "Eigenschaft"]
        )

        # Filter rows based on specific conditions
        filtered_word_list = word_list_input[
            ((word_list_input["Häufigkeit"] < 15) | (word_list_input["Wort"].str.len() >= 5))
            & (word_list_input["Häufigkeit"] < 21)
            ]

        # Convert the DataFrame to a list, excluding NaN values
        word_list = filtered_word_list[filtered_word_list["Wort"] != "nan"]["Wort"].dropna().tolist()

        # Expand words with parentheses or commas into individual words
        for idx, word_item in enumerate(word_list[:]):  # Use a slice to avoid modifying the list while iterating
            if "(" in word_item:
                words_split = word_item.split("(")
                for endung in words_split[1].split(","):
                    word_list.append(words_split[0] + endung.strip(")"))
                word_list.remove(word_item)
            elif "," in word_item:
                for word in word_item.split(","):
                    word_list.append(word.strip())
                word_list.remove(word_item)

    print("Word list generated. Number of words is ", len(word_list))
    return word_list


# Function to replace special German characters with their ASCII equivalents
def replace_special_characters(text):
    replacements = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss',
        'Ä': 'Ae',
        'Ö': 'Oe',
        'Ü': 'Ue'
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


# Function to validate if a word meets certain criteria
def isValidWord(text, max_length):
    return (
            2 <= len(text) <= max_length and
            text.isalpha() and
            not text[1:].isupper()
    )


# Function to create dictionaries of words categorized by length and characters
def create_word_char_dict(word_list_unfiltered, max_word_length):
    word_dict_by_length = {i: [] for i in range(2, max_word_length + 1)}
    word_dict_by_char = {}
    list_of_all_words = []

    for word_item in word_list_unfiltered:
        word = replace_special_characters(word_item)
        if isValidWord(word, max_word_length):
            word = word.lower()
            word_dict_by_length[len(word)].append(word)
            list_of_all_words.append(word)

            current_dict = word_dict_by_char
            for char in word:
                if char not in current_dict:
                    current_dict[char] = {"Word_Count": 0}
                current_dict = current_dict[char]
                current_dict["Word_Count"] += 1
            current_dict["Word"] = word

    return list_of_all_words, word_dict_by_length, word_dict_by_char