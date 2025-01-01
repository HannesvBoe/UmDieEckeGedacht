import numpy as np
import pandas as pd
import os
from datetime import datetime
import re
import csv
import time
import random

# Load the word list from a CSV file
#word_list = pd.read_csv("data/wordlist-german.txt", names=["Wort"], header=None)["Wort"].dropna().tolist()

word_list_input = pd.read_csv(
    "data/Wortliste.csv",
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
            text[0].isupper() and
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


# Function to get a relevant dictionary based on the beginning of the word
def get_relevant_dict(relevant_char_dict, word_beginning):
    current_dict = relevant_char_dict
    for char in word_beginning:
        if char in current_dict:
            current_dict = current_dict[char]
        else:
            return {}
    return current_dict


# Function to find potential next words for a riddle
def find_potential_next_words(riddle_inst, relevant_char_dict, word_beginning, row_num, sum_possible_row_words):
    next_possible_chars = {}
    current_column_next_chars = set(get_relevant_dict(relevant_char_dict, word_beginning).keys()) - {"Word",
                                                                                                     "Word_Count"}
    current_row_next_chars_dict = riddle_inst.current_row_char_dict[row_num]

    for next_row_char in current_row_next_chars_dict:
        if next_row_char in current_column_next_chars:
            next_possible_chars[next_row_char] = current_row_next_chars_dict[next_row_char]["Word_Count"]

    if not next_possible_chars:
        return

    next_possible_char_list = sorted(next_possible_chars, key=next_possible_chars.get, reverse=True)

    for next_char in next_possible_char_list:
        new_beginning = word_beginning + next_char
        new_column_dict = get_relevant_dict(relevant_char_dict, new_beginning)
        sum_next_possible_row_words = sum_possible_row_words + next_possible_chars[next_char]

        if "Word" in new_column_dict:
            new_word = new_column_dict["Word"]
            riddle_inst.current_relevant_words[new_word] = sum_next_possible_row_words

        if row_num < riddle_inst.num_rows - 1:
            find_potential_next_words(riddle_inst, relevant_char_dict, new_beginning, row_num + 1,
                                      sum_next_possible_row_words)
