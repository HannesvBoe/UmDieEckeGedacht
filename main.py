import numpy as np
import os
from datetime import datetime
import csv
import time


# %%
from create_word_list import *

word_list = read_word_list(use_long_list=False)

# Initialize the puzzle parameters
num_rows = 10
num_cols = 20
max_word_length = max(num_cols, num_rows)

# Create the character dictionaries
list_char_dicts = [create_word_char_dict(word_list, i)[2] for i in range(max_word_length + 1)]
list_of_all_words, word_dict_by_length, word_dict_by_char = create_word_char_dict(word_list, max_word_length)


# %%
# Define the main class for the puzzle
class UmDieEckeGedacht(object):
    def __init__(self, num_rows, num_cols, preferred_words=[]):
        # Initialize class attributes
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.max_word_length = max(num_rows, num_cols)
        self.char_grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        self.row_words = [[] for _ in range(num_rows)]
        self.col_words = [[] for _ in range(num_cols)]
        self.current_index = [0, 0]
        self.current_relevant_words = {}
        self.used_words = []
        self.preferred_words = sorted([s.lower() for s in preferred_words], key=len)
        self.start_time = time.time()

        # Initialize row character dictionaries
        self.current_row_char_dict = [list_char_dicts[self.num_cols] for _ in range(num_rows)]

    def save_to_csv(self, filename):
        filepath = "Riddle_Results\\" + filename
        print("Save riddle!")
        # Check if the file already exists
        if os.path.exists(filepath):
            # Get the current date and time
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Add the current date and time to the beginning of the filename
            filename = f"Riddle_Results\\{current_time}_{filename}"
            print("New Filename: ", filename)

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write char_grid
            writer.writerow(['char_grid'])
            for row in self.char_grid:
                writer.writerow(row)

            # Write a separator
            writer.writerow([])

            # Write row_words
            writer.writerow(['row_words'])
            for row in self.row_words:
                writer.writerow(row)

            # Write a separator
            writer.writerow([])

            # Write col_words
            writer.writerow(['col_words'])
            for col in self.col_words:
                writer.writerow(col)

    def clone(self):
        # Create a new instance of the class
        cloned_instance = UmDieEckeGedacht(self.num_rows, self.num_cols, self.preferred_words)

        # Copy all attributes from the current instance to the cloned instance
        cloned_instance.char_grid = [row[:] for row in self.char_grid]
        cloned_instance.row_words = [list(words) for words in self.row_words]
        cloned_instance.col_words = [list(words) for words in self.col_words]
        cloned_instance.used_words = self.used_words[:]
        cloned_instance.preferred_words = self.preferred_words[:]
        cloned_instance.current_index = self.current_index[:]
        cloned_instance.current_relevant_words = dict(self.current_relevant_words)
        cloned_instance.current_row_char_dict = [dict(char_dict) for char_dict in self.current_row_char_dict]
        cloned_instance.start_time = self.start_time

        return cloned_instance

    def check_completed(self):
        # If the current index is at the end of the last col, or even after the last col, return True
        if self.current_index[1] == self.num_cols:
            return True
        elif self.current_index[1] == self.num_cols - 1 and \
                self.current_index[0] >= self.num_rows - 1:
            return True
        else:
            return False

    def add_first_word(self, first_words, probs):
        # Add the first word to the puzzle
        # Take randomly weighted one from first words, or a random choice from the preferred words
        if not self.preferred_words:
            first_word = np.random.choice(first_words, 1, replace=False, p=probs)[0]
        else:
            first_word = np.random.choice(self.preferred_words)
        print("Chosen first word: ", first_word)
        self.add_word(first_word)

    # Add a word to the puzzle grid
    def add_word(self, next_word):
        # Add the word to the col_words and the used_words
        self.col_words[self.current_index[1]].append(next_word)
        self.used_words.append(next_word)

        # Ad it char by char to the grid, and adapt the row dictionaries
        for char_index, char in enumerate(next_word):
            row_idx = self.current_index[0] + char_index
            self.char_grid[row_idx][self.current_index[1]] = char
            self.current_row_char_dict[row_idx] = self.current_row_char_dict[row_idx][char]

        # Adapt the current index
        self.current_index[0] += len(next_word)
        current_row, current_col = self.current_index

        # We are close to the end of one column
        if current_row >= self.num_rows - 1:
            if current_row == self.num_rows - 1:
                # The word has finished one row before the last - just take a random valid character for the last row word
                list_last_row_chars = self.current_row_char_dict[current_row].keys() - {"Word", "Word_Count"}
                if list_last_row_chars:
                    next_row_char = np.random.choice(list(list_last_row_chars))
                    self.char_grid[current_row][current_col] = next_row_char
                    self.current_row_char_dict[current_row] = self.current_row_char_dict[current_row][next_row_char]

            for row in range(self.num_rows):
                # Loop through all rows and check if any row words have been completed
                current_row_dict = self.current_row_char_dict[row]
                if "Word" in current_row_dict:
                    # Add the word if it is longer than two characters or we are already at the last colums
                    if (len(current_row_dict["Word"]) > 2) | (current_col > self.num_cols - 2):
                        # If so, append this word to the row_words and reset the row dictionary to the original state
                        row_word = self.current_row_char_dict[row]["Word"]
                        self.row_words[row].append(row_word)
                        self.used_words.append(row_word)
                        # If we are in any column but the last, take the dictionary with the corresponding maximal word lengths
                        if current_col >= self.num_cols - 2:
                            self.current_row_char_dict[row] = list_char_dicts[-1]
                        else:
                            # Else, take the whole dictionary, as the character is only relevant for the last col word
                            self.current_row_char_dict[row] = list_char_dicts[self.num_cols - 1 - current_col]

            # If we are close to finishing, print out details every time we add a complete column
            if current_col > self.num_cols * 3 / 4:
                print("Column Completed. Current Index was: ", self.current_index)
                print("Grid: ", self.char_grid)
                print("Row Words: ", self.row_words)
                print("New col words:", self.col_words[current_col])

            self.current_index = [0, current_col + 1]

    # Function to find potential next words for a riddle
    def find_potential_next_words(self, current_col_char_dict, word_beginning, row_num, sum_possible_row_words):
        next_possible_chars = {}
        current_column_next_chars = set(current_col_char_dict.keys()) - {"Word", "Word_Count"}
        current_row_next_chars_dict = self.current_row_char_dict[row_num]

        for next_row_char in current_row_next_chars_dict:
            if next_row_char in current_column_next_chars:
                next_possible_chars[next_row_char] = current_row_next_chars_dict[next_row_char]["Word_Count"]

        if not next_possible_chars:
            return

        next_possible_char_list = sorted(next_possible_chars, key=next_possible_chars.get, reverse=True)

        for next_char in next_possible_char_list:
            new_beginning = word_beginning + next_char
            new_column_dict = current_col_char_dict[next_char]
            sum_next_possible_row_words = sum_possible_row_words + next_possible_chars[next_char]

            if "Word" in new_column_dict.keys():
                new_word = new_column_dict["Word"]
                self.current_relevant_words[new_word] = sum_next_possible_row_words

            if row_num < self.num_rows - 1:
                self.find_potential_next_words(new_column_dict, new_beginning, row_num + 1,
                                               sum_next_possible_row_words)

    # Function to fill the puzzle
    def fill_riddle(self):
        # If we took more than 30 seconds per filled column up to now, restart the riddle
        if (time.time() - self.start_time) / (self.current_index[1] + 1) > 30:
            self = UmDieEckeGedacht(self.num_rows, self.num_cols, self.preferred_words)
            self.add_first_word()
            self.start_time = time.time()
            print("New riddle generated")

        current_row = self.current_index[0]
        max_word_length = self.num_rows - current_row
        relevant_char_dict = list_char_dicts[max_word_length]

        # Find all possible words, that start at the current index and go at most to the end of the column
        self.find_potential_next_words(relevant_char_dict, word_beginning="", row_num=current_row,
                                       sum_possible_row_words=0)

        # Sort the possible words by their dictionary value, i.e., the number of words they enable in the next columns
        current_relevant_words = sorted(self.current_relevant_words, key=self.current_relevant_words.get,
                                        reverse=True)
        # Sort preferred words to the beginning, if there are any
        list_start_with_preferred_words = [word for word in self.preferred_words if word in current_relevant_words] + \
                                          [word for word in current_relevant_words if
                                           word not in self.preferred_words]
        # If there are no possible words to be filled in, return
        if not current_relevant_words:
            return None
        # Reset the list of current relevant words to be an empty dict
        self.current_relevant_words = {}
        # Loop through all possible next words that are not yet used in the riddle
        for next_word in list_start_with_preferred_words:
            if next_word not in self.used_words:
                # Create a copy of the current self and add the new word.
                # We need a copy to be able to add another word, if the first word failed, without undoing all operations
                # that are necessarily done when adding a word
                riddle_next_iteration = self.clone()
                riddle_next_iteration.add_word(next_word)

                # If the riddle is completely solved, return the completed riddle and save it to a csv file
                if riddle_next_iteration.check_completed():
                    return riddle_next_iteration

                # Otherwise, recursively call the fill_riddle function with the riddle_next_iteration, iteratively
                # adding one word at a time
                riddle_filled = riddle_next_iteration.fill_riddle()

                # If a valid solution was returned by the method, return this solution and stop the loop
                if riddle_filled is not None:
                    return riddle_filled

        return None


# %%

# Initialize the puzzle to find all potential first words with their weights
riddle = UmDieEckeGedacht(num_rows, num_cols)
# Find the list of potential first words
riddle.find_potential_next_words(list_char_dicts[max_word_length], word_beginning="", row_num=0,
                                 sum_possible_row_words=0)
first_words = list(riddle.current_relevant_words.keys())
probs = list(riddle.current_relevant_words.values())
probs = [x / sum(probs) for x in probs]

# %%
# Solve the puzzle
riddle_solution = None
counter = 0
while riddle_solution is None:
    riddle = UmDieEckeGedacht(num_rows, num_cols, preferred_words=["Fahrrad", "Sattel", "Lenker", "Pedale",
                                                                   "Fahrradtour", "Weltreise", "Zelt", "Camping"])
    riddle.add_first_word(first_words, probs)
    print("--------------------")
    print("--------------------")
    print("Iteration: ", counter)
    print("--------------------")
    riddle_solution = riddle.fill_riddle()
    counter += 1
riddle_solution.save_to_csv("Result.csv")
