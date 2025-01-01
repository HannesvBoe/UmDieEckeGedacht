from utils import *  # Import necessary utilities


# Define the main class for the puzzle
class Um_die_ecke_gedacht(object):
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
        self.preferred_words = sorted(preferred_words, key=len)
        self.start_time = time.time()

        # Initialize row character dictionaries
        self.current_row_char_dict = [list_char_dicts[self.num_cols] for _ in range(num_rows)]

    def save_to_csv(self, filename):
        filename = "Riddle_Results/" + filename
        print("Save riddle!")
        # Check if the file already exists
        if os.path.exists(filename):
            # Get the current date and time
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Add the current date and time to the beginning of the filename
            filename = f"{current_time}_{filename}"
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
        cloned_instance = Um_die_ecke_gedacht(self.num_rows, self.num_cols)

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

    def add_first_word(self):
        # Add the first word to the puzzle
        if not self.preferred_words:
            first_word = np.random.choice(word_dict_by_length[self.num_rows])
        else:
            first_word = np.random.choice(self.preferred_words)
        print(first_word)
        self.add_word(first_word)

    def add_word(self, word):
        # Add a word to the puzzle grid
        self.col_words[self.current_index[1]].append(word)
        self.used_words.append(word)
        for char_index, char in enumerate(word):
            row_idx = self.current_index[0] + char_index
            self.char_grid[row_idx][self.current_index[1]] = char
            self.current_row_char_dict[row_idx] = self.current_row_char_dict[row_idx][char]

        self.current_index[0] += len(word)
        current_row, current_col = self.current_index

        if current_row >= self.num_rows - 1:
            if current_row == self.num_rows - 1:
                list_last_row_chars = self.current_row_char_dict[current_row].keys() - {"Word", "Word_Count"}
                if list_last_row_chars:
                    next_row_char = np.random.choice(list(list_last_row_chars))
                    self.char_grid[current_row][current_col] = next_row_char
                    self.current_row_char_dict[current_row] = self.current_row_char_dict[current_row][next_row_char]

            for row in range(self.num_rows):
                if "Word" in self.current_row_char_dict[row]:
                    row_word = self.current_row_char_dict[row]["Word"]
                    self.row_words[row].append(row_word)
                    self.used_words.append(row_word)
                    if current_col >= self.num_cols - 2:
                        self.current_row_char_dict[row] = list_char_dicts[-1]
                    else:
                        self.current_row_char_dict[row] = list_char_dicts[self.num_cols - 1 - current_col]

            if current_col > 4:
                print("Column Completed. Current Index was: ", self.current_index)
                print("Grid: ", self.char_grid)
                print("Row Words: ", self.row_words)
                print("New col words:", self.col_words[current_col])

            self.current_index = [0, current_col + 1]


# Function to fill the puzzle
def fill_riddle(riddle_inst):
    if (time.time() - riddle_inst.start_time) / (riddle_inst.current_index[1] + 1) > 30:
        riddle_inst = Um_die_ecke_gedacht(riddle_inst.num_rows, riddle_inst.num_cols)
        riddle_inst.add_first_word()
        riddle_inst.start_time = time.time()
        print("New riddle generated")

    current_row = riddle_inst.current_index[0]
    max_word_length = riddle_inst.num_rows - current_row
    relevant_char_dict = list_char_dicts[max_word_length]

    find_potential_next_words(riddle_inst, relevant_char_dict, word_beginning="", row_num=current_row,
                              sum_possible_row_words=0)

    current_relevant_words = sorted(riddle_inst.current_relevant_words, key=riddle_inst.current_relevant_words.get,
                                    reverse=True)
    list_start_with_preferred_words = [word for word in riddle_inst.preferred_words if word in current_relevant_words] + \
                                      [word for word in current_relevant_words if
                                       word not in riddle_inst.preferred_words]

    if not current_relevant_words:
        return None

    riddle_inst.current_relevant_words = {}
    for next_word in list_start_with_preferred_words:
        if next_word not in riddle_inst.used_words:
            riddle_next_iteration = riddle_inst.clone()
            riddle_next_iteration.add_word(next_word)

            if riddle_next_iteration.current_index[1] == riddle_inst.num_cols:
                riddle_next_iteration.save_to_csv("Result.csv")
                return riddle_next_iteration

            if riddle_next_iteration.current_index[1] == riddle_inst.num_cols - 1 and \
                    riddle_next_iteration.current_index[0] >= riddle_inst.num_rows - 1:
                riddle_next_iteration.save_to_csv("Result.csv")
                return riddle_next_iteration

            riddle_filled = fill_riddle(riddle_next_iteration)
            if riddle_filled is not None:
                return riddle_filled

    return None


# Initialize the puzzle parameters
num_rows = 10
num_cols = 20
max_word_length = max(num_cols, num_rows)

# Create the character dictionaries
list_char_dicts = [create_word_char_dict(word_list, i)[2] for i in range(max_word_length + 1)]
list_of_all_words, word_dict_by_length, word_dict_by_char = create_word_char_dict(word_list, max_word_length)

# Initialize the puzzle
riddle = Um_die_ecke_gedacht(num_rows, num_cols)
find_potential_next_words(riddle, list_char_dicts[max_word_length], word_beginning="", row_num=0,
                          sum_possible_row_words=0)
first_words = list(riddle.current_relevant_words.keys())
probs = list(riddle.current_relevant_words.values())
probs = [x / sum(probs) for x in probs]

#%%
# Solve the puzzle
riddle_solution = None
counter = 0
while riddle_solution is None:
    riddle = Um_die_ecke_gedacht(num_rows, num_cols)
    first_word = np.random.choice(first_words, 1, replace=False, p=probs)[0]
    print("--------------------")
    print("--------------------")
    print("Iteration: ", counter)
    print("First word: ", first_word)
    print("--------------------")
    riddle.add_word(first_word)
    riddle_solution = fill_riddle(riddle)
    counter += 1

#%%
riddle_solution.save_to_csv("Result.csv")