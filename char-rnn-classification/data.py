import glob, unicodedata
import string, argparse
import time, os, math, random, sys
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_printoptions(linewidth=180)
import numpy as np


class Data:
   
    def __init__(self, train_perc, valid_perc):
        """ Lines are the same as (last) names in this context 
        
        Also, when doing train/valid split, we do these splits for each of the
        categories, which ensures same ratio among different languages.
        """
        self.train_perc     = train_perc
        self.valid_perc     = valid_perc
        self.all_letters    = string.ascii_letters + " .,;'-"
        self.n_letters      = len(self.all_letters)
        self.category_lines = {} # train and valid
        self.cat_to_names_t = {} # train
        self.cat_to_names_v = {} # valid
        self.cat_to_maxlen  = {} 
        self.cat_to_avglen  = {} 
        self.all_categories = []
        self.max_all_names  = -1
        self.total_train    = 0
        self.total_valid    = 0

        # Load all the words into `self.category_lines`, one list per language.
        for filename in self.findFiles('../data/names/*.txt'):
            category = filename.split('/')[-1].split('.')[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines
        self.n_categories = len(self.all_categories)

        # Form train/test/valid splits and track lengths. For now, no test set.
        assert train_perc + valid_perc == 1

        for cat in self.all_categories:
            # Shuffle _within_ each language group; shuffle across full data later
            names = self.category_lines[cat] # all names in this language in data
            N = len(names)
            lengths = [len(x) for x in names]
            np.random.shuffle(names)
            num_train = int(train_perc * N)

            # Keep track of lengths to help us with padding later, etc.
            self.cat_to_names_t[cat] = names[:num_train]
            self.cat_to_names_v[cat] = names[num_train:]
            self.cat_to_maxlen[cat] = np.max(lengths)
            self.cat_to_avglen[cat] = np.mean(lengths)
            if self.cat_to_maxlen[cat] > self.max_all_names:
                self.max_all_names = self.cat_to_maxlen[cat]
            self.total_train += num_train
            self.total_valid += (N - num_train)

        L = int(self.max_all_names)

        # Create torch data out of above info, w/separate info for lengths
        self.X_train = torch.zeros(L, self.total_train, self.n_letters)
        self.X_valid = torch.zeros(L, self.total_valid, self.n_letters)
        self.y_train = torch.zeros(self.total_train)
        self.y_valid = torch.zeros(self.total_valid)
        self.X_train_lengths = torch.zeros(self.total_train)
        self.X_valid_lengths = torch.zeros(self.total_valid)
        self._create_torch_data('train')
        self._create_torch_data('valid')
        self._print_debug()
          

    def _create_torch_data(self, datatype):
        """ Iterates by category to form torch data; ALSO handles shuffling!! """
        if datatype == 'train':
            data = self.cat_to_names_t
        elif datatype == 'valid':
            data = self.cat_to_names_v

        name_idx = 0
        for cat in self.all_categories:
            name_list = data[cat]
            assert type(name_list) is list

            for name in name_list:
                # For each name (at `name_idx`), go through one _letter_ at a time
                for (l_idx,letter) in enumerate(name):
                    # All zero except for _this_ spot (one-hot)
                    idx = self.letterToIndex(letter)        
                    if datatype == 'train':
                        self.X_train[l_idx][name_idx][idx] = 1
                        self.y_train[name_idx] = self.all_categories.index(cat)
                    elif datatype == 'valid':
                        self.X_valid[l_idx][name_idx][idx] = 1
                        self.y_valid[name_idx] = self.all_categories.index(cat)

                # For later, to tell PyTorch to stop further computation
                if datatype == 'train':
                    self.X_train_lengths[name_idx] = len(name)
                elif datatype == 'valid':
                    self.X_valid_lengths[name_idx] = len(name)

                # Don't forget!!
                name_idx += 1
        
        # Need to shuffle by index 1, NOT index 0 (which indexes 'time')
        inds = np.random.permutation(name_idx)

        if datatype == 'train':
            assert name_idx == self.total_train
            self.X_train = torch.transpose(self.X_train, 0, 1)[inds]
            self.X_train = torch.transpose(self.X_train, 0, 1) # back to original
            self.y_train = self.y_train[inds]
            self.X_train_lengths = self.X_train_lengths[inds]
        elif datatype == 'valid':
            assert name_idx == self.total_valid
            self.X_valid = torch.transpose(self.X_valid, 0, 1)[inds]
            self.X_valid = torch.transpose(self.X_valid, 0, 1) # back to original
            self.y_valid = self.y_valid[inds]
            self.X_valid_lengths = self.X_valid_lengths[inds]


    def unicodeToAscii(self, s):
        """ Turn a Unicode string to plain ASCII, thanks to 
        http://stackoverflow.com/a/518232/2809427
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )
    
    def readLines(self, filename):
        """ Read a file and split into lines """
        lines = open(filename).read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]
    
    def letterToIndex(self, letter):
        """ Find letter index from all_letters, e.g. "a" = 0 """
        return self.all_letters.find(letter)
    
    # deprecated
    def lineToTensor(self, line):
        """ Turn a line into a <line_length x 1 x n_letters>,
        or an array of one-hot letter vectors
        """
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

    # deprecated
    def random_training_pair(self):
        """ `line_tensor` is the rnn input, `category_tensor` is target, i.e.
        the index of the language corresponding to `line_tensor` 
        """
        category        = self.randomChoice(self.all_categories)
        line            = self.randomChoice(self.category_lines[category])
        category_tensor = torch.LongTensor([self.all_categories.index(category)])
        line_tensor     = self.lineToTensor(line)
        #print("{}\n{}\n{}\n{}".format(category, line, category_tensor, line_tensor))
        return category, line, category_tensor, line_tensor

    def findFiles(self, path): 
        return glob.glob(path)
  
    # deprecated
    @staticmethod
    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]

    def _print_debug(self):
        print("\nFinished loading data.")
        print("    all_letters: {}".format(self.all_letters))
        print("    n_letters: {}".format(self.n_letters))
        print("    n_categories: {}".format(self.n_categories))
        print("\n(language) - (num) - (train) - (valid) - (maxlen) - (avglen)")
        for cat in self.all_categories:
            print("    {0:10}  {1:6}  {2:6}  {3:6}  {4:6}  {5:6.2f}".format(cat, 
                    len(self.category_lines[cat]),
                    len(self.cat_to_names_t[cat]),
                    len(self.cat_to_names_v[cat]),
                    self.cat_to_maxlen[cat], 
                    self.cat_to_avglen[cat])
            )
        print("\ntotal_train: {}\ntotal_valid: {}".format(
                self.total_train, self.total_valid))
        print("\nX_train.shape: {}".format(self.X_train.shape))
        print("X_valid.shape: {}\n".format(self.X_valid.shape))
        num = 5
        print("y_train[:{}]:         {}".format(num, self.y_train[:num]))
        print("X_train_lengths[:{}]: {}".format(num, self.X_train_lengths[:num]))
        print("y_valid[:{}]:         {}".format(num, self.y_valid[:num]))
        print("X_valid_lengths[:{}]: {}\n".format(num, self.X_valid_lengths[:num]))
        print("DONE\n")
 

if __name__ == "__main__":
    train_perc = 0.8
    valid_perc = 1.0 - train_perc
    d = Data(train_perc=train_perc, valid_perc=valid_perc)
