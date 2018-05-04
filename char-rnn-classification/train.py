import glob
import unicodedata
import string, argparse
import time, os, math, random, sys
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_printoptions(linewidth=180)
import numpy as np


# Various helper methods

def get_optimizer(args, rnn):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(rnn.parameters(), 
                              lr=args.learning_rate,
                              momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(rnn.parameters(), 
                               lr=args.learning_rate)
    else:
        raise ValueError()
    return optimizer


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



class RNN(nn.Module):
    """ 
    In this network design the output node is a direct function of both the
    input and the hidden unit.
    """
    def __init__(self, args, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1) # If didn't have this, use CE loss
        self.args = args

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        hid = torch.zeros(1, self.hidden_size)
        if self.args.cuda:
            hid = hid.cuda()
        return hid



class Data:
   
    def __init__(self):
        """ Builds category_lines dictionary, a list of lines per category """
        self.all_letters = string.ascii_letters + " .,;'-"
        self.n_letters = len(self.all_letters)
        self.category_lines = {}
        self.all_categories = []

        for filename in self.findFiles('../data/names/*.txt'):
            category = filename.split('/')[-1].split('.')[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines
        self.n_categories = len(self.all_categories)

        print("Finished loading data.")
        print("    self.all_letters: {}".format(self.all_letters))
        print("    self.n_letters: {}".format(self.n_letters))
        print("    self.n_categories: {}".format(self.n_categories))
        for cat in self.all_categories:
            print("    {0:10}  {1:5}".format(cat, len(self.category_lines[cat])))
        print("")
 
    def findFiles(self, path): 
        return glob.glob(path)
    
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
    
    def lineToTensor(self, line):
        """ Turn a line into a <line_length x 1 x n_letters>,
        or an array of one-hot letter vectors
        """
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor

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

    @staticmethod
    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]


def categoryFromOutput(output, data):
    """ Pick the highest index (category) from the (log) softmax.
    Returns (category_name, category_index). 
    """
    top_n, top_i = output.topk(1) # Returns (values_list, indices_list)
    category_i = top_i[0][0]
    return data.all_categories[category_i], category_i


def train(args, data, rnn, optimizer, criterion, head):
    """ Train, print logs to stdout, save the (final) model.
    Each epoch is actually just one training minibatch.
    """
    start = time.time()
    current_loss = 0
    all_losses = []
    all_times = []
    print("epoch - train_frac - time_since - this_mb_loss - line - guess - correct")
   
    for epoch in range(1, args.n_epochs+1):
        category, line, category_tensor, line_tensor = data.random_training_pair()
        if args.cuda:
            category_tensor = category_tensor.cuda()
            line_tensor = line_tensor.cuda()
        hidden = rnn.initHidden()
        optimizer.zero_grad()

        # Iterate over all letters in the word(s) in mini-batch.
        # output.size(): [batch_size, n_categories]
        # hidden.size(): [batch_size, n_hidden]
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()
        optimizer.step()
        current_loss += loss
    
        # Check whether we got this particular minibatch correct
        if epoch % args.print_every == 0:
            guess, guess_i = categoryFromOutput(output, data)
            correct = '✓' if guess == category else '✗ (%s)' % category
            t_since = timeSince(start)
            frac = epoch / args.n_epochs * 100
            print("{} {}% ({}) {:.4f} {} / {} {}".format(
                    epoch, frac, t_since, loss, line, guess, correct))
    
        # Average losses across all minibatches in each plotting interval
        if epoch % args.plot_every == 0:
            all_losses.append(current_loss / args.plot_every)
            all_times.append( (time.time()-start)/(60) )
            current_loss = 0
    
    # For plotting later
    torch.save(rnn, '{}/char-rnn-classify.pt'.format(head))
    np.savetxt('{}/losses.txt'.format(head), all_losses)
    np.savetxt('{}/times.txt'.format(head), all_times)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch RNN Example')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=200000)
    parser.add_argument('--print_every', type=int, default=5000)
    parser.add_argument('--plot_every', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='sgd')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("Our arguments:\n{}\n".format(args))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    head = 'results/lr-{}-optim-{}-gpu-{}-nhidden-{}-bsize-{}-seed-{}'.format(
        args.learning_rate, args.optimizer, args.cuda, args.n_hidden, 
        args.batch_size, args.seed)
    assert not os.path.exists(head)
    os.makedirs(head)

    # Load data, build RNN, create optimizer/loss, and train. be sure to move
    # the model to GPU via `.cuda()` _before_ constructing the optimizer.
    data = Data()
    rnn = RNN(args, data.n_letters, args.n_hidden, data.n_categories)
    if args.cuda:
        rnn.cuda()

    print("\nHere is our RNN:")
    for name, param in rnn.named_parameters():
        if param.requires_grad:
            print("  {}:  {}".format(name, param.size()))
    print("")

    optimizer = get_optimizer(args, rnn)
    criterion = nn.NLLLoss()
    train(args, data, rnn, optimizer, criterion, head)
