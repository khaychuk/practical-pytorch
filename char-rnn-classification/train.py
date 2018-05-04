import argparse
import time, os, math, random, sys
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_printoptions(linewidth=180)
import numpy as np
from data import Data


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



def categoryFromOutput(output, data):
    """ Pick the highest index (category) from the (log) softmax.
    Returns (category_name, category_index). 
    """
    top_n, top_i = output.topk(1) # Returns (values_list, indices_list)
    category_i = top_i[0][0]
    return data.all_categories[category_i], category_i


def train(args, data, rnn, optimizer, criterion, head):
    """ Train, print logs to stdout, save the (final) model.
    Each epoch is actually just one training minibatch, bad terminology.
    """
    start = time.time()
    current_loss = 0
    all_losses = []
    all_times = []
    print("epoch - train_frac - time_since - this_mb_loss - line - guess - correct")
   
    for epoch in range(1, args.n_epochs+1):
        #category, line, category_tensor, line_tensor = data.random_training_pair()
        category, line, category_tensor, line_tensor = data.random_training_pair()
        if args.cuda:
            category_tensor = category_tensor.cuda()
            line_tensor = line_tensor.cuda()

        sys.exit()

        hidden = rnn.initHidden()
        optimizer.zero_grad()

        # Iterate over all letters in the word(s) in mini-batch.
        # output.size(): [batch_size, n_categories]
        # hidden.size(): [batch_size, n_hidden]
        # TODO: only store output if it comes from the last relevant index
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
    parser.add_argument('--train_frac', type=float, default=0.8)
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
    data = Data(args.train_frac, 1.0-args.train_frac, args.batch_size)
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
