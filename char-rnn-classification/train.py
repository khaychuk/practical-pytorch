import argparse
import time, os, math, random, sys
import torch
import torch.nn as nn
import torch.optim as optim
torch.set_printoptions(linewidth=240)
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

    def init_hidden(self, batch):
        """ Adding `batch` here since validation could use different size """
        hid = torch.zeros(batch, self.hidden_size)
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

    Each epoch is actually just one training minibatch, bad terminology. If we
    benchmark, we should probably manually adjust the number of epochs so that
    different minibatch sizes result in roughly the same data consumed.
    """
    start = time.time()
    current_loss = 0
    all_losses = []
    all_v_losses = []
    all_v_correct = []
    print("epoch - train_frac - time_since - train_loss - val_loss - val_correct")
   
    for epoch in range(0, args.n_epochs+1):
        mb = data.get_train_minibatch()
        maxlen = int(max(mb['X_len']).item())

        # Put last relevant output here since words have different lengths
        output_real = torch.zeros(args.batch_size, data.n_categories)
        if args.cuda:
            output_real = output_real.cuda()

        rnn.train()
        hidden = rnn.init_hidden(args.batch_size)
        optimizer.zero_grad()

        # Iterate over all RELEVANT letters in the word(s) in mini-batch.
        for l_idx in range(maxlen):
            output, hidden = rnn(mb['X_data'][l_idx], hidden)
            finished_words = (mb['X_len'] == l_idx+1).nonzero().squeeze() # careful, l_idx+1
            output_real[finished_words] = output[finished_words]

        # Takes in (input, target), shapes (N,C) and (N), resp: C = #classes.
        loss = criterion(output_real, mb['y_data'])
        loss.backward()
        optimizer.step()
        current_loss += loss

        if epoch % args.plot_every == 0:
            # Average losses across all minibatches in each plotting interval
            train_loss_avg = current_loss / args.plot_every
            all_losses.append(train_loss_avg)
            current_loss = 0

            # Validation set performance, using some reasonable minibatch size
            rnn.eval()
            bsize_v = 50
            loss_valids = []
            correct_valids = []

            for ss in range(0, data.total_valid, bsize_v):
                if ss+bsize_v >= data.total_valid:
                    break
                
                # new hidden state, get minibatch data/targets, etc.
                hidden_v = rnn.init_hidden(bsize_v)
                output_valid = torch.zeros(bsize_v, data.n_categories)
                if args.cuda:
                    output_valid = output_valid.cuda()
                data_v = data.X_valid[:, ss:ss+bsize_v, :]
                targ_v = data.y_valid[ss : ss+bsize_v]
                lengths_v = data.X_valid_lengths[ss : ss+bsize_v]
                max_len_v = int(max(lengths_v).item())

                for l_idx in range(max_len_v):
                    output, hidden_v = rnn(data_v[l_idx], hidden_v)
                    finished_words = (lengths_v == l_idx+1).nonzero().squeeze()
                    output_valid[finished_words] = output[finished_words]
                loss_v = criterion(output_valid, targ_v)
                loss_valids.append(loss_v.item())

                # Returns (values_list, inds_list), handles minibatches correctly
                top_n, top_i = output_valid.topk(1) # last dim of input by default
                top_n, top_i = top_n.squeeze(), top_i.squeeze()
                frac_correct = (top_i == targ_v).nonzero().sum() / len(targ_v)
                correct_valids.append(frac_correct)

            # Averages across minibatches (all of equal size, ignoring last one)
            all_v_losses.append(np.mean(loss_valids))
            all_v_correct.append(np.mean(correct_valids))

            if epoch % args.print_every == 0:
                t_since = timeSince(start)
                frac = epoch / args.n_epochs * 100
                print("{0} {1:5.2f}% ({2});  {3:10.4f}  {4:10.4f}  {5:10.4f}".format(
                        epoch, frac, t_since, train_loss_avg, all_v_losses[-1], 
                        all_v_correct[-1]))

    torch.save(rnn, '{}/char-rnn-classify.pt'.format(head))
    np.savetxt('{}/losses.txt'.format(head), all_losses)
    np.savetxt('{}/running_time.txt'.format(head), (time.time()-start)/60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch RNN Example')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=100000)
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--plot_every', type=int, default=100)
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
    assert args.print_every % args.plot_every == 0

    head = 'results/lr-{}-optim-{}-gpu-{}-nhidden-{}-bsize-{}-seed-{}'.format(
        args.learning_rate, args.optimizer, args.cuda, args.n_hidden, 
        args.batch_size, args.seed)
    assert not os.path.exists(head)
    os.makedirs(head)

    # Load data, build RNN, create optimizer/loss, and train. be sure to move
    # the model to GPU via `.cuda()` _before_ constructing the optimizer.
    data = Data(args.train_frac, 1.0-args.train_frac, args.batch_size, args.cuda)
    rnn = RNN(args, data.n_letters, args.n_hidden, data.n_categories)
    if args.cuda:
        rnn.cuda()

    print("\nHere is our RNN:")
    for name, param in rnn.named_parameters():
        if param.requires_grad:
            print("  {}:  {}".format(name, param.size()))
    print("RNN on GPU? {}\n".format(next(rnn.parameters()).is_cuda))

    optimizer = get_optimizer(args, rnn)
    criterion = nn.NLLLoss() # Could also consider a weight due to class imbalance
    train(args, data, rnn, optimizer, criterion, head)
