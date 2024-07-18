from __future__ import print_function

import torch
import os
import torch.optim as optim
import argparse
import sys

sys.path.append("../..")
from util.utils import progress_bar, Logger, mkdir_p, adjust_learning_rate
from util.evaluation import Evaluation
from openmax import compute_train_score_and_mavs_and_dists, fit_weibull, openmax
from util.data_preprocess import *
from network.net import *
from dataset.data_generator import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
parser = argparse.ArgumentParser(description='Open-set recognition for Appliance identfication')
parser.add_argument('--dataset', default="plaid", type=str, choices=['plaid', 'cooll'])
"""
For the PLAID dataset, the range of unknown labels is 0-10; 
for the COOLL dataset, the range of unknown labels is 0-11.
Multiple unknown labels are separated by '_'.
"""
parser.add_argument('--u_class', default='0', type=str, help='unknown_class', choices=['0', '4', '3_4_5', '7_3_4_5'])
parser.add_argument('--train_class_num', type=int, help='Classes used in training')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--es', default=150, type=int, help='epoch size')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')

# Parameters for weibull distribution fitting.
parser.add_argument('--weibull_tail', default=20, type=int, help='Classes used in testing')
parser.add_argument('--weibull_alpha', default=3, type=int, help='Classes used in testing')
parser.add_argument('--weibull_threshold', default=0.9, type=float, help='Classes used in testing')

# Parameters for stage plotting
parser.add_argument('--plot', default=True, action='store_true', help='Plotting the training set.')
parser.add_argument('--plot_max', default=0, type=int, help='max examples to plot in each class, 0 indicates all.')
parser.add_argument('--plot_quality', default=200, type=int, help='DPI of plot figure')

args = parser.parse_args()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print("unknown classes:", args.u_class)
    start_epoch = 0     # start from epoch 0 or last checkpoint epoch
    in_channels = 3

    # checkpoint
    args.checkpoint = f'./checkpoints/{args.dataset}/' + args.u_class
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    # load HSV VI-trajectory and labels
    data, labels = load_data(args.dataset)
    # DataLoader
    u_class = string_to_list(args.u_class)
    train_data, train_labels, test_data, test_labels = split_open_set(data, labels, unknown_class = u_class)
    loader = get_loaders(train_data, test_data, train_labels, test_labels)

    args.train_class_num = len(np.unique(train_labels))
    # Model
    net = Conv2D(in_channels, args.train_class_num).to(device)

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            print("BEST_ACCURACY: "+str(best_acc))
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc.', 'Test Loss', 'Test Acc.'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    epoch = 0
    if not args.evaluate:
        for epoch in range(start_epoch, args.es):
            print('\nEpoch: %d   Learning rate: %f' % (epoch+1, optimizer.param_groups[0]['lr']))
            adjust_learning_rate(optimizer, epoch, args.lr, step=20)
            train_loss, train_acc = train(net, loader['train'], optimizer, criterion, device)
            save_model(net, None, epoch, os.path.join(args.checkpoint, 'last_model.pth'))
            test_loss, test_acc = 0, 0

            logger.append([epoch+1, optimizer.param_groups[0]['lr'], train_loss, train_acc, test_loss, test_acc])

            if epoch % 10 == 0 and epoch != 0:
                test(epoch, net, loader['train'], loader['test'], criterion, device)

    test(0, net, loader['train'], loader['test'], criterion, device)
    logger.close()


# Training
def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), correct/total


def test(epoch, net, trainloader,  testloader, criterion, device):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()
            # _, predicted = outputs.max(1)
            scores.append(outputs)
            labels.append(targets)

            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader))

    # Get the prdict results.
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)

    # Fit the weibull distribution from training data.
    print("Fittting Weibull distribution...")
    _, mavs, dists = compute_train_score_and_mavs_and_dists(args.train_class_num, trainloader, device, net)
    categories = list(range(0, args.train_class_num))
    weibull_model = fit_weibull(mavs, dists, categories, args.weibull_tail, "euclidean")

    pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
    score_softmax, score_openmax = [], []
    for score in scores:
        so, ss = openmax(weibull_model, categories, score,
                         0.5, args.weibull_alpha, "euclidean")
        # print(f"so  {so} \n ss  {ss}")# openmax_prob, softmax_prob
        pred_softmax.append(np.argmax(ss))
        pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= args.weibull_threshold else args.train_class_num)
        pred_openmax.append(np.argmax(so) if np.max(so) >= args.weibull_threshold else args.train_class_num)
        score_softmax.append(ss)
        score_openmax.append(so)

    print("Evaluation...")
    eval_softmax = Evaluation(pred_softmax, labels, score_softmax)
    eval_softmax_threshold = Evaluation(pred_softmax_threshold, labels, score_softmax)
    eval_openmax = Evaluation(pred_openmax, labels, score_openmax)

    torch.save(eval_softmax, os.path.join(args.checkpoint, 'eval_softmax.pkl'))
    torch.save(eval_softmax_threshold, os.path.join(args.checkpoint, 'eval_softmax_threshold.pkl'))
    torch.save(eval_openmax, os.path.join(args.checkpoint, 'eval_openmax.pkl'))

    print(f"Softmax accuracy is %.3f" % (eval_softmax.accuracy))
    print(f"Softmax F1 is %.3f" % (eval_softmax.f1_measure))
    print(f"Softmax f1_macro is %.3f" % (eval_softmax.f1_macro))
    print(f"Softmax f1_macro_weighted is %.3f" % (eval_softmax.f1_macro_weighted))
    print(f"Softmax area_under_roc is %.3f" % (eval_softmax.area_under_roc))
    print(f"_________________________________________")

    print(f"SoftmaxThreshold accuracy is %.3f" % (eval_softmax_threshold.accuracy))
    print(f"SoftmaxThreshold F1 is %.3f" % (eval_softmax_threshold.f1_measure))
    print(f"SoftmaxThreshold f1_macro is %.3f" % (eval_softmax_threshold.f1_macro))
    print(f"SoftmaxThreshold f1_macro_weighted is %.3f" % (eval_softmax_threshold.f1_macro_weighted))
    print(f"SoftmaxThreshold area_under_roc is %.3f" % (eval_softmax_threshold.area_under_roc))
    print(f"_________________________________________")

    print(f"OpenMax accuracy is %.3f" % (eval_openmax.accuracy))
    print(f"OpenMax F1 is %.3f" % (eval_openmax.f1_measure))
    print(f"OpenMax f1_macro is %.3f" % (eval_openmax.f1_macro))
    print(f"OpenMax f1_macro_weighted is %.3f" % (eval_openmax.f1_macro_weighted))
    print(f"OpenMax area_under_roc is %.3f" % (eval_openmax.area_under_roc))
    print(f"_________________________________________")


def save_model(net, acc, epoch, path):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'testacc': acc,
        'epoch': epoch,
    }
    torch.save(state, path)


if __name__ == '__main__':
    main()

