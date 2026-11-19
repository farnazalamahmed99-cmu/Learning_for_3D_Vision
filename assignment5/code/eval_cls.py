import numpy as np
import argparse

import torch
from models import cls_model, dgcnn_cls
from utils import create_dir, rotate_x, viz_cls

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--rot_deg', type=float, default=0, help='tilt test objects by this many degrees about x')
    parser.add_argument('--viz', action='store_true', help='save gifs of random samples and failures')
    parser.add_argument('--model', type=str, default='pointnet', help='pointnet (Q1) or dgcnn (Q4)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='use ./checkpoints_dgcnn for Q4')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = (dgcnn_cls() if args.model == "dgcnn" else cls_model()).to(args.device)

    # Load Model Checkpoint
    model_path = '{}/cls/{}.pt'.format(args.checkpoint_dir, args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))
    if args.rot_deg != 0:
        test_data = rotate_x(test_data, args.rot_deg)

    # ------ TO DO: Make Prediction ------
    preds=[]
    with torch.no_grad():
        for i in range(0, test_data.size(0),32):
            batch=test_data[i:i+32].to(args.device)
            preds.append(model(batch).argmax(dim=1).cpu())

    pred_label = torch.cat(preds)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    if args.viz:
        names = ['chair', 'vase', 'lamp']
        test_label = test_label.long()
        tag = args.exp_name

        # A few random samples: the same 5 objects on every run (fixed seed) so settings can be compared
        rng = np.random.default_rng(0)
        for idx in rng.choice(len(test_label), 5, replace=False):
            gt, pr = names[test_label[idx]], names[pred_label[idx]]
            print(f"sample {idx}: true={gt}  pred={pr}")
            viz_cls(test_data[idx], f"{args.output_dir}/{tag}_sample{idx}_true-{gt}_pred-{pr}.gif")

        # One failure per true class
        wrong = (pred_label != test_label).nonzero().flatten()
        print("total wrong:", len(wrong), "of", len(test_label))
        for c in range(3):
            idxs = wrong[test_label[wrong] == c]
            print(f"{names[c]} misclassified ({len(idxs)}):", [(i.item(), names[pred_label[i]]) for i in idxs])
            if len(idxs) > 0:
                i = idxs[0].item()
                viz_cls(test_data[i], f"{args.output_dir}/{tag}_fail{i}_true-{names[c]}_pred-{names[pred_label[i]]}.gif")

