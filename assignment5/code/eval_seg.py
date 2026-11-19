import numpy as np
import argparse

import torch
from models import seg_model, dgcnn_seg
from data_loader import get_data_loader
from utils import create_dir, viz_seg, rotate_x


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--rot_deg', type=float, default=0, help='tilt test objects by this many degrees about x')
    parser.add_argument('--model', type=str, default='pointnet', help='pointnet (Q2) or dgcnn (Q4)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='use ./checkpoints_dgcnn for Q4')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = (dgcnn_seg() if args.model == "dgcnn" else seg_model()).to(args.device)

    # Load Model Checkpoint
    model_path = '{}/seg/{}.pt'.format(args.checkpoint_dir, args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])
    if args.rot_deg != 0:
        test_data = rotate_x(test_data, args.rot_deg)

    # ------ TO DO: Make Prediction ------
    preds=[]
    with torch.no_grad():
        for i in range(0, test_data.size(0),32):
            batch=test_data[i:i+32].to(args.device)
            preds.append(model(batch).argmax(dim=2).cpu())
        pred_label = torch.cat(preds)


    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))
    per_obj_acc = (pred_label == test_label).float().mean(dim=1)      # accuracy of each chair
    print("object {} accuracy: {:.4f}".format(args.i, per_obj_acc[args.i]))
    print("worst 5:", [(i, round(per_obj_acc[i].item(), 3)) for i in per_obj_acc.argsort()[:5].tolist()])
    print("best 5: ", [(i, round(per_obj_acc[i].item(), 3)) for i in per_obj_acc.argsort(descending=True)[:5].tolist()])


    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), "cpu")
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), "cpu")
