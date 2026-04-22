import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from preprocess import preprocess_dataframe

from argparse import ArgumentParser
from tqdm import tqdm
import os
import pandas as pd

torch.multiprocessing.set_sharing_strategy('file_system')


def load_model(name):
    model = torch.load(os.path.join(args.model_dir, f'model_{name}.pt'))
    return model


def parse_input():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='models/test', help="Model directory")
    parser.add_argument("--model_name", type=str, default='best', help="Model name")
    parser.add_argument("--data_path", type=str, default='inputFiles/top_benchmark/', help="Path to training data file")

    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of workers")

    parser.add_argument("--num_const", type=int, default=100, help="Number of constituents")
    parser.add_argument("--limit_const", action="store_true", help="Only use jets with at least num_const constituents")
    parser.add_argument("--num_events", type=int, default=10000, help="Number of events for training")
    parser.add_argument("--num_bins", type=int, nargs=3, default=[41, 31, 31], help="Number of bins per feature")
    parser.add_argument("--reverse", action='store_true', help="Whether to reverse pt order")

    args = parser.parse_args()
    return args


def load_data(path, n_events):
    df = pd.read_hdf(path, 'discretized', stop=n_events)
    x, padding_mask, _ = preprocess_dataframe(df, num_features=num_features,
                                num_bins=num_bins,
                                to_tensor=True,
                                num_const=args.num_const,
                                reverse=args.reverse,
                                limit_nconst=args.limit_const)
    labels = torch.zeros(len(x))

    df = pd.read_hdf(path.replace('qcd', 'top'), 'discretized', stop=n_events)
    x1, padding_mask1, _ = preprocess_dataframe(df, num_features=num_features,
                                num_bins=num_bins,
                                to_tensor=True,
                                num_const=args.num_const,
                                reverse=args.reverse,
                                limit_nconst=args.limit_const)
    labels = torch.concat((labels, torch.ones(len(x1))))
    x = torch.concat((x, x1), dim=0)
    padding_mask = torch.concat((padding_mask, padding_mask1), dim=0)

    train_dataset = TensorDataset(x, padding_mask, labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    print(x.shape)
    return train_loader


if __name__ == '__main__':
    args = parse_input()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    num_features = 3
    num_bins = tuple(args.num_bins)

    print(f"Using bins: {num_bins}")
    print(f"{'Not' if not args.reverse else ''} reversing pt order")

    # load and preprocess data
    print(f"Loading test set")
    test_loader = load_data(args.data_path, args.num_events)

    # construct model
    model = load_model(args.model_name)
    print(f"Loaded model '{args.model_name}'")
    model.to(device)
    model.eval()

    loss_list = []
    prediction_list = []
    label_list = []
    min_val_loss = np.inf

    for x, padding_mask, label in tqdm(test_loader, total=len(test_loader), desc=f'Testing'):
        label_list.append(label.detach().numpy())
        x = x.to(device)
        padding_mask = padding_mask.to(device)
        label = label.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = model(x, padding_mask)
                predictions = torch.sigmoid(logits)
                loss = model.loss(logits, label.view(-1, 1))

        loss_list.append(loss.cpu().detach().numpy())
        prediction_list.append(predictions.cpu().detach().numpy())

    predictions = np.concatenate(prediction_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    print(predictions.shape)
    print(label_list.shape)
    auc = roc_auc_score(label_list, predictions)
    print(auc)

    np.savez(os.path.join(args.model_dir, f'predictions_{args.model_name}.npz'),
            predictions=predictions,
            labels=label_list)
    fpr, tpr, _ = roc_curve(label_list, predictions)
    np.savez(os.path.join(args.model_dir, f"roc_{args.model_name}.npz"),
             fpr=fpr,
             tpr=tpr,
             auc=auc)
