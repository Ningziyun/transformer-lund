#!/usr/bin/env python3

from helpers import *
from helpers_unbinned import *
from helpers_plotting import *

if __name__ == "__main__":

    #Load arguments
    args = parse_input()
    set_seeds(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    print(f"Running on device: {device}", flush=True)

    # load and preprocess data
    print(f"Loading training set", flush=True)
    train_loader,test_loader=get_loaders(args.input_format,train_file=args.train_file,val_file=args.val_file,
    batch_size=args.batch_size, num_workers=args.num_workers,shuffle=args.shuffle)
    X_example=next(iter(train_loader))

    #load model
    model,checkpoint_info=load_checkpoint(X_example.shape,args,ignore_args=["plot_dir"])

    #Make validation plots
    validate_unbinned_models( [model], test_loader, args, labels=["original", "generated"], make_projection=True,)
    print("Done")
