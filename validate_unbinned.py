#!/usr/bin/env python3
import sys

from helpers import *
from helpers_unbinned import *
from helpers_plotting import *

if __name__ == "__main__":

    #Load arguments
    args = parse_input()
    ignore_list=[]
    for argv in sys.argv[1:]:
        if "--" in argv: ignore_list.append(argv.replace("--","").replace("-","_"))
    load_checkpoint_args(args,ignore_args=ignore_list)
    set_seeds(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    print(f"Running on device: {device}", flush=True)

    # load and preprocess data
    print(f"Loading training set", flush=True)
    train_loader,test_loader=get_loaders(args.input_format,train_file=args.train_file,val_file=args.val_file,
    batch_size=args.batch_size, num_workers=args.num_workers,shuffle=args.shuffle)
    X_example=next(iter(train_loader))

    #load model
    print(f"Loading model", flush=True)
    model=load_checkpoint_model(X_example.shape,args)

    #Make validation plots
    validate_unbinned_models( [model], test_loader, args, labels=["original", "generated"], make_projection=True,)
    print("Done")
