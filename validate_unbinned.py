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
    train_loader,test_loader=get_loaders(args)
    X_example,_,_=format_input(next(iter(train_loader)), args, device)

    #load model
    print(f"Loading model", flush=True)
    model=load_checkpoint_model(X_example.shape,args,device)

    if args.architecture=="MDN":
        output=model(X_example)
        stop=None
        if args.mixed_loss:
            output,stop=output

        #grabbing for first jet
        alpha=F.softmax(output[0,:,:,0]) #[Nconst,Nmix]
        mu=output[0,:,:,1:X_example.shape[-1]+1] #[Nconst,Nmix,Nfeatures]
        sigma2 = torch.exp(output[0,:,:, X_example.shape[-1]+1:]) #[Nconst,Nmix,Nfeatures]

        #find max/min mixture compotnents for first and last const and print
        argmax0=torch.argmax(alpha[0])
        argmin0=torch.argmin(alpha[0])
        argmaxn1=torch.argmax(alpha[-1])
        argminn1=torch.argmin(alpha[-1])
        print("\nalphas at 0\n",alpha[0].tolist())
        print("\nargmax/min at 0:",argmax0.item(),argmin0.item())
        print("alpha:",alpha[0,argmax0].item(),alpha[0,argmin0].item())
        print("mu:",mu[0,argmax0].tolist(),mu[0,argmin0].tolist())
        print("sigma2:",sigma2[0,argmax0].tolist(),sigma2[0,argmin0].tolist())
        print("\nalphas at -1\n",alpha[-1].tolist())
        print("\nargmax/min at -1:",argmaxn1.item(),argminn1.item())
        print("alpha:",alpha[-1,argmaxn1].item(),alpha[-1,argminn1].item())
        print("mu:",mu[-1,argmaxn1].tolist(),mu[-1,argminn1].tolist())
        print("sigma2:",sigma2[-1,argmax0].tolist(),sigma2[-1,argminn1].tolist())

        #What is the min/max values in total
        print("\nmax/min alpha:",torch.max(alpha).item(),torch.min(alpha).item())
        print("max/min mu:",torch.max(mu).item(),torch.min(mu).item())
        print("max/min sigma2:",torch.max(sigma2).item(),torch.min(sigma2).item())

    #Make validation plots
    validate_unbinned_models( [model], test_loader, args, labels=["original", "generated"])

    print("Done")
