#!/usr/bin/env python3

import os,sys,time

import helpers 
import helpers_binned 
import helpers_plotting

import torch
import torch.nn.functional as F

from torchinfo import summary

class NonFiniteLossError(RuntimeError):
  pass

# ---------------------------------------------------------------------
# Functions for train and test loop
# ---------------------------------------------------------------------
def evaluate_loss(model,args,codes,c=None):
  #Token sequence [BOS] tokens [EOS] [PAD]..., learn p(token_i | tokens_<i, c)
  seq = model.encode_sequence(codes)   # (batch, 1+Nconst*Nlevels+1)
  inputs = seq[:, :-1]                 # all but last, starting with the start token (replaced by c if conditional)
  targets = seq[:, 1:]                 # shifted by one, ends with the EOS
  logits = model(inputs, c)            # (batch, seq_len, vocab)

  w = (targets != model.PAD).float()  #dim=[Nbatch,seq_len], only real tokens + the EOS
  loss = (model.ce_loss(logits, targets)*w).sum()
  return loss, w.sum()

def train(model,train_loader,args):
  model.train() #Set to training mode to caluclate gradients

  #Store some values
  best_loss=1e6
  epoch_loss=0.0
  n_samples=0

  #Loop batches
  for batch, X in enumerate(train_loader):

      #input data
      codes, jet=helpers_binned.format_codes(X, args, device)

      #calculate loss across the batch (summed and also seperate averaged value)
      optimizer.zero_grad()
      loss,sum_w =evaluate_loss(model,args, codes, jet)
      loss_per_sample = loss / sum_w #average the loss across batch (per token)

      #safety check
      if not torch.isfinite(loss_per_sample):
        raise NonFiniteLossError( f"Non-finite training loss at batch {batch}: {loss_per_sample.item()}")

      #backprob
      loss_per_sample.backward()

      #safety check
      if args.grad_clip is not None and args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
          raise NonFiniteLossError( f"Non-finite gradient at batch {batch} in parameter {name}")

      #backprob
      optimizer.step()

      #safety check
      for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
          raise NonFiniteLossError( f"Non-finite parameter after optimizer step at batch {batch}: {name}")

      #Print loss and save some for later
      if batch % 100 == 0:
        print(f"batch: {batch} loss:{loss_per_sample.item()}", flush=True)
      if loss_per_sample.item()<best_loss: best_loss=loss_per_sample.item()
      epoch_loss += loss.item() #Sum across epoch
      n_samples += sum_w.item()
      if args.test and batch>1000: break

  #Get the average loss across whole epoch (not same as average of per-batch averages)
  epoch_loss /= n_samples
  print(f"train loss={epoch_loss} best_batch_loss={best_loss}", flush=True)
  return epoch_loss

def test(model, test_loader, args):
  model.eval()  # disable dropout for evaluation

  #Store some values
  n_samples=0
  epoch_loss = 0.0

  with torch.no_grad():

    #Loop batches
    for batch, X in enumerate(test_loader):

      #input data
      codes, jet=helpers_binned.format_codes(X, args, device)

      #calculate loss across the batch (summed, not averaged)
      loss,sum_w = evaluate_loss(model, args, codes, jet)

      #safety check
      if not torch.isfinite(loss):
        raise NonFiniteLossError( f"Non-finite test loss at batch {batch}: {loss.item()}")

      #Print loss and save some for later
      if batch % 100 == 0:
        loss_per_sample = loss / sum_w
        print(f"test batch: {batch} loss:{loss_per_sample}", flush=True)
      epoch_loss += loss.item() #sum the loss across the batch, rolling sum across all batches
      n_samples+=sum_w.item()

    #Get the average loss across whole batch
    epoch_loss /= n_samples #Divide total number of tokens
    print(f"test loss={epoch_loss}", flush=True)
    return epoch_loss

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Preamble
    # ---------------------------------------------------------------------
    #Load arguments
    args = helpers_binned.parse_input()
    if args.architecture!="VQGPT":
        raise ValueError("train_binned.py trains --architecture VQGPT; use train_unbinned.py for the others")
    helpers.set_seeds(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    print(f"Running on device: {device}", flush=True)

    #If continuing
    if args.model_checkpoint:
        checkpoint_info = load_checkpoint_args(args, sys.argv[1:])
    for name in ("tokenizer", "train_codes", "test_codes"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name.replace('_', '-')} is required")

    # load the codes
    print(f"Loading training set", flush=True)
    train_loader,test_loader=helpers_binned.get_code_loaders(args)
    codes_example, jet_example=helpers_binned.format_codes(next(iter(train_loader)), args, device)

    #Number of constituents comes from the codes; the validation loader must use the same
    n_const=train_loader.dataset.n_const
    if args.num_constituents!=n_const:
        print(f"Setting --num-constituents {n_const} (from the codes file, was {args.num_constituents})", flush=True)
        args.num_constituents=n_const

    #Shape for later
    X_shape=codes_example.shape
    cond_shape=[0]
    if args.conditional:
        cond_shape=jet_example.shape

    # construct model
    if args.model_checkpoint:
        model = load_checkpoint_model(args, X_shape, cond_shape, checkpoint_info)
    else:
        model = helpers_binned.build_binned_model(args, X_shape, cond_shape)
        if args.conditional:
            model.fit_cond_normalisation(train_loader.dataset.cond) #standardise the jet conditioning (stored in the state dict)
    model.to(device)
    if model.Q!=train_loader.dataset.n_levels:
        raise ValueError(f"codes have {train_loader.dataset.n_levels} levels, tokenizer has {model.Q}")

    #Make output directory and make metadata file to save arguments
    helpers.save_argument_metadata(args)
    print(f"Logging to {args.log_dir}", flush=True)

    #Plot the model summary
    input_example={"x": model.encode_sequence(codes_example)[:, :-1]}
    if args.conditional:
        input_example["c"]=jet_example
    print("Codes shape,",X_shape, "-> token sequence", tuple(input_example["x"].shape), flush=True)
    print(f"Vocabulary {model.n_code_tokens} codes ({model.Q} levels x {model.K}) + EOS", flush=True)
    if args.conditional:
        print("Conditioning shape,", cond_shape, flush=True)
    modelstats=summary(model, input_data=input_example, col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
    print("Output shape,", model(**input_example).shape, flush=True)

    #Set the scheduler
    optimizer = helpers.make_optimizer(args, model)
    scheduler = helpers.make_scheduler(args, optimizer)
    start_epoch = 0
    if args.model_checkpoint:
        start_epoch=load_checkpoint_scheduler_optimizer(args, checkpoint_info, optimizer, scheduler)

    #Store loss and etc for per-epoch loop
    training_info={
        "best_loss":float("inf"),
        "best_epoch":-1,
        "patience_counter":0,
        "test_losses":[],
        "train_losses":[],
        "lr_history":[],
        "stopped_nonfinite":False,
    }
    if args.model_checkpoint:
        helpers.load_checkpoint_losses_lr(args, checkpoint_info, training_info)

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    #Loop over epochs
    train_time, test_time = 0.0, 0.0
    for epoch in range(start_epoch, args.epochs):
      print(f"\nEpoch {epoch+1}\n-------------------------------", flush=True)
      starttime=time.time()

      current_lr = optimizer.param_groups[0]["lr"]
      training_info["lr_history"].append(current_lr)

      #Run the training loop and check for errors
      try:
        starttime=time.time()
        train_loss = train(model,train_loader,args)
        training_info["train_losses"].append(train_loss)
        train_time=(time.time()-starttime)/60

        starttime=time.time()
        test_loss = test(model,test_loader,args)
        training_info["test_losses"].append(test_loss)
        test_time=(time.time()-starttime)/60

      except NonFiniteLossError as err:
        print(f"Stopping due to non-finite value: {err}", flush=True)
        training_info["stopped_nonfinite"] = True
        break
      print("Took %.2f(%.2f) minutes to run training(testing)"%(train_time,test_time), flush=True)

      #Save some best values
      best_metric = test_loss if test_loss is not None else train_loss
      improved = best_metric<training_info["best_loss"]
      if improved:
        training_info["best_loss"]=best_metric
        training_info["best_epoch"]=epoch
        training_info["patience_counter"]=0
      else:
        training_info["patience_counter"]+=1

      #Step the scheduler
      helpers.step_scheduler(scheduler, args, metric=best_metric)

      #Checkpoint info
      helpers.save_checkpoint(args, model, epoch, improved, optimizer=optimizer, scheduler=scheduler, training_info=training_info)

      #early stopping
      if training_info["patience_counter"]>=args.patience:
        print("Early stopping", flush=True)
        break

    # ---------------------------------------------------------------------
    # Save info
    # ---------------------------------------------------------------------
    #Make loss plots and scheduler plots
    helpers_plotting.loss_plot(training_info["train_losses"],training_info["test_losses"],out_dir=args.log_dir)
    helpers_plotting.lr_plot(training_info["lr_history"], out_dir=args.log_dir)

    #Store some results in a dictionary
    results={}
    results["model_paramaters"]=modelstats.total_params
    results["trainable_paramaters"]=modelstats.trainable_params
    results["best_epoch"]=training_info["best_epoch"]
    results["best_loss"]=training_info["best_loss"]
    results["checkpoint"]="checkpoints/best.pt"
    results["train_time"]=train_time
    results["test_time"]=test_time
    results["train_N"]=len(train_loader.dataset)
    results["test_N"]=len(test_loader.dataset)
    results["train_losses"]=training_info["train_losses"]
    results["test_losses"]=training_info["test_losses"]
    results["lr_history"]=training_info["lr_history"]

    #Make validation plots against the ORIGINAL validation file, same as for the other architectures
    print(f"Loading original-format validation set {args.val_file}", flush=True)
    val_loader=helpers.get_loader(args.val_file,args)
    if training_info["stopped_nonfinite"]:
      print("Training stopped on a non-finite value; final generated validation plots will be marked unavailable.", flush=True)
      helpers_plotting.validate_unbinned_models([model], val_loader, args, results=results, labels=["original", "generated"], unavailable_model_reasons=["training stopped on nan/inf loss or parameters"],)
    else:
      helpers_plotting.validate_unbinned_models([model], val_loader, args, results=results, labels=["original", "generated"])

    #update metadata with some result ddinfo
    helpers.append_result_metadata(args,results)

    print("Done")
