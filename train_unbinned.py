#!/usr/bin/env python3

from helpers import *
from helpers_unbinned import *
from helpers_plotting import *

from torchinfo import summary

class NonFiniteLossError(RuntimeError):
  pass

# ---------------------------------------------------------------------
# Functions for train and test loop
# ---------------------------------------------------------------------
def evaluate_loss(model,X,mask,args):

  if args.input_format=="4vec" and args.flatten:
    w=flatten_weight(X,device=device)
  else:
    w=torch.ones(X.shape[0:-1],device=device) #dim=[Nbatch,Nconst]

  #For all flow-based models
  if args.nf or args.diff or args.sde or args.cnf or args.fm:
    w2= w.repeat_interleave(X.shape[-1], dim=-1)
    X = X.view(X.shape[0], -1)

  #All the specifics of the loss function for each model
  if args.nf:
    loss=(model.nll_loss(X)*w2).sum()
    return loss,w.sum()
  elif args.diff:
    loss=(model.mse_loss(X)*w2).sum()
    return loss,w.sum()
  elif args.sde:
    loss=(model.mse_loss(X)*w2).sum()
    return loss,w.sum()
  elif args.cnf:
    loss = (model.nll_loss(X)*w2).sum()
    return loss,w.sum()
  elif args.fm:
    loss = (model.mse_loss(X)*w2).sum()
    return loss,w.sum()

  #Auto-regressive models
  else:
    inputs = F.pad(input=X[:, :-1, :], pad=(0,0,1,0), mode='constant', value=0) #X[:, :-1, :]   # all but last, with a 0 start token at front #pad=pad(left, right, top, bottom))
    targets = X # the whole f-vector
    pred = model(inputs)       # (batch, seq_len-1, feature_dim)

    if args.mixed_loss:
        pad_mask=(targets < 0).all(dim=-1)  # shape: [B, L]
    else:
        pad_mask=None

    if args.mdn:
        loss = (model.nll_loss(pred, targets, pad_mask)*w).sum()
    else:
        loss = (model.mse_loss(pred, targets, pad_mask)*w).sum()
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
      mask=None
      X = X.to(device)

      #calculate loss across the batch (summed and also seperate averaged value)
      optimizer.zero_grad()
      loss,sum_w =evaluate_loss(model, X, mask, args)
      loss_per_sample = loss / sum_w #average the loss across batch

      #safety check
      if not torch.isfinite(loss_per_sample):
        raise NonFiniteLossError( f"Non-finite training loss at batch {batch}: {loss_per_sample.item()}")

      #backprob
      loss.backward()

      #safety check
      for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
          raise NonFiniteLossError( f"Non-finite gradient at batch {batch} in parameter {name}")
      if args.grad_clip is not None and args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

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
      n_samples += sum_w
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

  # CNF needs gradients; others don't.
  with torch.set_grad_enabled(args.cnf): # CNF needs autograd w.r.t. x to estimate divergence; do NOT use torch.no_grad() here.

    #Loop batches
    for batch, X in enumerate(test_loader):

      #input data
      mask=None
      X = X.to(device)

      #calculate loss across the batch (summed, not averaged)
      loss,sum_w = evaluate_loss(model, X, mask, args)

      #safety check
      if not torch.isfinite(loss):
        raise NonFiniteLossError( f"Non-finite test loss at batch {batch}: {loss.item()}")

      #Print loss and save some for later
      if batch % 100 == 0:
        loss_per_sample = loss / sum_w
        print(f"test batch: {batch} loss:{loss_per_sample}", flush=True)
      epoch_loss += loss.item() #sum the loss across the batch, rolling sum across all batches
      n_samples+=sum_w

    #Get the average loss across whole batch
    epoch_loss /= n_samples #Divide total numper of events
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
    args = parse_input()
    set_seeds(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    print(f"Running on device: {device}", flush=True)

    #If continuning
    if args.contin:
        checkpoint_info=load_checkpoint_args(args)
    else:
        checkpoint_info = None

    # load and preprocess data
    print(f"Loading training set", flush=True)
    train_loader,test_loader=get_loaders(args)
    X_example=next(iter(train_loader))

    # construct model
    if args.contin:
        model = load_checkpoint_model(X_example.shape,args)
    else:
        model = build_unbinned_model(X_example.shape, args)

    #Make output directory and make metadata file to save arguments
    save_argument_metadata(args)
    print(f"Logging to {args.log_dir}", flush=True)

    #Plot the model summary
    if args.nf:
      X_example = X_example.view(X_example.shape[0], -1)
      print("Input shape,",X_example.shape, flush=True)
      modelstats=summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      print("Output shape,", model(X_example)[0].shape, model(X_example)[1].shape, flush=True)
    elif args.diff or args.sde or args.cnf or args.fm:
      X_example = X_example.view(X_example.shape[0], -1)
      print("Input shape,",X_example.shape, flush=True)
      modelstats=summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      print("Output shape,", model(X_example).shape,flush=True)
    else:
      print("Input shape,",X_example.shape, flush=True)
      modelstats=summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      if args.mixed_loss:
        print("Output shape,", model(X_example)[0].shape,model(X_example)[-1].shape, flush=True)
      else:
        print("Output shape,", model(X_example).shape, flush=True)
    model.to(device)

    #Set the scheduler
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)
    start_epoch = 0
    if checkpoint_info is not None:
      if checkpoint_info.get("optimizer_state_dict", None) is not None:
        optimizer.load_state_dict(checkpoint_info["optimizer_state_dict"])
      if scheduler is not None and checkpoint_info.get("scheduler_state_dict", None) is not None:
        scheduler.load_state_dict(checkpoint_info["scheduler_state_dict"])
      start_epoch = int(checkpoint_info.get("epoch", -1)) + 1
      print(f"Resuming after ep {start_epoch}", flush=True)

    #Store loss and etc for per-epoch loop
    best_loss=float("inf")
    best_epoch=-1
    patience_counter=0
    patience = args.patience
    test_losses=[]
    train_losses=[]
    lr_history=[]
    loss_curves={}
    stopped_nonfinite = False
    if checkpoint_info is not None:
      best_loss = checkpoint_info.get("best_loss", best_loss)
      if best_loss is None:
        best_loss = float("inf")
      else:
        best_loss = float(best_loss)
      best_epoch = checkpoint_info.get("best_epoch", best_epoch)
      test_losses = list(checkpoint_info.get("test_losses", []))
      train_losses = list(checkpoint_info.get("train_losses", []))
      lr_history = list(checkpoint_info.get("lr_history", []))
      loss_curves = dict(checkpoint_info.get("loss_curves", {}))
    epochs=args.epochs 

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    #Loop over epochs
    for epoch in range(start_epoch, epochs):
      print(f"\nEpoch {epoch+1}\n-------------------------------", flush=True)
      starttime=time.time()

      #Run the training loop and check for errors
      try:
        starttime=time.time()
        train_loss = train(model,train_loader,args)
        train_losses.append(train_loss.item())
        train_time=(time.time()-starttime)/60

        starttime=time.time()
        test_loss = test(model,test_loader,args)
        test_losses.append(test_loss.item())
        test_time=(time.time()-starttime)/60

      except NonFiniteLossError as err:
        print(f"Stopping due to non-finite value: {err}", flush=True)
        stopped_nonfinite = True
        break
      print("Took %.2f(%.2f) minutes to run training(testing)"%(train_time,test_time), flush=True)

      current_lr = optimizer.param_groups[0]["lr"]
      lr_history.append(current_lr)

      #Save some best values
      best_metric = test_loss if test_loss is not None else train_loss
      improved = best_metric<best_loss
      if improved:
        best_loss=best_metric
        best_epoch=epoch
        patience_counter=0
      else:
        patience_counter+=1

      #Step the scheduler
      step_scheduler(scheduler, args, metric=best_metric)

      #Checkpoint info
      save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        loss=best_metric,
        args=args,
        is_best=improved,
        train_losses=train_losses,
        test_losses=test_losses,
        loss_curves=loss_curves,
        lr_history=lr_history,
        best_epoch=best_epoch,
        best_loss=best_loss,
        current_lr=current_lr,
      )

      #early stopping
      if patience_counter>=patience:
        print("Early stopping", flush=True)
        break

    # ---------------------------------------------------------------------
    # Save info
    # ---------------------------------------------------------------------
    #Make loss plots and scheduler plots
    loss_plot(train_losses,test_losses,out_dir=args.log_dir,loss_curves=loss_curves)
    save_lr_csv(lr_history, out_dir=args.log_dir)
    save_lr_plot(lr_history, out_dir=args.log_dir)

    #Store some results in a dictionary
    results={}
    results["model_paramaters"]=modelstats.total_params
    results["best_epoch"]=best_epoch
    results["best_loss"]=best_loss
    results["checkpoint"]="checkpoints/best.pt"
    results["train_time"]=train_time
    results["test_time"]=test_time
    results["train_N"]=len(train_loader.dataset)
    results["test_N"]=len(test_loader.dataset)
    results["train_losses"]=train_losses
    results["test_losses"]=test_losses
    results["lr_history"]=lr_history

    #Make validation plots
    if stopped_nonfinite:
      print("Training stopped on a non-finite value; final generated validation plots will be marked unavailable.", flush=True)
      validate_unbinned_models( [model], test_loader, args, results=results, labels=["original", "generated"], unavailable_model_reasons=["training stopped on nan/inf loss or parameters"],)
    else:
      validate_unbinned_models( [model], test_loader, args, results=results, labels=["original", "generated"])

    #update metadata with some result ddinfo
    append_result_metadata(args,results)

    print("Done")
