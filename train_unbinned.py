#!/usr/bin/env python3

from helpers import *
from helpers_unbinned import *
from helpers_plotting import *

class NonFiniteLossError(RuntimeError):
  pass

# ---------------------------------------------------------------------
# Functions for train and test loop
# ---------------------------------------------------------------------
def evaluate_loss(model,X,mask,args):

  #All the specifics of the loss function for each model
  if args.nf:
    X = X.view(X.shape[0], -1)
    loss=model.nll_loss(X).sum()
    return loss
  elif args.diff:
    X = X.view(X.shape[0], -1)
    loss=model.mse_loss(X).sum()
    return loss
  elif args.sde:
    X = X.view(X.shape[0], -1)
    loss=model.mse_loss(X).sum()
    return loss
  elif args.cnf:
    X = X.view(X.shape[0], -1)
    loss = model.nll_loss(X).sum()
    return loss
  elif args.fm:
    X = X.view(X.shape[0], -1)
    loss = model.mse_loss(X).sum()
    return loss

  else:
    inputs = F.pad(input=X[:, :-1, :], pad=(0,0,1,0), mode='constant', value=0) #X[:, :-1, :]   # all but last, with a 0 start token at front #pad=pad(left, right, top, bottom))
    targets = X # the whole f-vector
    pred = model(inputs)       # (batch, seq_len-1, feature_dim)

    #mask = ~((targets[:, :, 0] == -1) & (targets[:, :, 1] == -1))  # shape: [B, L]
    mask=None

    if args.mdn:
        loss = model.nll_loss(pred, targets, mask)
    else:
        loss = model.mse_loss(pred, targets)
    loss=loss.sum()

  '''
  else:
    inputs = X[:, :-1, :]   # all but last
    targets = X[:, 1:, :]   # all but first
    pred = model(inputs)       # (batch, seq_len-1, feature_dim)

    if args.mixed_loss:
      lambd=1
      #print(loss_fn(pred[:,:,:-1],targets[:,:,:-1]).shape,loss_fn2(pred[:,:,-1],targets[:,:,-1]).shape)
      loss= loss_fn(pred[:,:,:-1],targets[:,:,:-1]).sum(dim=-1)+lambd*loss_fn2(pred[:,:,-1],targets[:,:,-1]) #mixed loss
    else:
      loss = loss_fn(pred, targets) #whole loss
    if mask==None:
      loss=loss.sum()
    else:
      loss=loss[mask].sum() 
  '''

  return loss

def train(model,train_loader,args):
  model.train() #Set to training mode to caluclate gradients

  #Store some values
  bestloss=1e6
  epoch_loss=0.0
  n_samples=0

  #Loop batches
  for batch, X in enumerate(train_loader):
      #input data
      mask=None
      X = X.to(device)

      #calculate loss across the batch (summed and also seperate averaged value)
      optimizer.zero_grad()
      loss=evaluate_loss(model, X, mask, args)
      loss_per_sample = loss / X.shape[0] #average the loss across batch

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
      if loss_per_sample.item()<bestloss: bestloss=loss_per_sample.item()
      epoch_loss += loss.item() #Sum across epoch
      n_samples+=X.shape[0]
      #if batch>500: break #FIXME

  #Get the average loss across whole epoch (not same as average of per-batch averages)
  avg_loss = epoch_loss / n_samples
  print(f"train loss={avg_loss} best_batch_loss={bestloss}", flush=True)
  loss_train.append(avg_loss)
  return avg_loss

def test(model, test_loader, args):
  model.eval()  # disable dropout for evaluation

  #Store some values
  #num_samples = len(test_loader.dataset)
  num_samples=0
  epochloss = 0.0

  # CNF needs gradients; others don't.
  #grad_context = nullcontext() if args.cnf else torch.no_grad()

  with torch.set_grad_enabled(args.cnf): # CNF needs autograd w.r.t. x to estimate divergence; do NOT use torch.no_grad() here.

    #Loop batches
    for batch, X in enumerate(test_loader):

      #input data
      mask=None
      X = X.to(device)

      #calculate loss across the batch (summed, not averaged)
      loss = evaluate_loss(model, X, mask, args)

      #safety check
      if not torch.isfinite(loss):
        raise NonFiniteLossError( f"Non-finite test loss at batch {batch}: {loss.item()}")

      #Print loss and save some for later
      if batch % 100 == 0:
        loss_per_sample = loss / X.shape[0]
        print(f"test batch: {batch} loss:{loss_per_sample}", flush=True)
      epochloss += loss.item() #sum the loss across the batch, rolling sum across all batches
      num_samples+=X.shape[0]

    #Get the average loss across whole batch
    epochloss /= num_samples #Divide total numper of events
    print(f"test loss ={epochloss}", flush=True)
    loss_test.append(epochloss)
    return epochloss

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
    train_loader,test_loader=get_loaders(args.input_format,train_file=args.train_file,val_file=args.val_file,
    batch_size=args.batch_size, num_workers=args.num_workers,shuffle=args.shuffle)
    X_example=next(iter(train_loader))

    # construct model
    if args.contin:
        model=load_checkpoint_model(X_example.shape,args)
    else:
        model = build_unbinned_model(X_example.shape, args)

    #Make output directory and make metadata file to save arguments
    save_arguments(args)
    print(f"Logging to {args.log_dir}", flush=True)

    #Plot the model summary
    if args.nf:
      X_example = X_example.view(X_example.shape[0], -1)
      print("Input shape,",X_example.shape, flush=True)
      summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      print("Output shape,", model(X_example)[0].shape, model(X_example)[1].shape, flush=True)
    elif args.diff or args.sde or args.cnf or args.fm:
      X_example = X_example.view(X_example.shape[0], -1)
      print("Input shape,",X_example.shape, flush=True)
      summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      print("Output shape,", model(X_example).shape,flush=True)
    elif args.mdn:
      print("Input shape,",X_example.shape, flush=True)
      summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      print("Output shape,", model(X_example).shape, flush=True)
    else:   
      print("Input shape,",X_example.shape, flush=True)
      summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
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
    loss_test=[]
    loss_train=[]
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
      loss_test = list(checkpoint_info.get("test_losses", []))
      loss_train = list(checkpoint_info.get("train_losses", []))
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
        train_loss = train(model,train_loader,args)
        test_loss = test(model,test_loader,args)
      except NonFiniteLossError as err:
        print(f"Stopping due to non-finite value: {err}", flush=True)
        stopped_nonfinite = True
        break
      print("Took %.2f minutes to run"%((time.time()-starttime)/60), flush=True)
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
        train_losses=loss_train,
        test_losses=loss_test,
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
    #update metadata with some result info
    append_training_metadata(args, best_epoch=best_epoch, best_loss=best_loss)

    #Make loss plots and scheduler plots
    loss_plot(loss_train,loss_test,out_dir=args.log_dir,loss_curves=loss_curves)
    save_lr_csv(lr_history, out_dir=args.log_dir)
    save_lr_plot(lr_history, out_dir=args.log_dir)

    #Make validation plots
    if stopped_nonfinite:
      print("Training stopped on a non-finite value; final generated validation plots will be marked unavailable.", flush=True)
      validate_unbinned_models( [model], test_loader, args, labels=["original", "generated"], make_projection=True, unavailable_model_reasons=["training stopped on nan/inf loss or parameters"],)
    else:
      validate_unbinned_models( [model], test_loader, args, labels=["original", "generated"], make_projection=True,)
    print("Done")
