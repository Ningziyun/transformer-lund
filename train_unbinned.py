#!/usr/bin/env python3

from helpers_unbinned import *

class NonFiniteLossError(RuntimeError):
  pass

def evaluate_loss(model,X,mask,args):
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
    loss = model.nll_loss(X).sum()
    return loss

  inputs = X[:, :-1, :]   # all but last
  targets = X[:, 1:, :]   # all but first
  pred = model(inputs)       # (batch, seq_len-1, feature_dim)

  #if args.mixed_loss:
  #  pred[:,:,-1]=sigmoid(pred[:,:,-1])

  #mask = ~((targets[:, :, 0] == -1) & (targets[:, :, 1] == -1))  # shape: [B, L]
  mask=None
  
  # CNF mode: likelihood-based loss
  if args.cnf:
    loss = model.nll_loss(inputs, targets, mask=mask)
  elif args.mdn:
    loss = model.nll_loss(pred, targets, mask)
    loss=loss.sum()
  else:
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

  return loss

def train(model,train_loader,args):
  model.train()
  bestloss=1e6
  epoch_loss=0.0
  n_batches=0
  for batch, X in enumerate(train_loader):
      mask=None
      X = X.to(device)
      optimizer.zero_grad()

      loss=evaluate_loss(model, X, mask, args)
      loss_per_sample = loss / X.shape[0]
      if not torch.isfinite(loss_per_sample):
        raise NonFiniteLossError(
          f"Non-finite training loss at batch {batch}: {loss_per_sample.item()}"
        )

      loss.backward()
      for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
          raise NonFiniteLossError(
            f"Non-finite gradient at batch {batch} in parameter {name}"
          )
      if args.grad_clip is not None and args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()
      for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
          raise NonFiniteLossError(
            f"Non-finite parameter after optimizer step at batch {batch}: {name}"
          )

      loss=loss_per_sample #average the loss across batch

      if batch % 100 == 0:
        print(f"batch: {batch} loss:{loss.item()}", flush=True)
      if loss.item()<bestloss: bestloss=loss.item()
      epoch_loss += loss.item()
      n_batches += 1
      #if batch>500: break #FIXME

  avg_loss = epoch_loss / max(n_batches, 1)
  print(f"train loss={avg_loss} best_batch_loss={bestloss}", flush=True)
  loss_train.append(avg_loss)
  return avg_loss

def test(model, test_loader, args):
  model.eval()  # disable dropout for evaluation
  # CNF needs autograd w.r.t. x to estimate divergence; do NOT use torch.no_grad() here.
  if args.cnf:
    num_samples = len(test_loader.dataset)
    epochloss = 0.0
    for batch, X in enumerate(test_loader):
      mask=None
      X = X.to(device)
      loss = evaluate_loss(model, X, mask, args)
      if not torch.isfinite(loss):
        raise NonFiniteLossError(
          f"Non-finite test loss at batch {batch}: {loss.item()}"
        )
      epochloss += loss.item() #sum the loss across epoch
      if batch % 100 == 0:
        print(f"test batch: {batch}", flush=True)
    epochloss /= num_samples
    print(f"test loss ={epochloss}", flush=True)
    loss_test.append(epochloss)
    return epochloss

  # Non-CNF models can safely disable grads during evaluation
  with torch.no_grad():
    num_samples = len(test_loader.dataset)
    epochloss = 0.0
    for batch, X in enumerate(test_loader):
      mask=None
      X = X.to(device)
      loss = evaluate_loss(model, X, mask, args)
      if not torch.isfinite(loss):
        raise NonFiniteLossError(
          f"Non-finite test loss at batch {batch}: {loss.item()}"
        )
      epochloss += loss.item() #sum the loss across epoch
      if batch % 100 == 0:
        print(f"test batch: {batch}", flush=True)

    epochloss /= num_samples # average loss across whole epoch
    print(f"test loss ={epochloss}", flush=True)
    loss_test.append(epochloss)
    return epochloss

if __name__ == "__main__":
    args = parse_input()
    set_seeds(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    print(f"Running on device: {device}", flush=True)

    num_features = args.num_features
    num_bins = tuple(args.num_bins)

    # load and preprocess data
    print(f"Loading training set", flush=True)
    train_loader,test_loader=get_loaders(args.input_format,train_file=args.train_file,val_file=args.val_file,
    batch_size=args.batch_size, num_workers=args.num_workers,shuffle=args.shuffle)
    X_example=next(iter(train_loader))

    # construct model
    resume_state = None
    if args.contin:
        if len(args.model_path) == 0:
          raise ValueError("--contin requires --model-path/--checkpoint")
        load_path = args.model_path[0] if isinstance(args.model_path, list) else args.model_path
        loaded = load_model(load_path)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
          resume_state = loaded
          loaded_args = loaded.get("args", {})
          for key in (
              "cnf",
              "mdn",
              "mixed_loss",
              "embed_dim",
              "num_heads",
              "num_layers",
              "ff_dim",
              "n_mix",
              "cnf_hidden",
              "cnf_steps",
              "flow_hidden",
          ):
            if key in loaded_args:
              setattr(args, key, loaded_args[key])
          model = build_unbinned_model(X_example.shape, args)
          model.load_state_dict(loaded["model_state_dict"])
        else:
          model = loaded
        print("Loaded model", flush=True)
    else:
        model = build_unbinned_model(X_example.shape, args)

    save_arguments(args)
    append_training_metadata(args)
    print(f"Logging to {args.log_dir}", flush=True)

    if args.cnf:
      args.mdn = False
      args.mixed_loss = False

    #Set the loss
    if not args.mdn and not args.cnf:
      loss_fn = nn.MSELoss(reduction='none')   # regression next-step prediction
      if args.mixed_loss:
        #loss_fn2 = nn.CrossEntropyLoss() #expects logits
        loss_fn2 = nn.BCEWithLogitsLoss(reduction='none') #expects logits
        sigmoid=nn.Sigmoid()

    #Plot the model summary
    if args.nf:
      X_example = X_example.view(X_example.shape[0], -1)
      print("Input shape,",X_example.shape, flush=True)
      summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      print("Output shape,", model(X_example)[0].shape, model(X_example)[1].shape, flush=True)
    elif args.diff:
      X_example = X_example.view(X_example.shape[0], -1)
      summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
    elif args.sde:
      X_example = X_example.view(X_example.shape[0], -1)
      summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
    elif args.cnf:
      X_example = X_example.view(X_example.shape[0], -1)
      print("Input shape,",X_example.shape, flush=True)
      summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      print("Output shape,", model(X_example).shape,flush=True)
    elif args.fm:
      X_example = X_example.view(X_example.shape[0], -1)
      print("Input shape,",X_example.shape, flush=True)
      summary(model, input_data=[X_example], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      print("Output shape,", model(X_example).shape,flush=True)
    else:
      print("Input shape,",X_example.shape, flush=True)
      summary(model, input_data=[X_example[:,:-1,:]], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      print("Output shape,", model(X_example[:,:-1,:]).shape, flush=True)
    model.to(device)

    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    start_epoch = 0
    if resume_state is not None:
      if resume_state.get("optimizer_state_dict", None) is not None:
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
      if scheduler is not None and resume_state.get("scheduler_state_dict", None) is not None:
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
      start_epoch = int(resume_state.get("epoch", -1)) + 1
      print(f"Resuming after ep {start_epoch}", flush=True)

    best_loss=float("inf")
    best_epoch=-1
    patience_counter=0
    patience = args.patience
    loss_test=[]
    loss_train=[]
    lr_history=[]
    loss_curves={}
    stopped_nonfinite = False
    if resume_state is not None:
      best_loss = resume_state.get("best_loss", best_loss)
      if best_loss is None:
        best_loss = float("inf")
      else:
        best_loss = float(best_loss)
      best_epoch = resume_state.get("best_epoch", best_epoch)
      loss_test = list(resume_state.get("test_losses", []))
      loss_train = list(resume_state.get("train_losses", []))
      lr_history = list(resume_state.get("lr_history", []))
      loss_curves = dict(resume_state.get("loss_curves", {}))
    epochs=args.epochs 


    for epoch in range(start_epoch, epochs):
      print(f"\nEpoch {epoch+1}\n-------------------------------", flush=True)
      starttime=time.time()
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

      best_metric = test_loss if test_loss is not None else train_loss
      improved = best_metric<best_loss
      if improved:
        best_loss=best_metric
        best_epoch=epoch
        patience_counter=0
      else:
        patience_counter+=1

      step_scheduler(scheduler, args, metric=best_metric)

      save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        loss=best_metric,
        args=args,
        train_losses=loss_train,
        test_losses=loss_test,
        loss_curves=loss_curves,
        lr_history=lr_history,
        best_epoch=best_epoch,
        best_loss=best_loss,
        current_lr=current_lr,
      )

      if improved:
        save_checkpoint(
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          epoch=epoch,
          loss=best_loss,
          args=args,
          ckpt_name="best.pt",
          train_losses=loss_train,
          test_losses=loss_test,
          loss_curves=loss_curves,
          lr_history=lr_history,
          best_epoch=best_epoch,
          best_loss=best_loss,
          current_lr=current_lr,
        )

      loss_plot(loss_train,loss_test,outdir=args.log_dir,loss_curves=loss_curves)
      save_lr_csv(lr_history, out_dir=args.log_dir)
      save_lr_plot(lr_history, out_dir=args.log_dir)

      if patience_counter>=patience:
        print("Early stopping", flush=True)
        break

    append_training_metadata(args, best_epoch=best_epoch, best_loss=best_loss)
    loss_plot(loss_train,loss_test,outdir=args.log_dir,loss_curves=loss_curves)
    save_lr_csv(lr_history, out_dir=args.log_dir)
    save_lr_plot(lr_history, out_dir=args.log_dir)
    if stopped_nonfinite:
      print("Training stopped on a non-finite value; final generated validation plots will be marked unavailable.", flush=True)
      validate_unbinned_models(
        [model],
        test_loader,
        args,
        labels=["original", "generated"],
        make_projection=True,
        unavailable_model_reasons=["training stopped on nan/inf loss or parameters"],
      )
    else:
      validate_unbinned_models(
        [model],
        test_loader,
        args,
        labels=["original", "generated"],
        make_projection=True,
      )
    print("Done")
