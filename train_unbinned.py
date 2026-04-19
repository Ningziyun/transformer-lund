#!/usr/bin/env python3

from helpers_unbinned import *

def evaluate_loss(model,X,args):
  inputs = X[:, :-1, :]   # all but last
  targets = X[:, 1:, :]   # all but first
  pred = model(inputs)       # (batch, seq_len-1, feature_dim)

  #if args.mixed_loss:
  #  pred[:,:,-1]=sigmoid(pred[:,:,-1])

  #mask = ~((targets[:, :, 0] == -1) & (targets[:, :, 1] == -1))  # shape: [B, L]
  mask=None
  
  # CNF mode: likelihood-based loss
  if args.cnf:
    loss = model.nll(inputs, targets, mask=mask)
  elif args.mdn:
    loss = loss_fn(pred, targets, mask=mask)
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
  bestloss=1e6
  for batch, X in enumerate(train_loader):
      X = X.to(device)
      optimizer.zero_grad()

      loss=evaluate_loss(model, X, args)

      loss.backward()
      optimizer.step()

      loss=loss/X.shape[0] #average the loss across batch

      if batch % 100 == 0:
        print(f"batch: {batch} loss:{loss.item()}")
      if loss.item()<bestloss: bestloss=loss.item()

  print(f"train loss={bestloss}")
  loss_train.append(bestloss)

def test(model, test_loader, args):
  model.eval()  # disable dropout for evaluation
  # CNF needs autograd w.r.t. x to estimate divergence; do NOT use torch.no_grad() here.
  if args.cnf:
    num_samples = len(test_loader.dataset)
    epochloss = 0.0
    for batch, X in enumerate(test_loader):
      X = X.to(device)
      loss = evaluate_loss(model, X, args)
      epochloss += loss.item() #sum the loss across epoch
    epochloss /= num_samples
    print(f"test loss ={epochloss}")
    loss_test.append(epochloss)
    return

  # Non-CNF models can safely disable grads during evaluation
  with torch.no_grad():
    num_samples = len(test_loader.dataset)
    epochloss = 0.0
    for batch, X in enumerate(test_loader):
      X = X.to(device)
      loss = evaluate_loss(model, X, args)
      epochloss += loss.item() #sum the loss across epoch

    epochloss /= num_samples # average loss across whole epoch
    print(f"test loss ={epochloss}")
    loss_test.append(epochloss)

def validate(model,test_loader,input_shape,args):
  with torch.no_grad():
      original=torch.empty([0,input_shape[-2],input_shape[-1]]) #[Ninput,Nconst,dimension]
      generated=torch.empty([0,input_shape[-2],input_shape[-1]])
      predicted=torch.empty([0,input_shape[-2],input_shape[-1]])

      device="cpu"
      for batch, X in enumerate(train_loader):
        if batch>=5: break
        X = X.to(device)
        start=X[:,0,:].unsqueeze(1)
        model.to(device)
        pred= model(X)
        generated_seq = model.generate(start, steps=X.shape[1]-1)

        if args.mixed_loss:
          pred[:,:,-1]=sigmoid(pred[:,:,-1])
          generated_seq[:,:,-1]=sigmoid(generated_seq[:,:,-1])
          generated_seq[:,0,-1]=X[:,0,-1]

        if batch==0:
          print("Input example")
          print(X[0])
          print("Generate example")
          print(generated_seq[0])

        original=torch.cat([original, X])
        generated=torch.cat([generated,generated_seq])

      projection_plot([original.flatten(0,1).numpy(),generated.flatten(0,1).numpy()],outdir=args.log_dir)

      if args.input_format=="ktdr":
        lund_plot([original.flatten(0,1).numpy(),generated.flatten(0,1).numpy()],outdir=args.log_dir)
      else:
        #original=original[:5,:,:]
        #generated=generated[:5,:,:]
        lund_original=make_lundplane(original.numpy())
        lund_generated=make_lundplane(generated.numpy())
        lund_plot([lund_original.reshape(-1, lund_original.shape[-1]),lund_generated.reshape(-1, lund_generated.shape[-1])],outdir=args.log_dir)

if __name__ == "__main__":
    args = parse_input()
    save_arguments(args)
    print(f"Logging to {args.log_dir}")
    set_seeds(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device
    print(f"Running on device: {device}")

    num_features = args.num_features
    num_bins = tuple(args.num_bins)

    # load and preprocess data
    print(f"Loading training set")
    #train_loader,test_loader=get_loaders(train_file=args.train_file,val_file=args.val_file,batch_size=args.batch_size, num_workers=args.num_workers,shuffle=args.shuffle)
    train_loader,test_loader=get_loaders(args.input_format,train_file=args.train_file,val_file=args.val_file,
    batch_size=args.batch_size, num_workers=args.num_workers,shuffle=args.shuffle)
    X_example=next(iter(train_loader))
    print("Input shape,",X_example.shape)

    # construct model
    if args.contin:
        model = load_model(model_path=args.model_path)
        print("Loaded model")
    else:
        if args.cnf:
            model = model_transformer_CNF(input_dim=X_example.shape[2],embed_dim=args.embed_dim,num_heads=args.num_heads,
                                num_layers=args.num_layers,ff_dim=args.ff_dim,cnf_hidden=args.cnf_hidden,cnf_steps=args.cnf_steps,)
        elif args.mdn:
          model=model_transformer_MDN(input_dim=X_example.shape[2],n_mix=args.n_mix,embed_dim=args.embed_dim,num_heads=args.num_heads,
                                num_layers=args.num_layers,ff_dim=args.ff_dim,)
        else:
          model = test_model(
            input_dim=X.shape[2],
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, ff_dim=args.ff_dim
          )

    #Set the loss
    if args.cnf:
      loss_fn = None  # CNF uses model.nll(...)
    elif args.mdn:
      loss_fn=mdn_loss
    if not args.mdn and not args.cnf:
      loss_fn = nn.MSELoss(reduction='none')   # regression next-step prediction
      if args.mixed_loss:
        #loss_fn2 = nn.CrossEntropyLoss() #expects logits
        loss_fn2 = nn.BCEWithLogitsLoss(reduction='none') #expects logits
        sigmoid=nn.Sigmoid()

    #Plot the model summary
    if not args.cnf:
      summary(model, input_data=[X_example[:,:-1,:]], col_names=["input_size","output_size","num_params","params_percent","mult_adds","trainable"])
      print("Output shape,", model(X_example[:,:-1,:]).shape)
    else:
      # Skip torchinfo for CNF to avoid autograd-mode issues.
      n_params = sum(p.numel() for p in model.parameters())
      print(f"Model params: {n_params}")
    model.to(device)

    aux_mdn_head = None
    aux_cnf_flow = None
    aux_optimizer = None

    if doMultiLossPlot:
      # We train ONLY the auxiliary head parameters on detached context,
      # so it never affects the main model iteration.
      if doCNF:
        # CNF main: log MDN-NLL with an auxiliary MDN head
        aux_mdn_head = _MDNHeadFromContext(
          context_dim=args.embed_dim, input_dim=X.shape[2], n_mix=args.n_mix
        ).to(device)
        aux_optimizer = torch.optim.Adam(aux_mdn_head.parameters(), lr=args.lr)

      elif doMDN:
        # MDN main: log CNF-NLL with an auxiliary flow head
        aux_cnf_flow = _CondRealNVP2D(context_dim=args.embed_dim, hidden=args.flow_hidden).to(device)
        aux_optimizer = torch.optim.Adam(aux_cnf_flow.parameters(), lr=args.lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss functions
    if doCNF:
      loss_fn = None  # we call cnf_loss(...) directly
    elif not doMDN:
      loss_fn = nn.MSELoss(reduction='none')
      if doMixedLoss:
        loss_fn2 = nn.BCEWithLogitsLoss(reduction='none')
        sigmoid=nn.Sigmoid()
    else:
      loss_fn=mdn_loss

    best_loss=1e6
    patience_counter=0
    patience = args.patience
    loss_test=[]
    loss_train=[]
    epochs=args.epochs 
    for epoch in range(epochs):
      print(f"\nEpoch {epoch+1}\n-------------------------------")
      starttime=time.time()
      train(model,train_loader,args)
      test(model,test_loader,args)
      print("Took %.2f minutes to run"%((time.time()-starttime)/60))
      save_model(model, args.log_dir, str(epoch))
  
      if loss_train[-1]<best_loss:
        best_loss=loss_train[-1]
        patience_counter=0
      else:
        patience_counter+=1
        if patience_counter>=patience:
          print("Early stoping")
          break

    loss_plot(loss_train,loss_test,outdir=args.log_dir)
    validate(model,test_loader,X_example.shape,args)
