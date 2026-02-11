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

  if args.mdn:
    loss = loss_fn(pred, targets, mask=mask)
    loss=loss.mean()
  else:
    if args.mixed_loss:
      lambd=1
      #print(loss_fn(pred[:,:,:-1],targets[:,:,:-1]).shape,loss_fn2(pred[:,:,-1],targets[:,:,-1]).shape)
      loss= loss_fn(pred[:,:,:-1],targets[:,:,:-1]).mean(dim=-1)+lambd*loss_fn2(pred[:,:,-1],targets[:,:,-1]) #mixed loss
    else:
      loss = loss_fn(pred, targets) #whole loss
    if mask==None:
      loss=loss.mean()
    else:
      loss=loss[mask].mean() 

  return loss

def train(model,train_loader,args):
  for batch, X in enumerate(train_loader):
      X = X.to(device)
      optimizer.zero_grad()

      loss=evaluate_loss(model, X, args)

      loss.backward()
      optimizer.step()

      if batch % 100 == 0:
        print(f"batch: {batch} loss:{loss.item()}")

  print(f"train loss={loss.item()}")
  loss_train.append(loss.item())

def test(model,test_loader,args):
  with torch.no_grad():
    num_samples = len(test_loader.dataset)
    epochloss=0
    for batch, X in enumerate(test_loader):
      X = X.to(device)

      loss=evaluate_loss(model, X, args)

      epochloss+=loss.item()

    epochloss/=num_samples
    print(f"test loss ={epochloss}")
    loss_test.append(epochloss)

def validate(model,test_loader,input_shape,args):
  with torch.no_grad():
      original=torch.empty([0,input_shape[-1]])
      generated=torch.empty([0,input_shape[-1]])
      predicted=torch.empty([0,input_shape[-1]])

      device="cpu"
      for batch, X in enumerate(train_loader):
        if batch>10: break
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

        original=torch.cat([original,X.flatten(0,1)])
        generated=torch.cat([generated,generated_seq.flatten(0,1)])

      projection_plot([original.numpy(),generated.numpy()])

      if args.input_format=="ktdr":
        lund_plot([original.numpy(),generated.numpy()])

if __name__ == "__main__":
    args = parse_input()
    print(f"Logging to {args.log_dir}")
    save_arguments(args)
    set_seeds(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device is None else args.device  # use CLI device if set
    print(f"Running on device: {device}")

    # load and preprocess data
    print(f"Loading training set")
    #train_loader,test_loader=get_loaders(train_file=args.train_file,val_file=args.val_file,batch_size=args.batch_size, num_workers=args.num_workers,shuffle=args.shuffle)
    train_loader,test_loader=get_loaders(args.input_format,batch_size=args.batch_size, num_workers=args.num_workers,shuffle=args.shuffle)
    X_example=next(iter(train_loader))
    print("Input shape,",X_example.shape)

    # construct model
    if args.contin:
        model = load_model(log_dir=args.model_path)
        print("Loaded model")
    else:
        if args.mdn:
          model=model_transformer_MDN(input_dim=X_example.shape[2],n_mix=args.n_mix,embed_dim=args.embed_dim,num_heads=args.num_heads,
                                num_layers=args.num_layers,ff_dim=args.ff_dim,)
        else:
          model=model_transformer(input_dim=X_example.shape[2],embed_dim=args.embed_dim,num_heads=args.num_heads,
                             num_layers=args.num_layers,ff_dim=args.ff_dim,)
          #model=model_DNN(X.shape[1],X_example.shape[2])

    #Set the loss
    if args.mdn:
      loss_fn=mdn_loss
    if not args.mdn:
      loss_fn = nn.MSELoss(reduction='none')   # regression next-step prediction
      if args.mixed_loss:
        #loss_fn2 = nn.CrossEntropyLoss() #expects logits
        loss_fn2 = nn.BCEWithLogitsLoss(reduction='none') #expects logits
        sigmoid=nn.Sigmoid()

    #Plot the model summary
    summary(model,input_data=[X_example[:,:-1,:]], col_names=["input_size", "output_size", "num_params","params_percent","mult_adds","trainable"])
    print("Output shape,",model(X_example[:,:-1,:]).shape)
    model.to(device)

    #Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #Run the training loop
    best_loss=1e6
    patience_counter=0
    patience=args.patience
    loss_test=[]
    loss_train=[]
    epochs=args.epochs 
    for t in range(epochs):
      print(f"\nEpoch {t+1}\n-------------------------------")
      starttime=time.time()
      train(model,train_loader,args)
      test(model,train_loader,args)
      print("Took %.2f minutes to run"%((time.time()-starttime)/60))
  
      if loss_train[-1]<best_loss:
        best_loss=loss_train[-1]
        patience_counter=0
      else:
        patience_counter+=1
        if patience_counter>=patience:
          print("Early stoping")
          break

    loss_plot(loss_train,loss_test)
    validate(model,test_loader,X_example.shape,args)

