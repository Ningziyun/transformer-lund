#!/usr/bin/env python3

from helpers_unbinned import *

def evaluate_loss(model,X,args):
  inputs = X[:, :-1, :]   # all but last
  targets = X[:, 1:, :]   # all but first
  pred = model(inputs)       # (batch, seq_len-1, feature_dim)

  #if args.doMixedLoss:
  #  pred[:,:,-1]=sigmoid(pred[:,:,-1])

  if args.doMDN:
    loss = loss_fn(pred, targets)
    loss=loss.mean()
  else:
    if args.doMixedLoss:
      lambd=1
      #print(loss_fn(pred[:,:,:-1],targets[:,:,:-1]).shape,loss_fn2(pred[:,:,-1],targets[:,:,-1]).shape)
      loss= loss_fn(pred[:,:,:-1],targets[:,:,:-1]).mean(dim=-1)+lambd*loss_fn2(pred[:,:,-1],targets[:,:,-1]) #mixed loss
    else:
      loss = loss_fn(pred, targets) #whole loss
    #mask = torch.ones(inputs.shape,device=device,dtype=torch.bool) #mask and loss dimension [Nbatch,NConst,Nfeatures]
    #mask = inputs[:,:,0]>-1
    #loss = loss[mask].mean()
    loss=loss.mean()

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

        if args.doMixedLoss:
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
      lund_plot([original.numpy(),generated.numpy()])

if __name__ == "__main__":
    args = parse_input()
    print(f"Logging to {args.log_dir}")
    save_arguments(args)
    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # load and preprocess data
    print(f"Loading training set")
    train_loader,test_loader=get_loaders(args.input_format)
    X_example=next(iter(train_loader))
    print("Input shape,",X_example.shape)

    # construct model
    if args.contin:
        model = load_model(log_dir=args.model_path)
        print("Loaded model")
    else:
        if args.doMDN:
          model=model_transformer_MDN(X_example.shape[2])
        else:
          model=model_transformer(X_example.shape[2])
          #model=model_DNN(X.shape[1],X_example.shape[2])

    #Set the loss
    if args.doMDN:
      loss_fn=mdn_loss
    if not args.doMDN:
      loss_fn = nn.MSELoss(reduction='none')   # regression next-step prediction
      if args.doMixedLoss:
        #loss_fn2 = nn.CrossEntropyLoss() #expects logits
        loss_fn2 = nn.BCEWithLogitsLoss(reduction='none') #expects logits
        sigmoid=nn.Sigmoid()

    #Plot the model summary
    summary(model,input_data=[X_example[:,:-1,:]], col_names=["input_size", "output_size", "num_params","params_percent","mult_adds","trainable"])
    print("Output shape,",model(X_example[:,:-1,:]).shape)
    model.to(device)

    #Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #Run the training loop
    best_loss=1e6
    patience_counter=0
    patience=3
    loss_test=[]
    loss_train=[]
    epochs=1
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

    validate(model,test_loader,X_example.shape,args)
