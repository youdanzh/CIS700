net = Net(4, 8)
xc = tr.nn.CrossEntropyLoss()
if tr.cuda.is_available():
    net= net.cuda()

opt = tr.optim.Adam(net.parameters(), lr=0.001)

num_iters = 200
verb_step = 20
train_loss = []
valid_accu =[]
valid_loss =[]
for i in trange(num_iters):

    example, label = random.choice(examples)
    if tr.cuda.is_available():
          example , label = example.cuda(), label.cuda()
    logits = net(example)
    loss = xc(logits, tr.tensor([label]))
    train_loss.append(loss.item())

    opt.zero_grad()
    loss.backward()
    opt.step()
    correct = []
    vloss = []
    
    
    if i % verb_step == 0 or i == num_iters-1:
       
        with tr.no_grad():
            for example_2, label_2 in validation:
               if tr.cuda.is_available():
                    example_2, label_2 = example_2.cuda(), label_2.cuda()
               logits = net(example_2)
               v_loss = xc(logits,tr.tensor([label_2]))  
               pred = logits.argmax()
               correct.append(np.absolute(label_2-pred))
               vloss.append(v_loss.item())
        valid_accu.append(1-np.mean(correct))
        valid_loss.append(np.mean(vloss))
        print(f'loss:{loss.item()} \t\tval_loss: {np.mean(vloss)}\t\t val_acc: {1-np.mean(correct)}  ')
        
              

#pt.plot(prediction)
pt.plot(train_loss)
pt.xlabel("Iteration")
pt.ylabel("Loss")
