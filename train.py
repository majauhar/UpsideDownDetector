import torch

def train(model, trainloader, optimizer, criterion, DEVICE):
    model.train()
    
    running_loss = 0
    for itr, data in enumerate(trainloader):
#         print(itr)
#         print(data[0].shape, data[1].shape)
#         print(len(trainloader))
#         if itr % 100 == 0:
#             print("itr: {}".format(itr))
        optimizer.zero_grad()
        
        imgs, target = data[0].to(DEVICE), data[1].to(DEVICE)
        output_logits = model(imgs)
        loss = criterion( output_logits, target)
        
        running_loss = loss.item()
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss/len(trainloader)
    print("epoch loss = {}".format(epoch_loss))

    return epoch_loss