import torch

def test(model, testloader, criterion, DEVICE):
    model.eval()
    test_loss, correct = 0.0, 0
    
    with torch.no_grad():
        for imgs, targets in testloader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            pred = model(imgs)
            loss = criterion(pred, targets)
            test_loss += loss.item()
            correct += (pred.argmax(1) == targets).type(torch.float).sum().item()
    
    # test_loss = test_loss / len(testloader)
    accuracy = correct / len(testloader.dataset) * 100

    return accuracy