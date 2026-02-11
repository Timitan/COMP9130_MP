import torch
import torch.nn as nn

def train_model(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for images, labels in train_loader:
            
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Test Accuracy: {accuracy:.2f}%")
    
    return train_losses, test_accuracies
