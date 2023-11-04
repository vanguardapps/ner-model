import torch.nn as nn

def default_loss_function(batch_outputs, batch_labels, batch_label_lengths):
    ce_loss = nn.CrossEntropyLoss(reduction='sum')
    sum_loss = ce_loss(batch_outputs, batch_labels.float())
    return sum_loss / batch_label_lengths.sum().float()
    
def train(model, loader, optimizer, num_epochs=5, loss_function=default_loss_function):
    for i in range(num_epochs):
        epoch_total_loss, epoch_average_batch_loss = train_epoch(model, loader, optimizer, loss_function)
        print(f"Epoch {str(i).zfill(4)} total loss: {epoch_total_loss:.2f} --- average batch loss: {epoch_average_batch_loss:.6f}")

def train_epoch(model, loader, optimizer, loss_function):
    total_loss = 0
    batch_count = 0
    for batch_inputs, batch_labels, batch_label_lengths in loader:
        # Clear gradients in all parameters passed to optimizer during instantiation
        optimizer.zero_grad()
        
        # Run forward pass through the model
        batch_outputs = model(batch_inputs)
        
        # Calculate batch loss
        batch_loss = loss_function(batch_outputs, batch_labels, batch_label_lengths)
        
        # Backpropagate
        batch_loss.backward()
        
        # Update all parameters passed to optimizer during instantiation
        optimizer.step()

        # Track overall loss
        total_loss += batch_loss
        batch_count += 1
    
    average_batch_loss = total_loss / batch_count
    return total_loss, average_batch_loss


