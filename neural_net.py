from torch import nn
from pickle import Pickler, Unpickler
import set_info as info
import torch
import sys
import os

class NeuralNetwork(nn.Module):
    def __init__(self, net_structure):
        super().__init__()
        self.net_structure = net_structure

    def forward(self, X):
        logits = self.net_structure(X)
        return logits
    
class MLP():
    def __init__(self, name, neural_net, loss_fn, optimizer, is_classifier=True):
        self.name = name
        self.neural_net = neural_net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.is_classifier = is_classifier

        self.epochs = 0

    def train(self, training_dataloader, validation_dataloader=None, num_epochs=10, batch_print_freq=10000, save_weights=True):
        print(f'Training model for {num_epochs} epochs...')
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}\n---------------------------------")
            self._training_loop(training_dataloader, batch_print_freq)
            if save_weights:
                self.epochs += 1
                self.save_model()
            if not validation_dataloader is None:
                print('\nValidation Set:')
                accuracy, avg_loss = self.test_model(validation_dataloader)
                self.log_epoch(accuracy, avg_loss)

    def _training_loop(self, dataloader, batch_print_freq):
        size = len(dataloader.dataset)
        self.neural_net.train() # Set the net to track gradient info
        for batch_num, (X, y) in enumerate(dataloader):
            pred = self.neural_net(X)
            loss = self.loss_fn(pred, y)

            # Backpropogation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if batch_num % batch_print_freq == 0: 
                print(f'loss: {loss.item():>7.3f}  ', end='')
                if self.is_classifier:
                    accuracy = (pred.argmax(dim=1) == y).sum().item() / len(y)
                    print(f'Accuracy: {100 * accuracy:>5.1f}%  ', end='')
                print(f'[{(batch_num+1) * len(X):>7d}/{size:>7d}]')

    def test_model(self, dataloader):
        self.neural_net.eval() # Set model to not affect gradients
        
        avg_loss, accuracy = 0,0
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.neural_net(X)
                avg_loss += self.loss_fn(pred, y).item()
                if self.is_classifier:
                    accuracy += (pred.argmax(dim=1) == y).sum().item()

        # Compute summary statistics
        num_batches = len(dataloader)
        avg_loss /= num_batches

        if self.is_classifier:
            num_samples = len(dataloader.dataset)
            accuracy /= num_samples # divide by the number of samples b/c we don't divide by batchsize in for loop
            print(f'Accuracy: {(100*accuracy):>0.1f}%')
        print(f'Avg loss: {avg_loss:>8f}')
        return accuracy, avg_loss
    
    def log_epoch(self, accuracy, avg_loss):
        os.makedirs(f'models/{self.name}', exist_ok=True)
        with open(f'models/{self.name}/log.txt', "a") as log:
            log.write(f'Epoch: {self.epochs}\n')
            if self.is_classifier:
                log.write(f'Accuracy: {(100*accuracy):>0.1f}%\n')
            log.write(f'Avg loss: {avg_loss:>0.3f}\n')

    def save_model(self):
        os.makedirs(f'models/{self.name}', exist_ok=True)
        torch.save(self.neural_net.state_dict(), f'models/{self.name}/{self.epochs}.pth')
        with open(f'models/{self.name}/pickle.pickle', "wb") as metadata_file:
            Pickler(metadata_file).dump(self.epochs)
        print('Model Saved!')
    
    def load_model(self):
        try:
            with open(f'models/{self.name}/pickle.pickle', "rb") as metadata_file:
                self.epochs = Unpickler(metadata_file).load()
            self.neural_net.load_state_dict(torch.load(f'models/{self.name}/{self.epochs}.pth', map_location=info.DEVICE))
            print('Model Loaded!')
        except FileNotFoundError:
            print('Failed to load model. File not found.', file=sys.stderr)

    def __call__(self, X):
        self.neural_net.eval()
        return self.neural_net(X)

def main():
    pass

if __name__ == '__main__':
    main()