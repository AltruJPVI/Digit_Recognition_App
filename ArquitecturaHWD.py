import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.arquitectura = nn.Sequential(

            nn.Conv2d(1, 8, kernel_size=3, padding=1),  
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  

            nn.Conv2d(8,16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  


            # Flatten
            nn.Flatten(),  

            # Capas densas
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Capa de salida
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        return self.arquitectura(x)