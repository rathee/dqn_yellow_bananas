
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """Q Learning Model using neural network"""

    def __init__(self, state_size, action_size, seed = 42, hidden_layers = [], drop_out = 0.2):
        """
        Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        if not hidden_layers:
            hidden_layers = [state_size*4, state_size*4]

        INPUT = state_size
        OUTPUT = action_size

        self.hidden_layers = nn.ModuleList([nn.Linear(INPUT, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(x,y) for x,y in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], OUTPUT)
        self.drop_out = drop_out

    def forward(self, state):

        for hidden_layer in self.hidden_layers:
            state = hidden_layer(state)
            state = F.relu(state)

        state = self.output(state)
        return state

if __name__ == '__main__':
    q_net = QNetwork(state_size=4, action_size=4)
    output = q_net(torch.from_numpy(np.array([[1.0,2.0,3.0,4.0], [2.0,1.0,4.0,3.0]])).float())
    print(output)
    print(output.gather(1,torch.from_numpy(np.array([[3],[2]]))))
    print(output.max(1)[0].data[0])
