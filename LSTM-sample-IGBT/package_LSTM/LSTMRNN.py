from torch import nn

# LSTM Neural Nerworks defined as follows : 
class LSTMRNN (nn.Module) :
    '''
    
    @para : 
        input_size : feature size
        hidden_size : number of hidden units
        output_size : number of output
        num_layers : layers of LSTM to stack

    '''

    def __init__(self, input_size, hidden_size = 1, output_size = 1, num_layers = 1) : 
        super().__init__()

        # utilize the LSTM model in torch.nn
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        
        # linear layer (fully connected layer)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, _x) : 

        #_x is input, size (seq_len, batch, input_size)
        x, _ = self.lstm(_x)

        # x is output, size (seq_len, batch, hidden_size)
        # but the usage of var s, b, h has not found so far
        s, b, h = x.shape
        x = self.linear(x)

        return x[-1, :, :]