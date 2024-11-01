import torch

class Net(torch.nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(Net, self).__init__()
        # variable cooling rate
        self.r = torch.nn.Parameter(data=torch.tensor([0.]))
        # network
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y, n_epochs, lr, loss_fn, loss2=None, loss2_weight=0.1):
        # convert to tensors
        X = torch.tensor(X, dtype=torch.float32).reshape(len(X), -1)
        y = torch.tensor(y, dtype=torch.float32).reshape(len(y), -1)
        # optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # train
        self.train()
        losses = []
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = self.forward(X)
            loss = loss_fn(y, output)
            if loss2 is not None:
                loss += loss2_weight * loss2(self)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, loss {loss.item()}')
        return losses
    
    def predict(self, X):
        # convert to tensor
        X = torch.tensor(X, dtype=torch.float32).reshape(len(X), -1)
        # predict
        self.eval()
        output = self.forward(X)
        return output


def grad(outputs, inputs):
    """
    Computes the partial derivative of an output with respect to an input.
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )
