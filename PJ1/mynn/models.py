from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self):
        super().__init__()

        fc1_scale = np.sqrt(2.0 / (32 * 7 * 7))
        fc2_scale = np.sqrt(2.0 / 64)
        fc1_init = lambda size: np.random.normal(size=size) * fc1_scale
        fc2_init = lambda size: np.random.normal(size=size) * fc2_scale

        self.layers = [
            conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            ReLU(),
            conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Linear(in_dim=32 * 7 * 7, out_dim=64, initialize_method=fc1_init),
            ReLU(),
            Linear(in_dim=64, out_dim=10, initialize_method=fc2_init)
        ]

        self.flatten_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        X: [batch, 1, 28, 28]
        return: logits [batch, 10]
        """
        out = X
        out = self.layers[0](out)
        out = self.layers[1](out)
        out = self.layers[2](out)
        out = self.layers[3](out)

        self.flatten_shape = out.shape
        out = out.reshape(out.shape[0], -1)

        out = self.layers[4](out)
        out = self.layers[5](out)
        out = self.layers[6](out)

        return out

    def backward(self, loss_grad):
        """
        loss_grad: [batch, 10]
        """
        grad = loss_grad

        grad = self.layers[6].backward(grad)
        grad = self.layers[5].backward(grad)
        grad = self.layers[4].backward(grad)

        grad = grad.reshape(self.flatten_shape)

        grad = self.layers[3].backward(grad)
        grad = self.layers[2].backward(grad)

        grad = self.layers[1].backward(grad)
        grad = self.layers[0].backward(grad)

        return grad
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        idx = 0
        for layer in self.layers:
            if layer.optimizable:
                layer.W = param_list[idx]['W']
                layer.b = param_list[idx]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[idx]['weight_decay']
                layer.weight_decay_lambda = param_list[idx]['lambda']
                idx += 1   
        
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)        
