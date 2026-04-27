from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        self.W = self.params['W']
        self.b = self.params['b']
        return np.matmul(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.matmul(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return np.matmul(grad, self.params['W'].T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * scale
        self.b = np.zeros((1, out_channels, 1, 1))
        self.grads = {'W': None, 'b': None}
        self.input = None # Record the input for backward process.
        self.input_padded = None
        self.input_cols = None
        self.out_h = None
        self.out_w = None
        self.col_k = None
        self.col_i = None
        self.col_j = None
        self._im2col_cache_key = None
        self._im2col_cache = None

        self.params = {'W': self.W, 'b': self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
        
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X

        W_param = self.params['W']
        b_param = self.params['b']

        batch_size, in_channels, H, W_in = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        if p > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)
        else:
            X_pad = X

        self.input_padded = X_pad

        H_pad, W_pad = X_pad.shape[2], X_pad.shape[3]
        out_h = (H_pad - k) // s + 1
        out_w = (W_pad - k) // s + 1

        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=X.dtype)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                h_end = h_start + k
                w_start = j * s
                w_end = w_start + k

                window = X_pad[:, :, h_start:h_end, w_start:w_end]

                output[:, :, i, j] = np.tensordot(window, W_param, axes=([1, 2, 3], [1, 2, 3]))

        output += b_param

        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        W_param = self.params['W']

        X = self.input
        X_pad = self.input_padded

        batch_size, in_channels, H, W_in = X.shape
        _, out_channels, out_h, out_w = grads.shape

        k = self.kernel_size
        s = self.stride
        p = self.padding

        dW = np.zeros_like(W_param)
        db = np.sum(grads, axis=(0, 2, 3), keepdims=True)
        dX_pad = np.zeros_like(X_pad)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * s
                h_end = h_start + k
                w_start = j * s
                w_end = w_start + k

                window = X_pad[:, :, h_start:h_end, w_start:w_end]

                for oc in range(out_channels):
                    dW[oc] += np.sum(window * grads[:, oc: oc+1, i: i+1, j: j+1], axis=0)

                for n in range(batch_size):
                    dX_pad[n, :, h_start:h_end, w_start:w_end] += np.sum(
                        W_param * grads[n, :, i, j].reshape(out_channels, 1, 1, 1),
                        axis=0
                    )

        self.grads['W'] = dW
        self.grads['b'] = db

        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * W_param

        if p > 0:
            return dX_pad[:, :, p:p + H, p:p + W_in]
        else:
            return dX_pad
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.labels = labels.astype(np.int64).reshape(-1)

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        batch_size = predicts.shape[0]
        probs = np.clip(self.probs, 1e-12, 1.0)
        correct_probs = probs[np.arange(batch_size), self.labels]
        return -np.mean(np.log(correct_probs))
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        batch_size = self.labels.shape[0]
        one_hot = np.zeros_like(self.probs)
        one_hot[np.arange(batch_size), self.labels] = 1
        if self.has_softmax:
            self.grads = (self.probs - one_hot) / batch_size
        else:
            self.grads = -one_hot / (np.clip(self.probs, 1e-12, 1.0) * batch_size)
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
