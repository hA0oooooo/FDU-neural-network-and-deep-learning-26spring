# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os

# fixed seed for experiment
np.random.seed(309)
os.makedirs('./outputs', exist_ok=True)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
idx_path = os.path.join('./outputs', 'idx.pickle')
with open(idx_path, 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = (train_imgs / train_imgs.max()).astype(np.float32)
valid_imgs = (valid_imgs / valid_imgs.max()).astype(np.float32)

'''
# partA: mlp baseline

output_dir_mlp = './outputs/mlp'
os.makedirs(output_dir_mlp, exist_ok=True)

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 128, 10], 'ReLU', [1e-4, 1e-4])
optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=256)

runner.train([train_imgs, train_labs], 
             [valid_imgs, valid_labs], 
             num_epochs=10,
             log_iters=20, 
             save_dir=output_dir_mlp,
             eval_iters=20
)

train_score, train_loss = runner.evaluate([train_imgs, train_labs])
valid_score, valid_loss = runner.evaluate([valid_imgs, valid_labs])
if valid_score > runner.best_score:
    runner.best_score = valid_score

print(f"Train loss: {train_loss}")      
print(f"Train accuracy: {train_score}")
print(f"Validation loss: {valid_loss}")
print(f"Validation accuracy: {valid_score}")

metric_path = os.path.join(output_dir_mlp, 'metrics.txt')

with open(metric_path, 'w') as f:
    f.write("MLP baseline on MNIST\n")
    f.write(f"train_loss: {train_loss}\n")
    f.write(f"train_accuracy: {train_score}\n")
    f.write(f"validation_loss: {valid_loss}\n")
    f.write(f"validation_accuracy: {valid_score}\n")
    f.write(f"best_validation_accuracy: {runner.best_score}\n")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.set_tight_layout(1)

plot(runner, axes)

figure_path = os.path.join(output_dir_mlp, 'learning_curve.png')
plt.savefig(figure_path, dpi=150)
plt.show()
'''
'''
# partB: cnn

output_dir_cnn = './outputs/cnn'
os.makedirs(output_dir_cnn, exist_ok=True)

cnn_train_imgs = train_imgs.reshape(-1, 1, 28, 28)
cnn_valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

cnn_model = nn.models.Model_CNN()
cnn_optimizer = nn.optimizer.SGD(init_lr=0.06, model=cnn_model)
cnn_loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max()+1)

cnn_runner = nn.runner.RunnerM(cnn_model, cnn_optimizer, nn.metric.accuracy, cnn_loss_fn, batch_size=256)

cnn_runner.train(
    [cnn_train_imgs, train_labs],
    [cnn_valid_imgs, valid_labs],
    num_epochs=10,
    log_iters=20,
    save_dir=output_dir_cnn,
    eval_iters=20
)

cnn_train_score, cnn_train_loss = cnn_runner.evaluate([cnn_train_imgs, train_labs])
cnn_valid_score, cnn_valid_loss = cnn_runner.evaluate([cnn_valid_imgs, valid_labs])
if cnn_valid_score > cnn_runner.best_score:
    cnn_runner.best_score = cnn_valid_score

print(f"CNN Train loss: {cnn_train_loss}")
print(f"CNN Train accuracy: {cnn_train_score}")
print(f"CNN Validation loss: {cnn_valid_loss}")
print(f"CNN Validation accuracy: {cnn_valid_score}")

cnn_metric_path = os.path.join(output_dir_cnn, 'metrics.txt')

with open(cnn_metric_path, 'w') as f:
    f.write("CNN on MNIST\n")
    f.write(f"train_loss: {cnn_train_loss}\n")
    f.write(f"train_accuracy: {cnn_train_score}\n")
    f.write(f"validation_loss: {cnn_valid_loss}\n")
    f.write(f"validation_accuracy: {cnn_valid_score}\n")
    f.write(f"best_validation_accuracy: {cnn_runner.best_score}\n")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.set_tight_layout(1)

plot(cnn_runner, axes)

cnn_figure_path = os.path.join(output_dir_cnn, 'learning_curve.png')
plt.savefig(cnn_figure_path, dpi=150)
plt.show()
'''

# partC: data augmentation and robustness analysis

output_dir_aug = './outputs/cnn_dataaug'
os.makedirs(output_dir_aug, exist_ok=True)

aug_train_imgs = np.load(r'.\dataset\MNIST_augment\train_images.npy')
aug_train_labs = np.load(r'.\dataset\MNIST_augment\train_labels.npy')

idx = np.random.permutation(np.arange(aug_train_imgs.shape[0]))
aug_train_imgs = aug_train_imgs[idx]
aug_train_labs = aug_train_labs[idx]

aug_valid_imgs = aug_train_imgs[:10000]
aug_valid_labs = aug_train_labs[:10000]
aug_train_imgs = aug_train_imgs[10000:]
aug_train_labs = aug_train_labs[10000:]

aug_train_imgs = aug_train_imgs.reshape(-1, 1, 28, 28)
aug_valid_imgs = aug_valid_imgs.reshape(-1, 1, 28, 28)

aug_model = nn.models.Model_CNN()
aug_optimizer = nn.optimizer.SGD(init_lr=0.06, model=aug_model)
aug_loss_fn = nn.op.MultiCrossEntropyLoss(model=aug_model, max_classes=aug_train_labs.max()+1)

aug_runner = nn.runner.RunnerM(aug_model, aug_optimizer, nn.metric.accuracy, aug_loss_fn, batch_size=256)

aug_runner.train(
    [aug_train_imgs, aug_train_labs],
    [aug_valid_imgs, aug_valid_labs],
    num_epochs=10,
    log_iters=20,
    save_dir=output_dir_aug,
    eval_iters=20
)

aug_train_score, aug_train_loss = aug_runner.evaluate([aug_train_imgs, aug_train_labs])
aug_valid_score, aug_valid_loss = aug_runner.evaluate([aug_valid_imgs, aug_valid_labs])
if aug_valid_score > aug_runner.best_score:
    aug_runner.best_score = aug_valid_score

print(f"CNN Aug Train loss: {aug_train_loss}")
print(f"CNN Aug Train accuracy: {aug_train_score}")
print(f"CNN Aug Validation loss: {aug_valid_loss}")
print(f"CNN Aug Validation accuracy: {aug_valid_score}")

aug_metric_path = os.path.join(output_dir_aug, 'metrics.txt')

with open(aug_metric_path, 'w') as f:
    f.write("CNN with data augmentation on MNIST\n")
    f.write(f"train_loss: {aug_train_loss}\n")
    f.write(f"train_accuracy: {aug_train_score}\n")
    f.write(f"validation_loss: {aug_valid_loss}\n")
    f.write(f"validation_accuracy: {aug_valid_score}\n")
    f.write(f"best_validation_accuracy: {aug_runner.best_score}\n")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.set_tight_layout(1)

plot(aug_runner, axes)

aug_figure_path = os.path.join(output_dir_aug, 'learning_curve.png')
plt.savefig(aug_figure_path, dpi=150)
plt.show()