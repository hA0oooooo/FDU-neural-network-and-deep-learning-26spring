import argparse
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["mlp", "cnn", "cnn_dataaug"], required=True)
parser.add_argument("--data", default="clean")
parser.add_argument("--model_path", default=None)
args = parser.parse_args()

if args.model == "mlp":
        model = nn.models.Model_MLP()
        model_path = r'.\outputs\mlp\mlp.pickle'
if args.model == "cnn":
        model = nn.models.Model_CNN()
        model_path = r'.\outputs\cnn\cnn.pickle'
if args.model == "cnn_dataaug":
        model = nn.models.Model_CNN()
        model_path = r'.\outputs\cnn_dataaug\cnn_dataaug.pickle'

model.load_model(model_path)

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'
test_data_path = test_images_path

if args.data == "clean":
        with gzip.open(test_images_path, 'rb') as f:
                magic, num, rows, cols = unpack('>4I', f.read(16))
                test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
            
        with gzip.open(test_labels_path, 'rb') as f:
                magic, num = unpack('>2I', f.read(8))
                test_labs = np.frombuffer(f.read(), dtype=np.uint8)

        test_imgs = (test_imgs / test_imgs.max()).astype(np.float32)
else:
        test_data_path = fr'.\dataset\MNIST_robust\test_{args.data}.npy'
        test_imgs = np.load(test_data_path)
        test_labs = np.load(r'.\dataset\MNIST_robust\test_labels.npy')

if args.model in ["cnn", "cnn_dataaug"]:
        test_imgs = test_imgs.reshape(-1, 1, 28, 28)

logits = model(test_imgs)
accuracy = nn.metric.accuracy(logits, test_labs)
print(f"model: {args.model}")
print(f"model_path: {model_path}")
print(f"data: {args.data}")
print(f"test_data_path: {test_data_path}")
print(f"test_accuracy: {accuracy}")

output_dir = r'.\outputs'
os.makedirs(output_dir, exist_ok=True)
result_path = os.path.join(output_dir, 'test_model.csv')
write_header = not os.path.exists(result_path)
with open(result_path, 'a') as f:
        if write_header:
                f.write("model,model_path,data,test_data_path,test_accuracy\n")
        f.write(f"{args.model},{model_path},{args.data},{test_data_path},{accuracy}\n")
