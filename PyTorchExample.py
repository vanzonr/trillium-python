{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a4a65544-da2f-446e-975d-345118113337",
   "metadata": {},
   "source": [
    "# Example of PyTorch"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "404a994e-5707-443e-a9b7-167b17eff5aa",
   "metadata": {},
   "source": [
    "Following this tutorial: <https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "257b2fb1-2049-46f6-b3bb-a6a1f8638881",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch import nn\n",
    "from torch.utils.data import DataLoader\n",
    "from torchvision.transforms import ToTensor"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab689576-1b6c-44d8-8bce-81d8c8e50a42",
   "metadata": {},
   "source": [
    "## Write a separate script to download data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "072a4ca0-0ae8-450d-b01c-30d2823e314b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Writing 'shebang' (str) to file 'preload.py'.\n"
     ]
    }
   ],
   "source": [
    "shebang=\"#!\"+sys.executable\n",
    "%store shebang >preload.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "018fc275-fbd9-4aea-899a-03aa20730f94",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Appending to preload.py\n"
     ]
    }
   ],
   "source": [
    "%%writefile -a preload.py\n",
    "# run this on a login node in the same directory where you run this notebook\n",
    "from torchvision import datasets\n",
    "from torchvision.transforms import ToTensor\n",
    "training_data = datasets.FashionMNIST(\n",
    "    root=\"data\",\n",
    "    train=True,\n",
    "    download=True,\n",
    "    transform=ToTensor(),\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "a34bacb6-edc4-4bff-9aa5-d4606fd4b922",
   "metadata": {},
   "outputs": [],
   "source": [
    "!chmod +x preload.py"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e759994c-dd7f-451f-9983-0d274819fa01",
   "metadata": {},
   "source": [
    "This script should be run as ./preload.py once on a node with internet access (Trillium compute nodes do not have internet access)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "95453fd5-1401-4aaa-9a82-9364892597b8",
   "metadata": {},
   "source": [
    "## Load the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "cb6f0fc4-602b-40ad-a419-ccb2da7b7489",
   "metadata": {},
   "outputs": [],
   "source": [
    "# load training data from open datasets.\n",
    "training_data = datasets.FashionMNIST(\n",
    "    root=\"data\",\n",
    "    train=True,\n",
    "    download=False,\n",
    "    transform=ToTensor(),\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "2f50e9ee-c2c0-4a84-97ad-d2198becd723",
   "metadata": {},
   "outputs": [],
   "source": [
    "# load test data from open datasets.\n",
    "test_data = datasets.FashionMNIST(\n",
    "    root=\"data\",\n",
    "    train=False,\n",
    "    download=False,\n",
    "    transform=ToTensor(),\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "63159582-68ef-48ca-99e6-3872ece966bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])\n",
      "Shape of y: torch.Size([64]) torch.int64\n"
     ]
    }
   ],
   "source": [
    "batch_size = 64\n",
    "\n",
    "# Create data loaders.\n",
    "train_dataloader = DataLoader(training_data, batch_size=batch_size)\n",
    "test_dataloader = DataLoader(test_data, batch_size=batch_size)\n",
    "\n",
    "for X, y in test_dataloader:\n",
    "    print(f\"Shape of X [N, C, H, W]: {X.shape}\")\n",
    "    print(f\"Shape of y: {y.shape} {y.dtype}\")\n",
    "    break"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e131db2-c43f-4fc7-9202-d5485d7995cf",
   "metadata": {},
   "source": [
    "## Create models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "46c42865-1330-4bdd-91df-df5902baf43d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Using cuda device\n"
     ]
    }
   ],
   "source": [
    "device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else \"cpu\"\n",
    "print(f\"Using {device} device\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "cdcfe8b5-d089-42a5-a55c-df364b6dff94",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "NeuralNetwork(\n",
      "  (flatten): Flatten(start_dim=1, end_dim=-1)\n",
      "  (linear_relu_stack): Sequential(\n",
      "    (0): Linear(in_features=784, out_features=512, bias=True)\n",
      "    (1): ReLU()\n",
      "    (2): Linear(in_features=512, out_features=512, bias=True)\n",
      "    (3): ReLU()\n",
      "    (4): Linear(in_features=512, out_features=10, bias=True)\n",
      "  )\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "# Define model\n",
    "class NeuralNetwork(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.flatten = nn.Flatten()\n",
    "        self.linear_relu_stack = nn.Sequential(\n",
    "            nn.Linear(28*28, 512),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(512, 512),\n",
    "            nn.ReLU(),\n",
    "            nn.Linear(512, 10)\n",
    "        )\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.flatten(x)\n",
    "        logits = self.linear_relu_stack(x)\n",
    "        return logits\n",
    "\n",
    "model = NeuralNetwork().to(device)\n",
    "print(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "497c7ceb-89a1-410a-b54e-c4da729120b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# To train a model, we need a loss function and an optimizer.\n",
    "loss_fn = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "d67f028f-5fdb-4e9b-a17e-77071ec77e6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(dataloader, model, loss_fn, optimizer):\n",
    "    size = len(dataloader.dataset)\n",
    "    model.train()\n",
    "    for batch, (X, y) in enumerate(dataloader):\n",
    "        X, y = X.to(device), y.to(device)\n",
    "\n",
    "        # Compute prediction error\n",
    "        pred = model(X)\n",
    "        loss = loss_fn(pred, y)\n",
    "\n",
    "        # Backpropagation\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        if batch % 100 == 0:\n",
    "            loss, current = loss.item(), (batch + 1) * len(X)\n",
    "            print(f\"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "b93039e8-9f4d-4272-b6dc-b1545109ff0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def test(dataloader, model, loss_fn):\n",
    "    size = len(dataloader.dataset)\n",
    "    num_batches = len(dataloader)\n",
    "    model.eval()\n",
    "    test_loss, correct = 0, 0\n",
    "    with torch.no_grad():\n",
    "        for X, y in dataloader:\n",
    "            X, y = X.to(device), y.to(device)\n",
    "            pred = model(X)\n",
    "            test_loss += loss_fn(pred, y).item()\n",
    "            correct += (pred.argmax(1) == y).type(torch.float).sum().item()\n",
    "    test_loss /= num_batches\n",
    "    correct /= size\n",
    "    print(f\"Test Error: \\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "1432f71b-813e-40f6-9d16-fd90832f70d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1\n",
      "-------------------------------\n",
      "loss: 2.298294  [   64/60000]\n",
      "loss: 2.281897  [ 6464/60000]\n",
      "loss: 2.253053  [12864/60000]\n",
      "loss: 2.248230  [19264/60000]\n",
      "loss: 2.243371  [25664/60000]\n",
      "loss: 2.188479  [32064/60000]\n",
      "loss: 2.214804  [38464/60000]\n",
      "loss: 2.162189  [44864/60000]\n",
      "loss: 2.160650  [51264/60000]\n",
      "loss: 2.126628  [57664/60000]\n",
      "Test Error: \n",
      " Accuracy: 43.5%, Avg loss: 2.116676 \n",
      "\n",
      "Epoch 2\n",
      "-------------------------------\n",
      "loss: 2.126642  [   64/60000]\n",
      "loss: 2.112704  [ 6464/60000]\n",
      "loss: 2.037798  [12864/60000]\n",
      "loss: 2.064141  [19264/60000]\n",
      "loss: 2.000686  [25664/60000]\n",
      "loss: 1.924183  [32064/60000]\n",
      "loss: 1.970937  [38464/60000]\n",
      "loss: 1.864061  [44864/60000]\n",
      "loss: 1.878115  [51264/60000]\n",
      "loss: 1.805702  [57664/60000]\n",
      "Test Error: \n",
      " Accuracy: 52.9%, Avg loss: 1.797026 \n",
      "\n",
      "Epoch 3\n",
      "-------------------------------\n",
      "loss: 1.833791  [   64/60000]\n",
      "loss: 1.799363  [ 6464/60000]\n",
      "loss: 1.663713  [12864/60000]\n",
      "loss: 1.725666  [19264/60000]\n",
      "loss: 1.602829  [25664/60000]\n",
      "loss: 1.560593  [32064/60000]\n",
      "loss: 1.602108  [38464/60000]\n",
      "loss: 1.490670  [44864/60000]\n",
      "loss: 1.529316  [51264/60000]\n",
      "loss: 1.435435  [57664/60000]\n",
      "Test Error: \n",
      " Accuracy: 61.3%, Avg loss: 1.443326 \n",
      "\n",
      "Epoch 4\n",
      "-------------------------------\n",
      "loss: 1.512941  [   64/60000]\n",
      "loss: 1.479608  [ 6464/60000]\n",
      "loss: 1.321832  [12864/60000]\n",
      "loss: 1.412202  [19264/60000]\n",
      "loss: 1.288858  [25664/60000]\n",
      "loss: 1.293962  [32064/60000]\n",
      "loss: 1.323267  [38464/60000]\n",
      "loss: 1.239295  [44864/60000]\n",
      "loss: 1.280720  [51264/60000]\n",
      "loss: 1.197617  [57664/60000]\n",
      "Test Error: \n",
      " Accuracy: 63.5%, Avg loss: 1.210305 \n",
      "\n",
      "Epoch 5\n",
      "-------------------------------\n",
      "loss: 1.285093  [   64/60000]\n",
      "loss: 1.268478  [ 6464/60000]\n",
      "loss: 1.098259  [12864/60000]\n",
      "loss: 1.215787  [19264/60000]\n",
      "loss: 1.089896  [25664/60000]\n",
      "loss: 1.120569  [32064/60000]\n",
      "loss: 1.153553  [38464/60000]\n",
      "loss: 1.083229  [44864/60000]\n",
      "loss: 1.124941  [51264/60000]\n",
      "loss: 1.055504  [57664/60000]\n",
      "Test Error: \n",
      " Accuracy: 64.6%, Avg loss: 1.063907 \n",
      "\n",
      "Done!\n"
     ]
    }
   ],
   "source": [
    "epochs = 5\n",
    "for t in range(epochs):\n",
    "    print(f\"Epoch {t+1}\\n-------------------------------\")\n",
    "    train(train_dataloader, model, loss_fn, optimizer)\n",
    "    test(test_dataloader, model, loss_fn)\n",
    "print(\"Done!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "07a2b63f-ade1-4add-bd52-fb85fb338380",
   "metadata": {},
   "source": [
    "## Saving Models"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "013220e8-9ca9-4e94-bcc8-b668887ab122",
   "metadata": {},
   "source": [
    "A common way to save a model is to serialize the internal state dictionary (containing the model parameters)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "8b5304b6-3016-4a07-b905-f6c851c535f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saved PyTorch Model State to model.pth\n"
     ]
    }
   ],
   "source": [
    "torch.save(model.state_dict(), \"model.pth\")\n",
    "print(\"Saved PyTorch Model State to model.pth\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7409b4fb-120f-4a19-86fb-357d6cc8a5db",
   "metadata": {},
   "source": [
    "## Loading Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "05bcc230-12e5-4c32-8c10-ef2c0fbaa5cf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = NeuralNetwork().to(device)\n",
    "model.load_state_dict(torch.load(\"model.pth\", weights_only=True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "630585fe-e7d5-4917-ace9-8b585319b094",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted: \"Ankle boot\", Actual: \"Ankle boot\"\n"
     ]
    }
   ],
   "source": [
    "classes = [\n",
    "    \"T-shirt/top\",\n",
    "    \"Trouser\",\n",
    "    \"Pullover\",\n",
    "    \"Dress\",\n",
    "    \"Coat\",\n",
    "    \"Sandal\",\n",
    "    \"Shirt\",\n",
    "    \"Sneaker\",\n",
    "    \"Bag\",\n",
    "    \"Ankle boot\",\n",
    "]\n",
    "\n",
    "model.eval()\n",
    "x, y = test_data[0][0], test_data[0][1]\n",
    "with torch.no_grad():\n",
    "    x = x.to(device)\n",
    "    pred = model(x)\n",
    "    predicted, actual = classes[pred[0].argmax(0)], classes[y]\n",
    "    print(f'Predicted: \"{predicted}\", Actual: \"{actual}\"')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30a29307-9221-42da-8c78-2c1a5c75be46",
   "metadata": {},
   "source": [
    "test_data[0][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "fa3090e8-2ee4-4923-8bda-3b80451e46f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "160899c9-4bf3-485c-b94f-562dd9975e12",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x1531b85a0410>"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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/wguLy9PqampF7yLpbm5WYWFhUZdDQ85OTmaPn26GhoarFsx8/k5wPlxoalTpyovL29Unh9r167Ve++9pw8//DDuz7cUFhaqu7tbLS0tcfuP1vNhsHW4mPLyckkaVufDsA+gjIwMzZkzR9XV1QO39ff3q7q6WvPmzTPszF5bW5uOHTumoqIi61bMlJaWqrCwMO78iMVi2rdv31V/fpw8eVJnz54dVeeHc05r167V9u3btXv3bpWWlsbdP2fOHKWnp8edD/X19Tp+/PioOh8utQ4Xc+jQIUkaXueD9bsgLsebb77pwuGw27p1q/vTn/7kHn74YZeTk+OampqsWxtS3//+911NTY1rbGx0v//9711FRYXLy8tzZ86csW4tqVpbW93BgwfdwYMHnST3wgsvuIMHD7q///3vzjnnfvrTn7qcnBy3c+dOd/jwYbds2TJXWlrqzp07Z9x5Yn3ZOrS2trrHH3/c1dXVucbGRvfBBx+4r33ta+766693nZ2d1q0nzJo1a1wkEnE1NTXu9OnTA1tHR8fAPo888oibPHmy2717t9u/f7+bN2+emzdvnmHXiXepdWhoaHA/+tGP3P79+11jY6PbuXOnmzp1qps/f75x5/FGRAA559zLL7/sJk+e7DIyMtzcuXPd3r17rVsacvfcc48rKipyGRkZ7itf+Yq75557XENDg3VbSffhhx86SRdsK1eudM6dfyv2008/7QoKClw4HHYLFy509fX1tk0nwZetQ0dHh1u0aJGbOHGiS09Pd1OmTHGrV68edd+kXezzl+S2bNkysM+5c+fcd7/7XXfNNde4sWPHurvuusudPn3arukkuNQ6HD9+3M2fP9/l5ua6cDjsrrvuOveDH/zARaNR28a/gD/HAAAwMexfAwIAjE4EEADABAEEADBBAAEATBBAAAATBBAAwAQBBAAwQQABAEwQQAAAEwQQAMAEAQQAMEEAAQBM/D+XLqEe7kRv9AAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(test_data[0][0][0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "37ea30ef-1c6a-4a72-a84c-8ca77e3f5119",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([9, 0, 0,  ..., 3, 0, 5])\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/rzon/.virtualenvs/torch29/lib/python3.13/site-packages/torchvision/datasets/mnist.py:66: UserWarning: train_labels has been renamed targets\n",
      "  warnings.warn(\"train_labels has been renamed targets\")\n"
     ]
    }
   ],
   "source": [
    "print(training_data.train_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "faf53921-9cbc-476c-b8f5-01d76fe0404b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "raw",
   "id": "80c434d2-a399-47ef-a38a-7bb58a660f13",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "torch29",
   "language": "python",
   "name": "torch29"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
