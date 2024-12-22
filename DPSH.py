from utils.tools import *
from network import *
import torch
import torch.optim as optim
import time

torch.multiprocessing.set_sharing_strategy('file_system')


# DPSH(IJCAI2016)
# paper [Feature Learning based Deep Supervised Hashing with Pairwise Labels](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf)
# code [DPSH-pytorch](https://github.com/jiangqy/DPSH-pytorch)

def get_config():
    config = {
        "alpha": 0.1,
        "need_PR": False,
        # "optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.005, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DPSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 16,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar10",
        #"dataset": "ucmd",
        #"dataset": "aid",
        "dataset": "whurs",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 150,
        "test_map": 5,
        "save_path": "save/",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [16, 32, 64],
    }
    config = config_dataset(config)
    return config

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

class DPSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPSHLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        # Assuming you know the target batch size
        #batch_size = y.shape[0]
        #y = y.reshape(config["batch_size"], config["n_class"])

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()
        #print("y shape:", y.shape)
        #print("self.Y shape:", self.Y.shape)


        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        quantization_loss = config["alpha"] * (u - u.sign()).pow(2).mean()

        return likelihood_loss + quantization_loss


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = whurs_dataset(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DPSHLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            label_onehot = EncodingOnehot(label, config["n_class"])
            #print(label_onehot.shape)
            #exit(0)

            image = image.to(device)
            label = label_onehot.to(device)

            optimizer.zero_grad()
            u = net(image)


            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.5f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)

def main():
    config = get_config()
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/alexnet/DPSH_{config['dataset']}_{bit}.json"
        train_val(config, bit)

if __name__ == "__main__":
    main()
