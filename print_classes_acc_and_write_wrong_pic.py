import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from models.modeling import VisionTransformer, CONFIGS
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    num_classes = 10
    config = CONFIGS["ViT-B_16"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionTransformer(config, 32, zero_head=True, num_classes=num_classes)
    model.load_state_dict(torch.load('./output/cifar10-5000_lr=3e-3_bz=400_scale=0.7-1.0_checkpoint.bin'))
    model.to(device)

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test)
    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=100,
                             num_workers=4,
                             pin_memory=True)
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    
    all_preds, all_label, all_pic = [], [], []
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_pic.append(x.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_pic[0] = np.append(all_pic[0], x.detach().cpu().numpy(), axis=0)
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    classes_acc = []
    for i in range(num_classes):
        is_class_i = np.logical_and(all_preds[0] == i,all_label[0] == i)
        classes_acc.append(np.sum(is_class_i) / np.sum(all_label[0] == i))
        print(classes[i]+'='+str(classes_acc[i])+'  count='+str(np.sum(all_label[0] == i)))

    w = np.argmin(classes_acc)
    is_class_i = np.logical_and(all_preds[0] != w,all_label[0] == w)
    all_pic = all_pic[0][is_class_i]

    unnormalize = transforms.Normalize(mean=[-m / s for m, s in zip((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))], std=[1 / s for s in (0.5, 0.5, 0.5)])
    for i in range(all_pic.shape[0]):
        img = unnormalize(torch.from_numpy(all_pic[i]))
        plt.imshow(np.transpose(img,(1,2,0)))
        plt.savefig('./img/'+str(i)+'.png')
