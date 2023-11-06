### 深度学习模型训练步骤
以图像分类模型为例

#### 1.加载训练和验证数据
````
# ====================================================
# Dataset 
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['id'].values
        self.labels = df[CFG.target_col].values
        self.transform = transform
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.file_path = f'../input/boolart-image-classification/train_image/{self.file_names[idx]}.jpg'
        image = np.array(Image.open(self.file_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = cv2.resize(image, (CFG.size, CFG.size))
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()
        label = torch.tensor(self.labels[idx]).long()
        return image/255, label

# ====================================================
# Transforms 定义数据增强
# ====================================================
def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
            A.Resize(CFG.size_w, CFG.size_h),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(), ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=.1),
                A.Blur(blur_limit=2, p=.1), ], p=0.2),
            A.OneOf([A.OpticalDistortion(p=0.3),
                     A.GridDistortion(p=.1),
                     A.IAAPiecewiseAffine(p=0.3), ], p=0.2),
            A.CoarseDropout(max_holes=8, max_height=int(224 * 0.1), max_width=int(224 * 0.1), p=0.5),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.size_w, CFG.size_h),
            ToTensorV2(),
        ])
````
#### 2.定义模型
````
class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained, in_chans=3)
        #print(self.model)
        if 'efficientnet' in self.cfg.model_name:
            self.n_features = self.model.classifier.in_features
            self.model.global_pool = nn.Identity()
            self.model.classifier = nn.Identity()
        elif 'resnet' in self.cfg.model_name:
            self.n_features = self.model.fc.in_features
            self.model.global_pool = nn.Identity()
            self.model.fc = nn.Identity()
        elif 'convnext' in self.cfg.model_name:
            self.n_features = self.model.head.fc.in_features
            self.model.head = nn.Identity()
            self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
                            #nn.Conv2d(self.n_features, self.n_features // 8, 1),
                            #nn.LeakyReLU(),
                            #nn.BatchNorm2d(self.n_features // 8),
                            nn.Conv2d(self.n_features, 44, 1),
                            #nn.Sigmoid()
                        )

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pool_feature = self.pooling(features)
        output = self.classifier(pool_feature).view(bs, -1)
        return output
		

# ====================================================
# loader
# ====================================================
fold = 0
folds = train
trn_idx = folds[folds['fold'] != fold].index
val_idx = folds[folds['fold'] == fold].index
train_folds = folds.loc[trn_idx].reset_index(drop=True)
valid_folds = folds.loc[val_idx].reset_index(drop=True)
valid_labels = valid_folds[CFG.target_col].values

train_dataset = TrainDataset(train_folds,
                             transform=get_transforms(data='train'))
valid_dataset = TrainDataset(valid_folds,
                             transform=get_transforms(data='valid'))

train_loader = DataLoader(train_dataset,
                          batch_size=CFG.batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
valid_loader = DataLoader(valid_dataset,
                          batch_size=CFG.batch_size * 2,
                          shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
						  
# 查看增强后的图片效果
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            img = img.permute(1,2,0).numpy()*255
            ax.imshow(img.astype(np.uint8))
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(y[i].item())
    return axes

X, y = next(iter(train_loader))
show_images(X, 8, 8, y);
````
#### 3.定义训练和验证流程
````
# ====================================================
# train,valid
# ====================================================
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    if CFG.apex:
        scaler = GradScaler()
    # switch to train mode
    model.train()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds = model(images)
        loss = criterion(y_preds, labels)

        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if CFG.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            if CFG.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Loss: {loss:.4f}'
                  .format(
                   epoch+1, step, len(train_loader),loss=loss.item(),
                   ))


def valid_fn(valid_loader, model, criterion, device):
    # switch to evaluation mode
    model.eval()
    preds = []
    acc = 0.
    for step, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # TTA
        with torch.no_grad():
            outputs1 = model(images)
            outputs2 = model(images.flip(-1))
            outputs3 = model(images.flip(-2))
            outputs4 = model(images.flip([-2, -1]))
            outputs5 = model(images.flip(-1).flip([-2, -1]))
            outputs6 = model(images.flip(-2).flip([-2, -1]))
            outputs = (outputs1 + outputs2 + outputs3 + outputs4 + outputs5 + outputs6) / 6
            loss = criterion(outputs, labels.long())
            _, predict_y = torch.max(outputs, dim=1)
            acc += (predict_y.to(device) == labels.to(device)).sum().item()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Loss: {loss:.4f}'
                  .format(step, len(valid_loader), loss=loss.item(),
                   ))
    return loss, acc
````
#### 4.加载数据、模型、优化器、学习率策略、损失函数进行训练
````
# scheduler
# ====================================================
def get_scheduler(optimizer):
    if CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True,
                                      eps=CFG.eps)
    elif CFG.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
    return scheduler

# ====================================================
# model & optimizer
# ====================================================
model = CustomModel(CFG, pretrained=True)
model.to(device)

#optimizer = SGD(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay,momentum=0.9)
optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
scheduler = get_scheduler(optimizer)

criterion = nn.CrossEntropyLoss()
best_score = 0.
best_loss = np.inf
for epoch in range(CFG.epochs):
    # train
    train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
    # eval
    avg_val_loss, acc = valid_fn(valid_loader, model, criterion, device)
    acc = acc / len(valid_dataset)
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_val_loss)
    elif isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()
    elif isinstance(scheduler, CosineAnnealingWarmRestarts):
        scheduler.step()
        
    print(f"epoch:{epoch+1}，acc:{acc}")
    if acc > best_score:
        best_score = acc
        torch.save({'model': model.state_dict(),
                    'preds': acc},
                   OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_score.pth')

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save({'model': model.state_dict(),
                    'preds': acc},
                   OUTPUT_DIR + f'{CFG.model_name}_fold{fold}_best_loss.pth')
				   
# 查看一个批次样本效果
def show_preimage(imgs,y,pre, num_rows, num_cols, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images.cpu())):
        if torch.is_tensor(img):
            # 图片张量
            img = img.permute(1,2,0).numpy()*255
            ax.imshow(img.astype(np.uint8))
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.text(30, 0, s=f"y:{y[i].item()}")
        ax.text(100, 0, s=f"pre:{pre[i].item()}")
		
images, labels = next(iter(valid_loader))
images = images.to(device)
labels = labels.to(device)
with torch.no_grad():
    outputs1 = model(images)
_, predict_y = torch.max(outputs1, dim=1)
show_preimage(images,labels,predict_y,8,8)	
````
#### 6.利用训练出来的best参数进行推理
````
TEST = '../input/boolart-image-classification/test_image/'
test_df = pd.read_csv('../input/boolart-image-classification/sample_submission.csv')

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df['id'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self.file_path = TEST + f"{self.df[idx]}.jpg"
        image = np.array(Image.open(self.file_path).convert("RGB"))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()
        return image/255,self.df[idx]
		
# 定义推理流程
def inference(model, models_path, test_loader, device):

    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    pre = []
    image_id = []
    for i, (images,img_ids) in tk0:
        image_id += list(img_ids.numpy())
        images = images.to(device)
#         avg_preds = []
        for model_path in models_path:
            model.load_state_dict(torch.load(model_path)['model'])
            model.eval()
            with torch.no_grad():
                y_preds1 = F.softmax(model(images))
                y_preds2 = F.softmax(model(images.flip(-1)))
                y_preds3 = F.softmax(model(images.flip(-2)))
                y_preds4 = F.softmax(model(images.flip([-2, -1])))
                y_preds5 = F.softmax(model(images.flip(-1).flip([-2, -1])))
                y_preds6 = F.softmax(model(images.flip(-2).flip([-2, -1])))
            y_preds = (y_preds1.to('cpu').numpy() + y_preds2.to('cpu').numpy() +
                       y_preds3.to('cpu').numpy() + y_preds4.to('cpu').numpy() + y_preds5.to(
                        'cpu').numpy() + y_preds6.to('cpu').numpy()) / 6
        avg_preds = F.softmax(torch.from_numpy(y_preds),dim=1)
        _,predict_y = torch.max(avg_preds,dim = 1)
        predict_y = np.array(predict_y).tolist()
        pre += predict_y
    return pre,image_id
	
test_dataset = TestDataset(test_df, transform=get_transforms(data='valid'))
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=True)
models_path = ['./tf_efficientnet_b2_fold0_best_score.pth']
predictions,img_id = inference(model, models_path, test_loader, device)

# submission
df = pd.DataFrame({
    "id": img_id,
    "predict": predictions
})
df.to_csv("./submission.csv", index=False)
df

````
#### 7.优化策略
1.调节CFG的基本配置，学习率大小（lr），bt大小（batch_size），学习率策略（scheduler）
2.调整backbone(当前使用model_name = 'tf_efficientnet_b2',可以调整为eff其他大小网络b1-b7,resnet50,或者convnext_small等较新的网络)
3.数据增强
4.调整TTA(减少或增加TTA), [TTA介绍](https://medium.com/analytics-vidhya/test-time-augmentation-using-pytorch-3da02d0a3188)
5.模型融合

#### 8.实验设置
product-10k作为数据集，进行图片分类，其中最后输出的target为360类。
GPU显卡2080Ti，内存42.9GB，操作系统windows

1.Resnet50 
epochs:120，batch_size:64，input_size:224*224 ,train accuracy: 52.49%
epochs:120，atch_size:128，input_size:224*224  变化不大

2.EfficientNet-B2 
epochs:10，batch_size:64，input_size:224*224，train accuracy: 74.74% (epoch超过5之后略微有下降趋势)
epochs:10，batch_size:128，input_size:224*224，变化不大
epochs:10，batch_size:64，input_size:224*224，train accuracy: 84.39% (增加训练数据集，前面实验为5.5w数据集，后增加到14w)

总结：EfficientNet-B2比Resnet50在该商品数据集上的训练效果好，另外增加训练集的提升效果明显。
