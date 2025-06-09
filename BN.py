# batch_normalization_experiments.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time
from torch.utils.data import DataLoader

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 创建保存结果的目录
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# 数据加载和预处理
def load_data(batch_size=128):
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载训练集和测试集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

# 定义简化版VGG-A模型
class VGG_A(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_A, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # 阶段1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 阶段2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 阶段3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 阶段4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 阶段5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 定义带批归一化的VGG-A模型
class VGG_A_BatchNorm(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_A_BatchNorm, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # 阶段1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 阶段2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 阶段3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 阶段4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 阶段5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 训练模型并记录损失和梯度
def train_model_with_logs(model, trainloader, testloader, criterion, optimizer, epochs=20, device='cuda'):
    model.to(device)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    grad_norms = []  # 记录梯度范数
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_gradients = []
        
        for inputs, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 记录梯度范数
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm(2).item()**2
            grad_norm = grad_norm**0.5
            epoch_gradients.append(grad_norm)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 计算平均梯度范数
        avg_grad_norm = sum(epoch_gradients) / len(epoch_gradients)
        grad_norms.append(avg_grad_norm)
        
        train_loss = running_loss / len(trainloader)
        train_accuracy = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # 评估模型
        test_loss, test_accuracy = evaluate_model(model, testloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, '
              f'Grad Norm: {avg_grad_norm:.4f}')
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), f'models/vgg_a_bn_{"with" if "BatchNorm" in str(type(model)) else "without"}_bn_best.pth')
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'grad_norms': grad_norms,
        'best_accuracy': best_accuracy
    }

# 评估模型
def evaluate_model(model, testloader, criterion, device='cuda'):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(testloader)
    test_accuracy = 100.0 * correct / total
    
    return test_loss, test_accuracy

# 可视化训练结果对比
def visualize_training_comparison(results_with_bn, results_without_bn):
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线对比
    plt.subplot(1, 3, 1)
    plt.plot(results_with_bn['train_losses'], label='Train Loss (with BN)')
    plt.plot(results_with_bn['test_losses'], label='Test Loss (with BN)')
    plt.plot(results_without_bn['train_losses'], label='Train Loss (without BN)', linestyle='--')
    plt.plot(results_without_bn['test_losses'], label='Test Loss (without BN)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Comparison')
    
    # 绘制准确率曲线对比
    plt.subplot(1, 3, 2)
    plt.plot(results_with_bn['train_accuracies'], label='Train Acc (with BN)')
    plt.plot(results_with_bn['test_accuracies'], label='Test Acc (with BN)')
    plt.plot(results_without_bn['train_accuracies'], label='Train Acc (without BN)', linestyle='--')
    plt.plot(results_without_bn['test_accuracies'], label='Test Acc (without BN)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Comparison')
    
    # 绘制梯度范数对比
    plt.subplot(1, 3, 3)
    plt.plot(results_with_bn['grad_norms'], label='Gradient Norm (with BN)')
    plt.plot(results_without_bn['grad_norms'], label='Gradient Norm (without BN)')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.title('Gradient Norm Comparison')
    
    plt.tight_layout()
    plt.savefig('figures/training_comparison.png')
    plt.close()
    
    print(f"Best Test Accuracy (with BN): {results_with_bn['best_accuracy']:.2f}%")
    print(f"Best Test Accuracy (without BN): {results_without_bn['best_accuracy']:.2f}%")

# 计算损失景观
def compute_loss_landscape(model, dataloader, criterion, device='cuda', resolution=20):
    model.to(device)
    model.eval()
    
    # 获取模型参数
    params = list(model.parameters())
    param_shapes = [p.shape for p in params]
    param_sizes = [p.numel() for p in params]
    total_params = sum(param_sizes)
    
    # 保存原始参数
    original_params = [p.clone().detach() for p in params]
    
    # 创建参数扰动方向
    direction1 = [torch.randn_like(p) for p in params]
    direction2 = [torch.randn_like(p) for p in params]
    
    # 归一化方向
    for d in [direction1, direction2]:
        d_norm = sum(torch.norm(d[i])**2 for i in range(len(d)))**0.5
        for i in range(len(d)):
            d[i] = d[i] / d_norm
    
    # 计算损失景观
    x_coords = np.linspace(-1, 1, resolution)
    y_coords = np.linspace(-1, 1, resolution)
    loss_surface = np.zeros((resolution, resolution))
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # 应用扰动
            for k in range(len(params)):
                params[k].data = original_params[k] + x * direction1[k] + y * direction2[k]
            
            # 计算损失
            total_loss = 0
            num_batches = 0
            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    num_batches += 1
            
            loss_surface[j, i] = total_loss / num_batches
    
    # 恢复原始参数
    for k in range(len(params)):
        params[k].data = original_params[k]
    
    return x_coords, y_coords, loss_surface

# 可视化损失景观
def visualize_loss_landscape(loss_surface_with_bn, loss_surface_without_bn, x_coords, y_coords):
    plt.figure(figsize=(15, 6))
    
    # 绘制带BN的损失景观
    plt.subplot(1, 2, 1)
    plt.contourf(x_coords, y_coords, loss_surface_with_bn, levels=50, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('Loss Landscape (with BN)')
    
    # 绘制不带BN的损失景观
    plt.subplot(1, 2, 2)
    plt.contourf(x_coords, y_coords, loss_surface_without_bn, levels=50, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('Loss Landscape (without BN)')
    
    plt.tight_layout()
    plt.savefig('figures/loss_landscape_comparison.png')
    plt.close()

# 主函数
def main():
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    trainloader, testloader, classes = load_data()
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 实验1：VGG-A有无BN的对比
    print("="*50)
    print("Training VGG-A without Batch Normalization")
    model_without_bn = VGG_A()
    optimizer_without_bn = optim.Adam(model_without_bn.parameters(), lr=0.001)
    results_without_bn = train_model_with_logs(
        model_without_bn, trainloader, testloader, criterion, optimizer_without_bn, epochs=20, device=device
    )
    
    print("="*50)
    print("Training VGG-A with Batch Normalization")
    model_with_bn = VGG_A_BatchNorm()
    optimizer_with_bn = optim.Adam(model_with_bn.parameters(), lr=0.001)
    results_with_bn = train_model_with_logs(
        model_with_bn, trainloader, testloader, criterion, optimizer_with_bn, epochs=20, device=device
    )
    
    # 可视化训练结果对比
    visualize_training_comparison(results_with_bn, results_without_bn)
    
    # 实验2：BN如何帮助优化？损失景观分析
    print("="*50)
    print("Computing loss landscapes...")
    
    # 为了节省时间，我们使用部分数据计算损失景观
    partial_trainloader = DataLoader(trainloader.dataset, batch_size=128, shuffle=True, num_workers=2)
    
    # 加载最佳模型
    model_with_bn.load_state_dict(torch.load('models/vgg_a_bn_with_bn_best.pth'))
    model_without_bn.load_state_dict(torch.load('models/vgg_a_bn_without_bn_best.pth'))
    
    # 计算损失景观
    x_coords, y_coords, loss_surface_with_bn = compute_loss_landscape(
        model_with_bn, partial_trainloader, criterion, device=device, resolution=15
    )
    
    x_coords, y_coords, loss_surface_without_bn = compute_loss_landscape(
        model_without_bn, partial_trainloader, criterion, device=device, resolution=15
    )
    
    # 可视化损失景观
    visualize_loss_landscape(loss_surface_with_bn, loss_surface_without_bn, x_coords, y_coords)
    
    # 计算并可视化梯度变化率和最大梯度变化
    print("="*50)
    print("Experiment results saved to 'figures' directory.")

if __name__ == "__main__":
    main()