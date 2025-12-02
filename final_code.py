%%writefile final_cifar10n_experiments.py
# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F  # softmax, log_softmax, kl_div
import pandas as pd  # saving results to CSV

# ============================================================
# Configuration
# ============================================================
class Config:
    def __init__(self):
        self.dataset_path = './cifar-10-100n/'
        self.coteaching_path = './Co-teaching/'
        self.output_dir = './results/'

        self.num_epochs = 40          # global default epoch budget
        self.batch_size = 128
        self.learning_rate = 0.01     # baseline / coteaching LR
        self.num_classes = 10
        self.seed = 42
        self.noise_ratio = 0.4        # CIFAR-10N worst noise level
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Self-distillation hyperparameters
        self.lambda_cons = 1.0        # consistency weight
        self.ema_momentum = 0.99      # if using EMA teacher

        # Additional SD tuning knobs
        self.sd_learning_rate = 0.005     # usually a bit smaller than baseline LR
        self.sd_conf_threshold = 0.5      # only trust teacher when confident
        self.sd_teacher_temp = 0.5        # T < 1 => sharper teacher distribution
        self.sd_warmup_epochs = 5         # first few epochs: pure CE, no distillation
        self.use_ema_teacher = False      # False = keep teacher fixed (strong baseline)


# ============================================================
# Custom Dataset for CIFAR-10N labels
# ============================================================
class CIFAR10N(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False, noisy_labels=None):
        super().__init__(root, train=train, transform=transform, download=download)
        if train and noisy_labels is not None:
            if len(self.targets) != len(noisy_labels):
                raise ValueError(
                    'Mismatch in number of samples: CIFAR-10 has {} but noisy labels have {}'.format(
                        len(self.targets), len(noisy_labels)
                    )
                )
            # replace clean labels with noisy labels
            self.targets = noisy_labels.tolist()


# Wrapper to apply transforms to a subset (or any dataset) separately
class DatasetWithTransform(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.base_dataset)


# ============================================================
# Data Loading
# ============================================================
def get_data_loaders(config):
    print('Loading data...')

    # Standard CIFAR-10 normalization
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # Transforms for training (noisy data)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Transforms for validation and test (clean data)
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Load CIFAR-10N noisy labels
    noisy_labels_path = os.path.join(config.dataset_path, 'data', 'CIFAR-10_human.pt')
    if not os.path.exists(noisy_labels_path):
        print(
            'Warning: CIFAR-10N noisy labels not found at {}. Please ensure the directory structure is correct.'.format(
                noisy_labels_path
            )
        )
        print('Falling back to clean labels for demonstration purposes.')
        base_cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        worst_labels = torch.tensor(base_cifar10_train.targets)
    else:
        noisy_data = torch.load(noisy_labels_path, weights_only=False)
        worst_labels = noisy_data['worse_label']  # tensor of noisy labels

    # Base CIFAR-10 train images (no transforms yet)
    base_cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

    # Full noisy training dataset
    full_noisy_train_dataset = CIFAR10N(
        root='./data',
        train=True,
        download=False,
        transform=None,
        noisy_labels=worst_labels,
    )

    # 80/20 noisy train / noisy val split
    train_size = int(0.8 * len(full_noisy_train_dataset))
    val_size = len(full_noisy_train_dataset) - train_size

    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        full_noisy_train_dataset, [train_size, val_size], generator=generator
    )

    train_dataset_noisy = DatasetWithTransform(train_subset, transform=transform_train)
    val_dataset_noisy = DatasetWithTransform(val_subset, transform=transform_eval)

    # Clean test set
    test_dataset_clean = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_eval)

    train_loader_noisy = DataLoader(
        train_dataset_noisy, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader_noisy = DataLoader(
        val_dataset_noisy, batch_size=100, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader_clean = DataLoader(
        test_dataset_clean, batch_size=100, shuffle=False, num_workers=2, pin_memory=True
    )

    print('Train (noisy) dataset size: {}'.format(len(train_dataset_noisy)))
    print('Validation (noisy) dataset size: {}'.format(len(val_dataset_noisy)))
    print('Test (clean) dataset size: {}'.format(len(test_dataset_clean)))

    return train_loader_noisy, val_loader_noisy, test_loader_clean


# ============================================================
# Model Definition
# ============================================================
def build_resnet34(num_classes=10, device=None):
    print('Building ResNet-34 model for {} classes...'.format(num_classes))
    model = models.resnet34(weights='IMAGENET1K_V1')  # ImageNet pre-trained

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if device:
        model = model.to(device)
    return model


def build_model(config):
    return build_resnet34(num_classes=config.num_classes, device=config.device)


# ============================================================
# Utility Functions
# ============================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def evaluate(model, dataloader, device, criterion=None):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples if criterion else 0
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def update_teacher(student_model, teacher_model, ema_momentum):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(ema_momentum).add_(student_param.data, alpha=1 - ema_momentum)


# ============================================================
# Baseline Experiment
# ============================================================
def baseline_experiment(model, train_loader, val_loader, test_loader, config):
    print('Running baseline experiment...')

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0,
        'test_acc': 0.0,
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(config.output_dir, 'baseline_best.pth')

    for epoch in range(1, config.num_epochs + 1):
        print('\nEpoch {}/{}'.format(epoch, config.num_epochs))
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        val_loss, val_acc = evaluate(model, val_loader, config.device, criterion)
        scheduler.step()

        print('Train Loss: {:.4f}, Train Acc: {:.4f}'.format(train_loss, train_acc))
        print('Val Loss: {:.4f}, Val Acc: {:.4f}'.format(val_loss, val_acc))

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping / best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print('New best baseline model at epoch {}: val_acc={:.4f}'.format(epoch, val_acc))

    # Final evaluation with best checkpoint
    print('\n--- Evaluating Baseline Model (best-val checkpoint) on Clean Test Set ---')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        test_loss_best, test_acc_best = evaluate(model, test_loader, config.device, criterion)
        print(
            'Baseline (best-val checkpoint at epoch {}): Test Loss = {:.4f}, Test Acc = {:.4f}'.format(
                best_epoch, test_loss_best, test_acc_best
            )
        )
        history['test_acc'] = test_acc_best
        history['best_epoch'] = best_epoch
    else:
        print('Warning: Best model checkpoint not found. Using last epoch model for test evaluation.')
        test_loss_last, test_acc_last = evaluate(model, test_loader, config.device, criterion)
        print('Baseline (last epoch model): Test Loss = {:.4f}, Test Acc = {:.4f}'.format(test_loss_last, test_acc_last))
        history['test_acc'] = test_acc_last

    return history


# ============================================================
# Co-Teaching Experiment
# ============================================================
def coteaching_experiment(model_unused, train_loader, val_loader, test_loader, config):
    print('Running Co-teaching experiment...')

    model1 = build_model(config)
    model2 = build_model(config)

    optimizer1 = optim.SGD(model1.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(model2.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=5e-4)

    scheduler1 = StepLR(optimizer1, step_size=50, gamma=0.1)
    scheduler2 = StepLR(optimizer2, step_size=50, gamma=0.1)

    criterion_reduct_none = nn.CrossEntropyLoss(reduction='none')
    criterion_reduct_mean = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': 0.0,
    }

    remember_rate = 1 - config.noise_ratio

    for epoch in range(1, config.num_epochs + 1):
        print('\nEpoch {}/{} (Co-teaching)'.format(epoch, config.num_epochs))
        model1.train()
        model2.train()

        for batch_idx, (inputs, labels) in enumerate(
            tqdm(train_loader, desc='Co-teaching Training Epoch {}'.format(epoch))
        ):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            batch_size = inputs.size(0)

            num_remember = int(batch_size * remember_rate)
            if num_remember < 1:
                num_remember = 1

            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            loss1_per_sample = criterion_reduct_none(outputs1, labels)
            loss2_per_sample = criterion_reduct_none(outputs2, labels)

            # Model 1 learns from samples chosen by Model 2
            _, idx_remembered_2 = torch.topk(loss2_per_sample, num_remember, largest=False)
            inputs_for_model1 = inputs[idx_remembered_2]
            labels_for_model1 = labels[idx_remembered_2]
            outputs1_selected = model1(inputs_for_model1)
            loss1_selected = criterion_reduct_mean(outputs1_selected, labels_for_model1)

            optimizer1.zero_grad()
            loss1_selected.backward()
            optimizer1.step()

            # Model 2 learns from samples chosen by Model 1
            _, idx_remembered_1 = torch.topk(loss1_per_sample, num_remember, largest=False)
            inputs_for_model2 = inputs[idx_remembered_1]
            labels_for_model2 = labels[idx_remembered_1]
            outputs2_selected = model2(inputs_for_model2)
            loss2_selected = criterion_reduct_mean(outputs2_selected, labels_for_model2)

            optimizer2.zero_grad()
            loss2_selected.backward()
            optimizer2.step()

        # Epoch-end evaluation
        train_loss1, train_acc1 = evaluate(model1, train_loader, config.device, criterion_reduct_mean)
        train_loss2, train_acc2 = evaluate(model2, train_loader, config.device, criterion_reduct_mean)
        avg_train_loss_epoch = (train_loss1 + train_loss2) / 2
        avg_train_acc_epoch = (train_acc1 + train_acc2) / 2

        val_loss1, val_acc1 = evaluate(model1, val_loader, config.device, criterion_reduct_mean)
        val_loss2, val_acc2 = evaluate(model2, val_loader, config.device, criterion_reduct_mean)
        avg_val_loss_epoch = (val_loss1 + val_loss2) / 2
        avg_val_acc_epoch = (val_acc1 + val_acc2) / 2

        scheduler1.step()
        scheduler2.step()

        print('Train Loss (Avg): {:.4f}, Train Acc (Avg): {:.4f}'.format(avg_train_loss_epoch, avg_train_acc_epoch))
        print('Val Loss (Avg): {:.4f}, Val Acc (Avg): {:.4f}'.format(avg_val_loss_epoch, avg_val_acc_epoch))

        history['train_loss'].append(avg_train_loss_epoch)
        history['train_acc'].append(avg_train_acc_epoch)
        history['val_loss'].append(avg_val_loss_epoch)
        history['val_acc'].append(avg_val_acc_epoch)

    print('\n--- Evaluating Co-teaching Model 1 on Clean Test Set ---')
    test_loss1, test_acc1 = evaluate(model1, test_loader, config.device, criterion_reduct_mean)
    print('Test Loss (Model 1): {:.4f}, Test Acc (Model 1): {:.4f}'.format(test_loss1, test_acc1))
    history['test_acc'] = test_acc1

    return history


# ============================================================
# Proposed Self-Distillation Experiment (Noise-Aware)
# ============================================================
def proposed_self_distillation_experiment(model_unused, train_loader, val_loader, test_loader, config):
    print('Running proposed self-distillation experiment...')

    student_model = build_model(config)

    # --- Teacher initialization: try to use strong baseline checkpoint ---
    teacher_model = build_model(config)
    baseline_ckpt = os.path.join(config.output_dir, 'baseline_best.pth')
    if os.path.exists(baseline_ckpt):
        print('Loading baseline_best.pth into teacher (strong initialization).')
        teacher_model.load_state_dict(torch.load(baseline_ckpt))
    else:
        print('Warning: baseline_best.pth not found. Initializing teacher from student weights.')
        teacher_model.load_state_dict(student_model.state_dict())

    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    # Optim & sched for student (use potentially smaller LR)
    sd_lr = getattr(config, 'sd_learning_rate', config.learning_rate)
    student_optimizer = optim.SGD(student_model.parameters(), lr=sd_lr, momentum=0.9, weight_decay=5e-4)
    student_scheduler = StepLR(student_optimizer, step_size=50, gamma=0.1)

    criterion_ce_none = nn.CrossEntropyLoss(reduction='none')
    criterion_ce_mean = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': 0.0,
        'best_epoch': 0,
    }

    best_val_acc = 0.0
    best_epoch = 0
    best_student_path = os.path.join(config.output_dir, 'self_distill_best.pth')

    for epoch in range(1, config.num_epochs + 1):
        print('\nEpoch {}/{} (Self-Distillation)'.format(epoch, config.num_epochs))
        student_model.train()
        total_train_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (inputs, noisy_labels) in enumerate(
            tqdm(train_loader, desc='SD Training Epoch {}'.format(epoch))
        ):
            inputs, noisy_labels = inputs.to(config.device), noisy_labels.to(config.device)

            # ---- Student forward ----
            student_logits = student_model(inputs)

            # ---- Warmup: first few epochs, just supervised CE on noisy labels ----
            if epoch <= getattr(config, 'sd_warmup_epochs', 0):
                L_sup = criterion_ce_mean(student_logits, noisy_labels)
                L_cons = torch.tensor(0.0, device=config.device)
                total_loss = L_sup
            else:
                # ---- Teacher forward (fixed or EMA) ----
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                    # Sharpen the teacher distribution
                    T = getattr(config, 'sd_teacher_temp', 1.0)
                    teacher_probs = F.softmax(teacher_logits / T, dim=1)

                # Confidence for noisy labels
                ce_losses_none = criterion_ce_none(student_logits, noisy_labels)
                confidence = teacher_probs[torch.arange(inputs.size(0), device=config.device), noisy_labels]

                # Confidence thresholding (noise-aware)
                thresh = getattr(config, 'sd_conf_threshold', 0.0)
                if thresh > 0.0:
                    mask = confidence >= thresh
                    if mask.sum() == 0:
                        # fallback: if teacher is low-conf on everything, use all samples
                        mask = torch.ones_like(confidence, dtype=torch.bool)
                else:
                    mask = torch.ones_like(confidence, dtype=torch.bool)

                selected_ce = ce_losses_none[mask]
                selected_conf = confidence[mask].detach()  # do not backprop through confidence

                # Normalize weights so they don't explode
                weights = selected_conf / (selected_conf.mean() + 1e-8)
                L_sup = (weights * selected_ce).mean()

                # Consistency loss: KL(student || teacher)
                student_log_probs = F.log_softmax(student_logits / T, dim=1)
                L_cons = criterion_kl(student_log_probs, teacher_probs)

                total_loss = L_sup + config.lambda_cons * L_cons

            student_optimizer.zero_grad()
            total_loss.backward()
            student_optimizer.step()

            total_train_loss += total_loss.item() * inputs.size(0)
            _, predicted = torch.max(student_logits.data, 1)
            total_samples += noisy_labels.size(0)
            correct_predictions += (predicted == noisy_labels).sum().item()

            # Optional EMA teacher update
            if getattr(config, 'use_ema_teacher', False) and epoch > getattr(config, 'sd_warmup_epochs', 0):
                update_teacher(student_model, teacher_model, config.ema_momentum)

        avg_train_loss = total_train_loss / total_samples
        avg_train_acc = correct_predictions / total_samples

        # Evaluate on noisy validation
        val_loss, val_acc = evaluate(student_model, val_loader, config.device, criterion_ce_mean)
        student_scheduler.step()

        print('Train Loss: {:.4f}, Train Acc: {:.4f}'.format(avg_train_loss, avg_train_acc))
        print('Val Loss: {:.4f}, Val Acc: {:.4f}'.format(val_loss, val_acc))

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping / best checkpoint for self-distillation
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(student_model.state_dict(), best_student_path)
            print('New best self-distill student at epoch {}: val_acc={:.4f}'.format(epoch, val_acc))

    # --- Final evaluation: best student checkpoint on clean test set ---
    print('\n--- Evaluating Proposed Self-Distillation Model on Clean Test Set ---')
    if os.path.exists(best_student_path):
        student_model.load_state_dict(torch.load(best_student_path))
        test_loss, test_acc = evaluate(student_model, test_loader, config.device, criterion_ce_mean)
        print(
            'Self-distill (best-val checkpoint at epoch {}): Test Loss = {:.4f}, Test Acc = {:.4f}'.format(
                best_epoch, test_loss, test_acc
            )
        )
        history['test_acc'] = test_acc
        history['best_epoch'] = best_epoch
    else:
        print('Warning: best self-distill checkpoint not found. Using last epoch model.')
        test_loss, test_acc = evaluate(student_model, test_loader, config.device, criterion_ce_mean)
        print('Self-distill (last epoch): Test Loss = {:.4f}, Test Acc = {:.4f}'.format(test_loss, test_acc))
        history['test_acc'] = test_acc

    return history


# ============================================================
# Plotting & Aggregation
# ============================================================
def plot_and_aggregate_results(all_results, config):
    output_dir = config.output_dir
    print('Plotting and aggregating results...')

    os.makedirs(output_dir, exist_ok=True)

    experiments = all_results.keys()

    # --- Plotting Learning Curves ---
    metrics = ['train_acc', 'val_acc', 'train_loss', 'val_loss']
    titles = {
        'train_acc': 'Training Accuracy vs. Epoch',
        'val_acc': 'Validation Accuracy vs. Epoch',
        'train_loss': 'Training Loss vs. Epoch',
        'val_loss': 'Validation Loss vs. Epoch',
    }
    y_labels = {
        'train_acc': 'Accuracy',
        'val_acc': 'Accuracy',
        'train_loss': 'Loss',
        'val_loss': 'Loss',
    }
    filenames = {
        'train_acc': 'train_acc_comparison.png',
        'val_acc': 'val_acc_comparison.png',
        'train_loss': 'train_loss_comparison.png',
        'val_loss': 'val_loss_comparison.png',
    }

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for exp_name, hist in all_results.items():
            if metric in hist and hist[metric]:
                plot_epochs = range(1, len(hist[metric]) + 1)
                plt.plot(plot_epochs, hist[metric], label=exp_name.replace('_', ' ').title())
        plt.title(titles[metric])
        plt.xlabel('Epoch')
        plt.ylabel(y_labels[metric])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filenames[metric]))
        plt.close()

    # --- Final Test Accuracies (Bar Chart) ---
    test_accuracies = {exp: all_results[exp]['test_acc'] for exp in experiments if 'test_acc' in all_results[exp]}
    exp_names = [exp.replace('_', ' ').title() for exp in test_accuracies.keys()]
    acc_values = list(test_accuracies.values())

    if len(acc_values) > 0:
        plt.figure(figsize=(8, 6))
        bars = plt.bar(exp_names, acc_values)
        plt.ylabel('Test Accuracy')
        plt.title('Final Test Accuracy Comparison on Clean CIFAR-10')
        plt.ylim(0, 1)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, '{:.2%}'.format(yval), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_accuracy_bar.png'))
        plt.close()

    # --- Save results to JSON and CSV ---
    with open(os.path.join(output_dir, 'results_history.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    summary_data = {
        'Experiment': [],
        'Final Train Loss': [],
        'Final Train Accuracy': [],
        'Final Val Loss': [],
        'Final Val Accuracy': [],
        'Final Test Accuracy': [],
    }

    for exp_name, hist in all_results.items():
        summary_data['Experiment'].append(exp_name.replace('_', ' ').title())
        if exp_name == 'baseline' and 'best_epoch' in hist and hist['best_epoch'] > 0:
            idx = hist['best_epoch'] - 1
            summary_data['Final Train Loss'].append(hist['train_loss'][idx])
            summary_data['Final Train Accuracy'].append(hist['train_acc'][idx])
            summary_data['Final Val Loss'].append(hist['val_loss'][idx])
            summary_data['Final Val Accuracy'].append(hist['val_acc'][idx])
        elif exp_name == 'self_distillation' and 'best_epoch' in hist and hist['best_epoch'] > 0:
            idx = hist['best_epoch'] - 1
            summary_data['Final Train Loss'].append(hist['train_loss'][idx])
            summary_data['Final Train Accuracy'].append(hist['train_acc'][idx])
            summary_data['Final Val Loss'].append(hist['val_loss'][idx])
            summary_data['Final Val Accuracy'].append(hist['val_acc'][idx])
        else:
            summary_data['Final Train Loss'].append(hist['train_loss'][-1] if hist['train_loss'] else np.nan)
            summary_data['Final Train Accuracy'].append(hist['train_acc'][-1] if hist['train_acc'] else np.nan)
            summary_data['Final Val Loss'].append(hist['val_loss'][-1] if hist['val_loss'] else np.nan)
            summary_data['Final Val Accuracy'].append(hist['val_acc'][-1] if hist['val_acc'] else np.nan)
        summary_data['Final Test Accuracy'].append(hist['test_acc'])

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)

    print('Results saved to JSON and CSV.')
    print('Plots saved to PNG files.')

    print('\n========== FINAL RESULTS (CIFAR-10N worst) ==========')
    if 'baseline' in all_results:
        print('Baseline (tuned with early stopping): Test Acc = {:.2%}'.format(all_results['baseline']['test_acc']))
    if 'coteaching' in all_results:
        print('Co-Teaching:                          Test Acc = {:.2%}'.format(all_results['coteaching']['test_acc']))
    if 'self_distillation' in all_results:
        print('Proposed (self-distill):              Test Acc = {:.2%}'.format(all_results['self_distillation']['test_acc']))
    print('=====================================================')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='CIFAR-10N Experiments')
    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        choices=['all', 'baseline', 'coteaching', 'self_distillation'],
        help='Specify which experiment to run',
    )
    args = parser.parse_args()

    config = Config()

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Output directory
    os.makedirs(config.output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(config)

    results = {}

    if args.experiment == 'all' or args.experiment == 'baseline':
        print('\n--- Running Baseline Experiment ---')
        model_baseline = build_model(config)
        results['baseline'] = baseline_experiment(model_baseline, train_loader, val_loader, test_loader, config)

    if args.experiment == 'all' or args.experiment == 'coteaching':
        print('\n--- Running Co-teaching Experiment ---')
        results['coteaching'] = coteaching_experiment(None, train_loader, val_loader, test_loader, config)

    if args.experiment == 'all' or args.experiment == 'self_distillation':
        print('\n--- Running Proposed Self-Distillation Experiment ---')
        results['self_distillation'] = proposed_self_distillation_experiment(
            None, train_loader, val_loader, test_loader, config
        )

    print('\n--- Experiments Finished ---')
    if len(results) > 0:
        plot_and_aggregate_results(results, config)


if __name__ == '__main__':
    main()
