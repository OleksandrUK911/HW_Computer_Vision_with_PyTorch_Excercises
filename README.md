# Computer Vision with PyTorch - Практичні вправи 🚀

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-97--99%25-brightgreen.svg)](README.md)

## 📋 Опис проекту

Комплексний практикум з **Computer Vision** та **глибокого навчання** з використанням PyTorch. Проект охоплює повний цикл розробки CNN моделі: від завантаження даних до аналізу помилок навченої моделі на датасеті MNIST.

## 🎯 Основні досягнення

- ✅ **Точність 97-99%** на тестових даних MNIST
- ✅ Реалізація архітектури **TinyVGG** (~5,000 параметрів)
- ✅ Порівняння швидкості навчання **CPU vs GPU** (3-10x прискорення)
- ✅ Детальний аналіз помилок через **матрицю плутанини**
- ✅ Експерименти з різними **kernel_size** у згорткових шарах

## 📚 Зміст роботи

### 1️⃣ Теоретична підготовка
- Дослідження застосувань Computer Vision у промисловості (медицина, автомобілебудування, retail)
- Інтерактивне вивчення CNN через [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- Розуміння процесу витягування ознак (feature extraction)

### 2️⃣ Підготовка даних
```python
# Завантаження MNIST датасету
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())

# Створення DataLoader'ів
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

**Характеристики MNIST:**
- 📊 60,000 навчальних + 10,000 тестових зображень
- 🖼️ Розмір: 28×28 пікселів (grayscale)
- 🔢 Класи: 10 цифр (0-9)

### 3️⃣ Архітектура TinyVGG

```python
class TinyVGG(nn.Module):
    """
    Компактна CNN архітектура для MNIST
    Структура: Conv-Conv-Pool → Conv-Conv-Pool → Flatten → FC → FC
    """
    def __init__(self, input_channels=1, hidden_units=10, output_classes=10):
        super().__init__()
        
        # Convolutional Block 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        )
        
        # Convolutional Block 2
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_classes)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
```

**Параметри моделі:**
- 🔧 Загальна кількість параметрів: ~5,000
- ⚡ Швидке навчання (1-10 хвилин на CPU/GPU)
- 💾 Компактний розмір моделі

### 4️⃣ Процес навчання

```python
# Конфігурація
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# Навчання
for epoch in range(epochs):
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Результати:**
- 📈 Train Accuracy: ~98-99%
- 📊 Test Accuracy: ~97-99%
- ⏱️ CPU: ~5-10 хвилин
- 🚀 GPU: ~1-2 хвилини (прискорення 3-10x)

### 5️⃣ Оцінка моделі

#### Матриця плутанини
```python
from sklearn.metrics import confusion_matrix, classification_report

# Отримання прогнозів
y_true, y_pred = get_all_predictions(model, test_loader)

# Побудова матриці
cm = confusion_matrix(y_true, y_pred)
```

**Типові плутанини:**
- 4 ↔ 9 (схожа форма)
- 3 ↔ 8 (округлі елементи)
- 5 ↔ 6 (схожі верхні частини)
- 1 ↔ 7 (вертикальні лінії)

#### Метрики класифікації
| Метрика | Значення |
|---------|----------|
| Accuracy | 97-99% |
| Precision | >0.97 |
| Recall | >0.97 |
| F1-Score | >0.97 |

### 6️⃣ Експерименти з Kernel Size

```python
# Вплив kernel_size на параметри та вихідний розмір
kernel_sizes = [1, 3, 5, 7, 9]

for k in kernel_sizes:
    conv = nn.Conv2d(3, 16, kernel_size=k, padding=0)
    output = conv(torch.randn(1, 3, 64, 64))
    print(f"Kernel {k}x{k}: {output.shape}, Params: {sum(p.numel() for p in conv.parameters())}")
```

**Ключові висновки:**
- 📦 Параметри зростають **квадратично** з kernel_size
- 📐 Без padding вихідний розмір зменшується: `output_size = input_size - kernel_size + 1`
- 👁️ Receptive field збільшується з kernel_size
- ⚡ 3×3 kernels - оптимальний баланс (стандарт індустрії)

## 🛠️ Технології та інструменти

### Основні бібліотеки
```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

### Встановлення залежностей
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib numpy scikit-learn tqdm
```

## 🚀 Швидкий старт

### 1. Клонування репозиторію
```bash
git clone https://github.com/your-username/computer-vision-pytorch.git
cd computer-vision-pytorch
```

### 2. Встановлення залежностей
```bash
pip install -r requirements.txt
```

### 3. Запуск Jupyter Notebook
```bash
jupyter notebook Computer_Vision_with_PyTorch_Excercises.ipynb
```

## 📊 Результати та візуалізації

### Навчання моделі
![Training Results](assets/training_results.png)
- Швидка конвергенція на перших 2-3 епохах
- Стабільне навчання без перенавчання
- Train і Test метрики близькі

### Матриця плутанини
![Confusion Matrix](assets/confusion_matrix.png)
- Високі значення на діагоналі (правильні прогнози)
- Типові плутанини: 4↔9, 3↔8, 5↔6

### Аналіз помилок
![Error Analysis](assets/error_analysis.png)
- Більшість помилок - амбігуозні зображення
- Низька впевненість моделі у складних випадках
- Деякі помилки складні навіть для людини

## 🎓 Навчальні цілі

Після виконання цієї роботи ви навчитеся:

1. ✅ Працювати з PyTorch та torchvision
2. ✅ Будувати власні CNN архітектури
3. ✅ Використовувати DataLoader для ефективного завантаження даних
4. ✅ Навчати моделі на CPU та GPU
5. ✅ Оцінювати якість моделі (accuracy, confusion matrix, classification report)
6. ✅ Візуалізувати результати та аналізувати помилки
7. ✅ Експериментувати з гіперпараметрами (kernel_size, padding, etc.)
8. ✅ Розуміти trade-offs у дизайні CNN

## 🔬 Експериментальні інсайти

### CPU vs GPU
| Пристрій | Час навчання (5 епох) | Прискорення |
|----------|----------------------|-------------|
| CPU | 5-10 хвилин | 1x |
| GPU | 1-2 хвилини | 3-10x |

### Kernel Size Impact
| Kernel | Параметри | Output (64x64 → ?) | Receptive Field |
|--------|-----------|-------------------|-----------------|
| 1×1 | 49 | 64×64 | 1×1 |
| 3×3 | 433 | 62×62 | 3×3 |
| 5×5 | 1,201 | 60×60 | 5×5 |
| 7×7 | 2,353 | 58×58 | 7×7 |
| 9×9 | 3,889 | 56×56 | 9×9 |

## 💡 Best Practices

### Архітектура CNN
- ✅ Використовуйте **3×3 kernels** як стандарт
- ✅ Застосовуйте **padding** для збереження розмірів
- ✅ Стекуйте маленькі kernels замість великих
- ✅ Використовуйте **1×1 convolutions** для зміни каналів
- ✅ Додавайте **ReLU** після кожної згортки

### Навчання
- ✅ Використовуйте **Adam optimizer** (lr=0.001)
- ✅ Застосовуйте **CrossEntropyLoss** для класифікації
- ✅ Встановлюйте **batch_size=32-64** для балансу
- ✅ Перемішуйте навчальні дані (**shuffle=True**)
- ✅ Моніторте обидві метрики: train і test

### Оцінка
- ✅ Візуалізуйте прогнози моделі
- ✅ Будуйте матрицю плутанини
- ✅ Аналізуйте помилкові прогнози
- ✅ Перевіряйте впевненість моделі (softmax probabilities)
- ✅ Використовуйте кілька метрик (accuracy, precision, recall, F1)

## 🤝 Внесок у проект

Вітаються будь-які покращення! Якщо ви хочете додати нові експерименти, покращити код або виправити помилки:

1. Fork репозиторію
2. Створіть нову гілку (`git checkout -b feature/amazing-feature`)
3. Commit зміни (`git commit -m 'Add amazing feature'`)
4. Push у гілку (`git push origin feature/amazing-feature`)
5. Відкрийте Pull Request

## 📝 Ліцензія

Цей проект розповсюджується під ліцензією MIT. Дивіться файл `LICENSE` для деталей.

## 🔗 Корисні посилання

### Документація
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision Datasets](https://pytorch.org/vision/stable/datasets.html)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

### Навчальні ресурси
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)

### Архітектури CNN
- [VGGNet Paper](https://arxiv.org/abs/1409.1556)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## 👨‍💻 Автор

**Олександр Скакун**

## 🙏 Подяки

- PyTorch team за чудовий framework
- CNN Explainer team за інтерактивний інструмент
- MNIST dataset creators
- Спільнота Deep Learning за безцінні ресурси

## 📧 Контакт

Якщо у вас є питання або пропозиції, створіть Issue або напишіть мені!

---

⭐ **Поставте зірку, якщо цей проект був корисним!** ⭐

