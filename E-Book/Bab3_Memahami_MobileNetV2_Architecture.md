# BAB 3: Memahami MobileNetV2 Architecture

## 3.1 Sejarah dan Evolusi MobileNet

### Latar Belakang MobileNet
MobileNet adalah keluarga arsitektur neural network yang dikembangkan oleh Google khusus untuk aplikasi mobile dan embedded systems. Arsitektur ini dirancang untuk memberikan performa yang baik dengan ukuran model yang kecil dan komputasi yang efisien.

### Timeline Evolusi MobileNet

#### MobileNetV1 (2017)
- **Penulis**: Andrew G. Howard et al.
- **Inovasi Utama**: Depthwise Separable Convolutions
- **Keunggulan**: Mengurangi parameter dan komputasi secara signifikan
- **Kelemahan**: Akurasi yang sedikit menurun dibanding model konvensional

```
Traditional Convolution vs Depthwise Separable Convolution:

Traditional: 
Input -> 3x3 Conv (all channels) -> Output

Depthwise Separable:
Input -> 3x3 Depthwise Conv -> 1x1 Pointwise Conv -> Output
```

#### MobileNetV2 (2018)
- **Penulis**: Mark Sandler et al.
- **Inovasi Utama**: Inverted Residual Blocks dengan Linear Bottlenecks
- **Keunggulan**: 
  - Akurasi lebih tinggi dari V1
  - Memory efisien
  - Gradient flow yang lebih baik
- **Penggunaan**: Menjadi backbone populer untuk transfer learning

#### MobileNetV3 (2019)
- **Inovasi**: Neural Architecture Search (NAS) dan Hard-Swish activation
- **Keunggulan**: Performa terbaik dalam keluarga MobileNet

### Mengapa MobileNetV2?
Dalam proyek AI pemula, MobileNetV2 menjadi pilihan ideal karena:

1. **Balance yang Optimal**: Memberikan keseimbangan terbaik antara akurasi dan efisiensi
2. **Transfer Learning Friendly**: Pre-trained weights yang sangat baik
3. **Dokumentasi Lengkap**: Dukungan komunitas yang kuat
4. **Framework Support**: Tersedia di semua framework deep learning populer

## 3.2 Arsitektur MobileNetV2

### Struktur Keseluruhan

MobileNetV2 terdiri dari 19 layer dengan total 3.4 juta parameter. Mari kita breakdown strukturnya:

```
Input Layer (224x224x3)
    ↓
Conv2D 32 filters (112x112x32)
    ↓
Inverted Residual Block 1 (112x112x16)
    ↓
Inverted Residual Block 2 (56x56x24)
    ↓
Inverted Residual Block 3 (28x28x32)
    ↓
Inverted Residual Block 4 (14x14x64)
    ↓
Inverted Residual Block 5 (14x14x96)
    ↓
Inverted Residual Block 6 (7x7x160)
    ↓
Inverted Residual Block 7 (7x7x320)
    ↓
Conv2D 1280 filters (7x7x1280)
    ↓
Global Average Pooling (1x1x1280)
    ↓
Dense 1000 classes (1000)
```

### Layer Detail dan Spesifikasi

| Layer | Input Size | Operator | Expansion | Output Channels | Stride |
|-------|------------|----------|-----------|-----------------|--------|
| 1 | 224² × 3 | conv2d | - | 32 | 2 |
| 2 | 112² × 32 | bottleneck | 1 | 16 | 1 |
| 3 | 112² × 16 | bottleneck | 6 | 24 | 2 |
| 4 | 56² × 24 | bottleneck | 6 | 24 | 1 |
| 5 | 56² × 24 | bottleneck | 6 | 32 | 2 |
| 6 | 28² × 32 | bottleneck | 6 | 32 | 1 |
| 7 | 28² × 32 | bottleneck | 6 | 32 | 1 |
| 8 | 28² × 32 | bottleneck | 6 | 64 | 2 |
| 9-11 | 14² × 64 | bottleneck | 6 | 64 | 1 |
| 12 | 14² × 64 | bottleneck | 6 | 96 | 1 |
| 13-14 | 14² × 96 | bottleneck | 6 | 96 | 1 |
| 15 | 14² × 96 | bottleneck | 6 | 160 | 2 |
| 16-17 | 7² × 160 | bottleneck | 6 | 160 | 1 |
| 18 | 7² × 160 | bottleneck | 6 | 320 | 1 |
| 19 | 7² × 320 | conv2d 1×1 | - | 1280 | 1 |
| 20 | 7² × 1280 | avgpool 7×7 | - | - | 1 |
| 21 | 1 × 1 × 1280 | conv2d 1×1 | - | k | - |

### Implementasi dalam Keras/TensorFlow

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# Memuat model pre-trained
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    alpha=1.0,  # Width multiplier
    include_top=True,  # Include final classification layer
    weights='imagenet'  # Pre-trained weights
)

# Melihat summary arsitektur
base_model.summary()

# Output example:
# Total params: 3,538,984
# Trainable params: 3,504,872
# Non-trainable params: 34,112
```

### Width Multiplier (Alpha)
MobileNetV2 memiliki parameter `alpha` yang mengontrol lebar network:

- **alpha=1.0**: Model penuh (default)
- **alpha=0.75**: 75% dari channel count
- **alpha=0.5**: 50% dari channel count
- **alpha=0.35**: Model terkecil

```python
# Contoh penggunaan width multiplier
model_full = MobileNetV2(alpha=1.0, weights='imagenet')
model_small = MobileNetV2(alpha=0.5, weights='imagenet')

print(f"Full model params: {model_full.count_params():,}")
print(f"Small model params: {model_small.count_params():,}")

# Output:
# Full model params: 3,538,984
# Small model params: 1,968,088
```

## 3.3 Inverted Residual Blocks

### Konsep Dasar

Inverted Residual Block adalah inovasi utama MobileNetV2. Mari kita pahami perbedaannya dengan Residual Block tradisional:

#### Traditional Residual Block (ResNet)
```
Input (256 channels)
    ↓
1x1 Conv (64 channels) - Compress
    ↓
3x3 Conv (64 channels) - Process
    ↓
1x1 Conv (256 channels) - Expand
    ↓
Add with Input (Skip Connection)
    ↓
ReLU Activation
```

#### Inverted Residual Block (MobileNetV2)
```
Input (24 channels)
    ↓
1x1 Conv (144 channels) - Expand
    ↓
3x3 Depthwise Conv (144 channels) - Process
    ↓
1x1 Conv (24 channels) - Compress
    ↓
Add with Input (Skip Connection)
```

### Struktur Detail Inverted Residual Block

```python
def inverted_residual_block(x, expansion_factor, output_channels, stride):
    """
    Implementasi Inverted Residual Block
    
    Args:
        x: Input tensor
        expansion_factor: Faktor ekspansi (biasanya 6)
        output_channels: Jumlah output channels
        stride: Stride untuk depthwise convolution
    """
    # 1. Expansion Layer (1x1 Convolution)
    input_channels = x.shape[-1]
    expanded_channels = input_channels * expansion_factor
    
    expanded = tf.keras.layers.Conv2D(
        expanded_channels, 
        kernel_size=1, 
        padding='same',
        use_bias=False
    )(x)
    expanded = tf.keras.layers.BatchNormalization()(expanded)
    expanded = tf.keras.layers.ReLU(max_value=6)(expanded)  # ReLU6
    
    # 2. Depthwise Convolution
    depthwise = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding='same',
        use_bias=False
    )(expanded)
    depthwise = tf.keras.layers.BatchNormalization()(depthwise)
    depthwise = tf.keras.layers.ReLU(max_value=6)(depthwise)
    
    # 3. Projection Layer (Linear Bottleneck)
    projected = tf.keras.layers.Conv2D(
        output_channels,
        kernel_size=1,
        padding='same',
        use_bias=False
    )(depthwise)
    projected = tf.keras.layers.BatchNormalization()(projected)
    # Note: No activation function here (Linear Bottleneck)
    
    # 4. Residual Connection (if stride=1 and same channels)
    if stride == 1 and input_channels == output_channels:
        return tf.keras.layers.Add()([x, projected])
    else:
        return projected
```

### Keunggulan Inverted Residual Block

#### 1. **Memory Efficiency**
```python
# Analisis memory usage
def calculate_memory_usage():
    # Traditional ResNet Block
    # Input: 56x56x256 = 802,816 values
    # Bottleneck: 56x56x64 = 200,704 values
    # Memory peak: 802,816 values
    
    # Inverted Residual Block
    # Input: 56x56x24 = 75,264 values
    # Expanded: 56x56x144 = 451,584 values
    # Output: 56x56x24 = 75,264 values
    # Memory peak: 451,584 values (masih lebih kecil)
    
    resnet_memory = 802_816
    mobilenet_memory = 451_584
    
    print(f"ResNet memory: {resnet_memory:,}")
    print(f"MobileNet memory: {mobilenet_memory:,}")
    print(f"Memory reduction: {(1 - mobilenet_memory/resnet_memory)*100:.1f}%")

calculate_memory_usage()
```

#### 2. **Linear Bottleneck**
```python
# Mengapa tidak ada ReLU di projection layer?
def demonstrate_linear_bottleneck():
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Simulasi informasi yang hilang dengan ReLU
    x = np.linspace(-3, 3, 1000)
    
    # Original manifold in high-dimensional space
    high_dim = np.sin(x) + 0.5 * np.cos(2*x)
    
    # Projection to low-dimensional space with ReLU
    projected_relu = np.maximum(0, high_dim)
    
    # Projection to low-dimensional space without ReLU (linear)
    projected_linear = high_dim
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x, high_dim)
    plt.title('Original High-Dim Manifold')
    
    plt.subplot(1, 3, 2)
    plt.plot(x, projected_relu)
    plt.title('With ReLU (Information Loss)')
    
    plt.subplot(1, 3, 3)
    plt.plot(x, projected_linear)
    plt.title('Linear Bottleneck (No Loss)')
    
    plt.tight_layout()
    plt.savefig('linear_bottleneck_comparison.png')
    plt.show()

# demonstrate_linear_bottleneck()
```

#### 3. **Gradient Flow**
Linear bottleneck memungkinkan gradient mengalir lebih baik karena tidak ada saturasi dari ReLU activation.

### Implementasi Praktis

```python
# Contoh penggunaan dalam transfer learning
def create_custom_mobilenetv2():
    # Base model tanpa top layer
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze beberapa layer pertama
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Tambahkan custom classifier
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes
    ])
    
    return model

# custom_model = create_custom_mobilenetv2()
# custom_model.summary()
```

## 3.4 Pre-trained Weights dengan ImageNet

### Apa itu ImageNet?

ImageNet adalah dataset image classification terbesar dan paling berpengaruh dalam computer vision:

- **Total Images**: 14 juta gambar
- **Training Set**: 1.28 juta gambar
- **Validation Set**: 50,000 gambar
- **Test Set**: 100,000 gambar
- **Classes**: 1,000 kategori objek
- **Hierarchy**: Berdasarkan WordNet ontology

### Proses Training MobileNetV2 pada ImageNet

```python
# Cara Google melatih MobileNetV2 (simplified version)
def train_mobilenetv2_on_imagenet():
    """
    Training configuration yang digunakan Google
    """
    training_config = {
        'optimizer': 'SGD',
        'learning_rate': 0.045,
        'momentum': 0.9,
        'weight_decay': 0.00004,
        'batch_size': 96,
        'epochs': 600,
        'learning_rate_schedule': 'cosine_decay',
        'data_augmentation': [
            'random_crop',
            'random_flip',
            'color_jitter',
            'auto_augment'
        ]
    }
    
    return training_config

config = train_mobilenetv2_on_imagenet()
print("MobileNetV2 Training Configuration:")
for key, value in config.items():
    print(f"{key}: {value}")
```

### Kualitas Pre-trained Weights

#### Performance Metrics
```python
# Performance MobileNetV2 pada ImageNet
imagenet_performance = {
    'top1_accuracy': 71.8,  # %
    'top5_accuracy': 90.6,  # %
    'parameters': 3.4,      # Million
    'mult_adds': 300,       # Million
    'model_size': 14        # MB
}

# Bandingkan dengan arsitektur lain
comparison = {
    'ResNet50': {'top1': 76.0, 'params': 25.6, 'size': 102},
    'InceptionV3': {'top1': 78.0, 'params': 23.8, 'size': 95},
    'MobileNetV2': {'top1': 71.8, 'params': 3.4, 'size': 14},
    'EfficientNetB0': {'top1': 77.3, 'params': 5.3, 'size': 21}
}

print("Model Comparison:")
for model, metrics in comparison.items():
    print(f"{model}: {metrics['top1']:.1f}% accuracy, {metrics['params']:.1f}M params, {metrics['size']}MB")
```

### Transfer Learning dengan Pre-trained Weights

#### 1. **Feature Extraction**
```python
def feature_extraction_approach():
    """
    Menggunakan MobileNetV2 sebagai fixed feature extractor
    """
    # Load pre-trained model tanpa classifier
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze semua layer
    base_model.trainable = False
    
    # Tambahkan classifier baru
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')  # 5 classes
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

#### 2. **Fine-tuning**
```python
def fine_tuning_approach():
    """
    Fine-tuning beberapa layer teratas
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze layer bawah, unfreeze layer atas
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    # Learning rate yang lebih kecil untuk fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### Visualisasi Features yang Dipelajari

```python
def visualize_learned_features():
    """
    Visualisasi apa yang dipelajari oleh layer-layer MobileNetV2
    """
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    # Load model
    model = MobileNetV2(weights='imagenet', include_top=False)
    
    # Pilih layer untuk divisualisasi
    layer_names = [
        'block_1_expand_relu',   # Early features
        'block_6_expand_relu',   # Mid-level features
        'block_13_expand_relu',  # High-level features
        'out_relu'               # Final features
    ]
    
    # Buat model untuk ekstraksi features
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Load dan preprocess image
    img_path = 'sample_image.jpg'  # Ganti dengan path image Anda
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Ekstraksi features
    activations = activation_model.predict(img_array)
    
    # Visualisasi
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, (layer_name, activation) in enumerate(zip(layer_names, activations)):
        # Pilih beberapa feature maps
        feature_map = activation[0, :, :, :8]  # 8 feature maps pertama
        
        for j in range(min(8, feature_map.shape[-1])):
            ax = axes[i//2, j if i < 2 else j-4]
            ax.imshow(feature_map[:, :, j], cmap='viridis')
            ax.set_title(f'{layer_name}\nFeature {j+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mobilenetv2_feature_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

# visualize_learned_features()
```

## 3.5 Cara Kerja decode_predictions()

### Fungsi dan Tujuan

`decode_predictions()` adalah utility function yang mengkonversi output numerik dari model menjadi label yang dapat dibaca manusia, lengkap dengan confidence score.

```python
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
import numpy as np

# Contoh output raw dari model
raw_predictions = np.array([[
    0.001, 0.002, 0.003, ..., 0.85, 0.12, 0.025, ...  # 1000 values
]])

# Decode ke label yang readable
decoded = decode_predictions(raw_predictions, top=5)
print(decoded)

# Output:
# [[('n02123045', 'tabby', 0.85),
#   ('n02123159', 'tiger_cat', 0.12),
#   ('n02124075', 'Egyptian_cat', 0.025),
#   ('n02123394', 'Persian_cat', 0.002),
#   ('n02127052', 'lynx', 0.001)]]
```

### Struktur Internal decode_predictions()

#### 1. **ImageNet Class Index**
```python
# Simplified version of class index mapping
imagenet_class_index = {
    0: ['n01440764', 'tench'],
    1: ['n01443537', 'goldfish'],
    2: ['n01484850', 'great_white_shark'],
    # ... total 1000 classes
    997: ['n15075141', 'toilet_tissue'],
    998: ['n99999999', 'sports_car'],
    999: ['n04579432', 'van']
}

def load_imagenet_class_index():
    """
    Fungsi ini memuat mapping antara index dan label
    """
    try:
        # TensorFlow menyimpan mapping ini dalam JSON
        import json
        import urllib.request
        
        url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        with urllib.request.urlopen(url) as response:
            class_index = json.loads(response.read().decode())
        
        return class_index
    except:
        return imagenet_class_index

class_index = load_imagenet_class_index()
print(f"Total classes: {len(class_index)}")
print(f"First 5 classes: {dict(list(class_index.items())[:5])}")
```

#### 2. **Implementasi decode_predictions()**
```python
def custom_decode_predictions(preds, top=5, class_list_path=None):
    """
    Implementasi custom decode_predictions untuk memahami cara kerjanya
    
    Args:
        preds: Predictions array dengan shape (batch_size, 1000)
        top: Jumlah top predictions yang dikembalikan
        class_list_path: Path ke file class index (optional)
    """
    import numpy as np
    import json
    
    # Load class index
    if class_list_path:
        with open(class_list_path) as f:
            class_index = json.load(f)
    else:
        class_index = load_imagenet_class_index()
    
    results = []
    
    for pred in preds:
        # Get top predictions
        top_indices = pred.argsort()[-top:][::-1]  # Descending order
        
        result = []
        for i in top_indices:
            class_id = class_index[str(i)][0]  # WordNet ID
            class_name = class_index[str(i)][1]  # Human readable name
            score = pred[i]
            
            result.append((class_id, class_name, score))
        
        results.append(result)
    
    return results

# Test custom implementation
test_predictions = np.random.rand(1, 1000)
test_predictions = test_predictions / test_predictions.sum()  # Normalize to probabilities

custom_result = custom_decode_predictions(test_predictions, top=3)
print("Custom decode result:")
for item in custom_result[0]:
    print(f"  {item[1]}: {item[2]:.4f}")
```

### WordNet ID System

#### Struktur WordNet ID
```python
def explain_wordnet_id():
    """
    Penjelasan sistem WordNet ID yang digunakan ImageNet
    """
    examples = {
        'n02123045': {
            'name': 'tabby',
            'explanation': 'n = noun, 02123045 = unique identifier',
            'hierarchy': ['entity', 'physical_entity', 'object', 'living_thing', 'organism', 'animal', 'chordate', 'vertebrate', 'mammal', 'carnivore', 'feline', 'cat', 'domestic_cat', 'tabby']
        },
        'n04579432': {
            'name': 'van',
            'explanation': 'n = noun, 04579432 = unique identifier',
            'hierarchy': ['entity', 'physical_entity', 'object', 'artifact', 'instrumentality', 'container', 'wheeled_vehicle', 'motor_vehicle', 'car', 'van']
        }
    }
    
    for wordnet_id, info in examples.items():
        print(f"\nWordNet ID: {wordnet_id}")
        print(f"Name: {info['name']}")
        print(f"Explanation: {info['explanation']}")
        print(f"Hierarchy: {' -> '.join(info['hierarchy'])}")

explain_wordnet_id()
```

### Practical Usage Examples

#### 1. **Basic Image Classification**
```python
def classify_single_image(image_path):
    """
    Klasifikasi gambar tunggal dengan decode_predictions
    """
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    # Load model
    model = MobileNetV2(weights='imagenet')
    
    # Load dan preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array)
    
    # Decode predictions
    decoded = decode_predictions(predictions, top=5)[0]
    
    print(f"Predictions for {image_path}:")
    for i, (class_id, class_name, score) in enumerate(decoded):
        print(f"  {i+1}. {class_name}: {score:.4f} ({score*100:.2f}%)")
    
    return decoded

# classify_single_image('path/to/your/image.jpg')
```

#### 2. **Batch Processing**
```python
def classify_multiple_images(image_paths):
    """
    Klasifikasi multiple images sekaligus
    """
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    model = MobileNetV2(weights='imagenet')
    
    # Prepare batch
    batch_images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        batch_images.append(img_array)
    
    # Convert to batch
    batch_array = np.array(batch_images)
    batch_array = preprocess_input(batch_array)
    
    # Batch prediction
    batch_predictions = model.predict(batch_array)
    
    # Decode all predictions
    all_decoded = decode_predictions(batch_predictions, top=3)
    
    # Print results
    for i, (img_path, decoded) in enumerate(zip(image_paths, all_decoded)):
        print(f"\n{img_path}:")
        for class_id, class_name, score in decoded:
            print(f"  {class_name}: {score:.4f}")
    
    return all_decoded

# image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
# classify_multiple_images(image_list)
```

#### 3. **Custom Threshold Processing**
```python
def classify_with_confidence_threshold(image_path, threshold=0.1):
    """
    Klasifikasi dengan confidence threshold
    """
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    model = MobileNetV2(weights='imagenet')
    
    # Process image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array)
    
    # Decode semua predictions
    all_decoded = decode_predictions(predictions, top=1000)[0]
    
    # Filter berdasarkan threshold
    high_confidence = [
        (class_id, class_name, score) 
        for class_id, class_name, score in all_decoded 
        if score >= threshold
    ]
    
    print(f"High confidence predictions (≥{threshold}):")
    for class_id, class_name, score in high_confidence:
        print(f"  {class_name}: {score:.4f}")
    
    if not high_confidence:
        print(f"No predictions above {threshold} threshold")
        print("Top 3 predictions:")
        for class_id, class_name, score in all_decoded[:3]:
            print(f"  {class_name}: {score:.4f}")
    
    return high_confidence

# classify_with_confidence_threshold('uncertain_image.jpg', threshold=0.05)
```

### Error Handling dan Best Practices

```python
def robust_decode_predictions(predictions, top=5):
    """
    Robust version dengan error handling
    """
    try:
        # Validasi input
        if predictions.ndim != 2:
            raise ValueError(f"Expected 2D array, got {predictions.ndim}D")
        
        if predictions.shape[1] != 1000:
            raise ValueError(f"Expected 1000 classes, got {predictions.shape[1]}")
        
        # Normalize predictions jika belum
        if not np.allclose(predictions.sum(axis=1), 1.0, rtol=1e-3):
            print("Warning: Predictions don't sum to 1, normalizing...")
            predictions = predictions / predictions.sum(axis=1, keepdims=True)
        
        # Decode
        decoded = decode_predictions(predictions, top=top)
        
        return decoded
        
    except Exception as e:
        print(f"Error in decode_predictions: {e}")
        return None

# Test dengan berbagai input
test_cases = [
    np.random.rand(1, 1000),  # Valid
    np.random.rand(2, 1000),  # Batch
    np.random.rand(1, 500),   # Wrong shape
]

for i, test_pred in enumerate(test_cases):
    print(f"\nTest case {i+1}:")
    result = robust_decode_predictions(test_pred)
    if result:
        print(f"Success: {len(result)} predictions decoded")
    else:
        print("Failed to decode")
```

### Performance Optimization

```python
def optimized_decode_predictions():
    """
    Tips untuk optimasi performance decode_predictions
    """
    import time
    import numpy as np
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
    
    # Prepare test data
    batch_size = 100
    predictions = np.random.rand(batch_size, 1000)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)
    
    # Method 1: Decode all at once (efficient)
    start_time = time.time()
    decoded_batch = decode_predictions(predictions, top=5)
    batch_time = time.time() - start_time
    
    # Method 2: Decode one by one (inefficient)
    start_time = time.time()
    decoded_individual = []
    for pred in predictions:
        decoded_individual.append(decode_predictions(np.expand_dims(pred, 0), top=5)[0])
    individual_time = time.time() - start_time
    
    print(f"Batch processing: {batch_time:.4f} seconds")
    print(f"Individual processing: {individual_time:.4f} seconds")
    print(f"Speedup: {individual_time/batch_time:.2f}x")
    
    # Memory usage comparison
    import sys
    batch_memory = sys.getsizeof(decoded_batch)
    individual_memory = sys.getsizeof(decoded_individual)
    
    print(f"Batch memory: {batch_memory} bytes")
    print(f"Individual memory: {individual_memory} bytes")

# optimized_decode_predictions()
```

---

## Ringkasan Bab 3

Dalam bab ini, kita telah mempelajari:

1. **Sejarah MobileNet**: Evolusi dari V1 hingga V3 dan alasan memilih V2
2. **Arsitektur MobileNetV2**: Struktur detail dan implementasi dalam Keras
3. **Inverted Residual Blocks**: Inovasi utama yang membuat MobileNetV2 efisien
4. **Pre-trained Weights**: Bagaimana ImageNet training menghasilkan weights berkualitas
5. **decode_predictions()**: Cara kerja internal dan penggunaan praktis

**Key Takeaways:**
- MobileNetV2 memberikan balance optimal antara akurasi dan efisiensi
- Inverted Residual Blocks dengan Linear Bottleneck adalah inovasi kunci
- Pre-trained weights dari ImageNet sangat powerful untuk transfer learning
- `decode_predictions()` mengkonversi output numerik menjadi label yang meaningful

Pada bab berikutnya, kita akan mulai implementasi praktis dengan membangun proyek klasifikasi gambar menggunakan semua konsep yang telah dipelajari.
