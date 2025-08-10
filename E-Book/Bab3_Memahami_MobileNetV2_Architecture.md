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
    Fungsi ini memuat mapping antara index dan label dari repository resmi ImageNet.
    Sistem klasifikasi ImageNet menggunakan hierarki WordNet yang mengorganisir
    konsep-konsep dalam struktur tree semantik yang kompleks.
    """
    try:
        # TensorFlow menyimpan mapping ini dalam format JSON yang dapat diakses secara publik
        import json
        import urllib.request
        
        url = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
        with urllib.request.urlopen(url) as response:
            class_index = json.loads(response.read().decode())
        
        return class_index
    except Exception as e:
        # Fallback ke mapping lokal jika terjadi error koneksi
        print(f"Warning: Could not load remote class index: {e}")
        return imagenet_class_index

class_index = load_imagenet_class_index()
print(f"Total classes loaded: {len(class_index)}")
print(f"Sample classes: {dict(list(class_index.items())[:5])}")
```

Sistem class indexing ini merupakan implementasi sophisticated dari WordNet hierarchy yang memungkinkan model untuk tidak hanya mengenali objek, tetapi juga memahami relasi semantik antar kategori. Setiap class identifier mengikuti konvensi penamaan yang konsisten dimana prefix 'n' menunjukkan noun (kata benda), diikuti dengan unique identifier numerik yang merujuk pada posisi spesifik dalam WordNet tree structure.

#### 2. **Implementasi Komprehensif decode_predictions()**

Untuk memahami mekanisme internal dari fungsi decode_predictions(), kita perlu menganalisis setiap komponen yang terlibat dalam proses transformasi dari raw numerical output menjadi human-readable labels dengan confidence scores yang terkalibasi.

```python
def custom_decode_predictions(preds, top=5, class_list_path=None):
    """
    Implementasi custom decode_predictions untuk memahami algoritma internal
    yang mengkonversi probability distribution menjadi ranked predictions.
    
    Fungsi ini mendemonstrasikan proses:
    1. Normalisasi probability distribution
    2. Sorting berdasarkan confidence scores
    3. Mapping index ke semantic labels
    4. Formatting output sesuai standar ImageNet
    
    Args:
        preds: Numpy array dengan shape (batch_size, 1000) berisi probability scores
        top: Integer yang menentukan jumlah top predictions yang dikembalikan
        class_list_path: Optional path ke file mapping class index
    
    Returns:
        List of tuples dalam format (wordnet_id, human_readable_name, confidence_score)
    """
    import numpy as np
    import json
    
    # Phase 1: Load semantic mapping dictionary
    if class_list_path:
        with open(class_list_path, 'r', encoding='utf-8') as f:
            class_index = json.load(f)
    else:
        class_index = load_imagenet_class_index()
    
    # Phase 2: Process each prediction in the batch
    results = []
    
    for pred in preds:
        # Ensure probability distribution sums to 1.0 for mathematical validity
        if not np.isclose(pred.sum(), 1.0, rtol=1e-5):
            print("Warning: Renormalizing prediction probabilities")
            pred = pred / pred.sum()
        
        # Phase 3: Implement efficient top-k selection algorithm
        # Using argpartition for O(n) complexity instead of O(n log n) sorting
        if top < len(pred):
            # Partial sorting - only sort the top k elements
            top_indices = np.argpartition(pred, -top)[-top:]
            top_indices = top_indices[np.argsort(pred[top_indices])[::-1]]
        else:
            # Full sorting if requesting all predictions
            top_indices = pred.argsort()[::-1][:top]
        
        # Phase 4: Construct semantic output with metadata
        result = []
        for i, idx in enumerate(top_indices):
            class_id = class_index[str(idx)][0]  # WordNet synset identifier
            class_name = class_index[str(idx)][1]  # Human-readable label
            confidence_score = float(pred[idx])  # Ensure JSON serializable
            
            result.append((class_id, class_name, confidence_score))
        
        results.append(result)
    
    return results

# Demonstration with synthetic probability distribution
def demonstrate_decode_functionality():
    """
    Demonstrasi komprehensif dari decode_predictions functionality
    menggunakan controlled synthetic data untuk validasi algoritma.
    """
    # Create realistic probability distribution
    synthetic_predictions = np.zeros((1, 1000))
    
    # Simulate high confidence predictions untuk beberapa classes
    target_classes = [281, 285, 282, 287, 283]  # Cat-related classes
    confidences = [0.45, 0.25, 0.15, 0.10, 0.05]
    
    for cls, conf in zip(target_classes, confidences):
        synthetic_predictions[0, cls] = conf
    
    # Distribute remaining probability mass
    remaining_prob = 1.0 - sum(confidences)
    noise_indices = np.random.choice(
        [i for i in range(1000) if i not in target_classes], 
        size=50, 
        replace=False
    )
    noise_values = np.random.dirichlet([1] * 50) * remaining_prob
    synthetic_predictions[0, noise_indices] = noise_values
    
    # Test custom implementation
    custom_result = custom_decode_predictions(synthetic_predictions, top=5)
    
    print("Custom decode_predictions output:")
    print("-" * 50)
    for i, (class_id, class_name, score) in enumerate(custom_result[0]):
        print(f"{i+1:2d}. {class_name:<20} {score:.6f} ({score*100:.2f}%)")
    
    return custom_result

# Execute demonstration
demonstration_results = demonstrate_decode_functionality()
```

Implementasi di atas mengilustrasikan complexity dari probability distribution processing dalam deep learning inference pipeline. Algoritma menggunakan efficient sorting techniques untuk mengoptimalkan computational overhead, sambil mempertahankan mathematical rigor dalam probability normalization dan semantic mapping.

### WordNet Semantic Hierarchy System

WordNet hierarchy yang digunakan dalam ImageNet classification merupakan foundation dari organized knowledge representation yang memungkinkan model untuk memahami tidak hanya visual features, tetapi juga semantic relationships antar categories.

#### Struktur Hierarkis WordNet dalam Konteks Computer Vision

```python
def analyze_wordnet_hierarchy():
    """
    Analisis mendalam tentang WordNet hierarchy system yang mendasari
    ImageNet classification scheme. Sistem ini mengorganisir konsep-konsep
    dalam directed acyclic graph (DAG) yang memungkinkan inheritance
    properties dan semantic similarity computation.
    """
    
    # Contoh comprehensive hierarki untuk different semantic domains
    wordnet_examples = {
        'n02123045': {
            'name': 'tabby_cat',
            'definition': 'a cat with a grey or tawny coat mottled with black',
            'synset_id': 'tabby.n.02',
            'hypernymy_chain': [
                'entity.n.01',           # Root concept
                'physical_entity.n.01',   # Has physical existence
                'object.n.01',           # Discrete physical entity
                'whole.n.02',            # Has organizational structure
                'living_thing.n.01',     # Exhibits life processes
                'organism.n.01',         # Individual biological entity
                'animal.n.01',           # Mobile multicellular organism
                'chordate.n.01',         # Has spinal cord
                'vertebrate.n.01',       # Has backbone
                'mammal.n.01',           # Warm-blooded vertebrate
                'placental.n.01',        # Eutherian mammal
                'carnivore.n.01',        # Flesh-eating mammal
                'feline.n.01',           # Cat family
                'cat.n.01',              # Domestic cat species
                'house_cat.n.01',        # Domesticated cat
                'tabby.n.02'             # Specific coat pattern
            ],
            'semantic_depth': 15,
            'domain': 'biological_taxonomy'
        },
        
        'n04579432': {
            'name': 'van',
            'definition': 'a truck with an enclosed cargo space',
            'synset_id': 'van.n.05',
            'hypernymy_chain': [
                'entity.n.01',           # Root concept
                'physical_entity.n.01',   # Physical manifestation
                'object.n.01',           # Discrete entity
                'whole.n.02',            # Complete structure
                'artifact.n.01',         # Human-made object
                'instrumentality.n.03',   # Serves as means
                'conveyance.n.03',       # Transportation device
                'vehicle.n.01',          # Self-propelled conveyance
                'wheeled_vehicle.n.01',   # Ground transportation
                'motor_vehicle.n.01',     # Engine-powered vehicle
                'car.n.01',              # Passenger vehicle
                'van.n.05'               # Cargo-oriented variant
            ],
            'semantic_depth': 11,
            'domain': 'artificial_objects'
        },
        
        'n01484850': {
            'name': 'great_white_shark',
            'definition': 'large aggressive shark widespread in warm seas',
            'synset_id': 'great_white_shark.n.01',
            'hypernymy_chain': [
                'entity.n.01',           # Universal root
                'physical_entity.n.01',   # Physical existence
                'object.n.01',           # Bounded entity
                'whole.n.02',            # Integrated system
                'living_thing.n.01',     # Biological entity
                'organism.n.01',         # Individual life form
                'animal.n.01',           # Mobile organism
                'chordate.n.01',         # Spinal cord bearer
                'vertebrate.n.01',       # Backbone structure
                'aquatic_vertebrate.n.01', # Water-dwelling vertebrate
                'fish.n.01',             # Gill-breathing vertebrate
                'cartilaginous_fish.n.01', # Cartilage skeleton
                'shark.n.01',            # Predatory fish
                'mackerel_shark.n.01',   # Fast-swimming shark
                'great_white_shark.n.01' # Apex predator species
            ],
            'semantic_depth': 14,
            'domain': 'marine_biology'
        }
    }
    
    print("WordNet Hierarchy Analysis for ImageNet Classes")
    print("=" * 60)
    
    for wordnet_id, concept_info in wordnet_examples.items():
        print(f"\nWordNet ID: {wordnet_id}")
        print(f"Concept Name: {concept_info['name']}")
        print(f"Definition: {concept_info['definition']}")
        print(f"Semantic Depth: {concept_info['semantic_depth']} levels")
        print(f"Domain: {concept_info['domain']}")
        print(f"Hypernymy Chain:")
        
        for i, hypernym in enumerate(concept_info['hypernymy_chain']):
            indent = "  " * i
            print(f"{indent}├─ {hypernym}")
    
    # Calculate semantic similarity metrics
    def calculate_semantic_distance(chain1, chain2):
        """
        Menghitung semantic distance berdasarkan Lowest Common Ancestor (LCA)
        dalam WordNet hierarchy tree structure.
        """
        # Find common ancestors
        common_ancestors = set(chain1) & set(chain2)
        if not common_ancestors:
            return float('inf')  # No common ancestors
        
        # Find deepest common ancestor (most specific)
        lca_depth = max(chain1.index(ancestor) for ancestor in common_ancestors)
        
        # Calculate path distance through LCA
        cat_depth = len(chain1) - 1
        van_depth = len(chain2) - 1
        
        distance = (cat_depth - lca_depth) + (van_depth - lca_depth)
        return distance
    
    # Demonstrate semantic similarity computation
    cat_chain = wordnet_examples['n02123045']['hypernymy_chain']
    van_chain = wordnet_examples['n04579432']['hypernymy_chain']
    shark_chain = wordnet_examples['n01484850']['hypernymy_chain']
    
    print(f"\nSemantic Distance Analysis:")
    print(f"Cat ↔ Van distance: {calculate_semantic_distance(cat_chain, van_chain)}")
    print(f"Cat ↔ Shark distance: {calculate_semantic_distance(cat_chain, shark_chain)}")
    print(f"Van ↔ Shark distance: {calculate_semantic_distance(van_chain, shark_chain)}")
    
    return wordnet_examples

# Execute comprehensive analysis
hierarchy_analysis = analyze_wordnet_hierarchy()
```

Sistem hierarki ini fundamental dalam memahami bagaimana neural networks dapat mengembangkan representations yang tidak hanya mengenali visual patterns, tetapi juga memahami conceptual relationships yang memungkinkan generalization across related categories.

Sistem hierarki ini fundamental dalam memahami bagaimana neural networks dapat mengembangkan representations yang tidak hanya mengenali visual patterns, tetapi juga memahami conceptual relationships yang memungkinkan generalization across related categories.

### Advanced Practical Applications dalam Real-World Scenarios

#### 1. **Confidence-Aware Classification System dengan Uncertainty Quantification**

Dalam aplikasi production-grade, penting untuk mengimplementasikan sistem yang dapat mengukur uncertainty dan mengambil keputusan berdasarkan confidence levels yang terkalibri. Implementasi berikut mendemonstrasikan advanced techniques untuk uncertainty quantification:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from scipy import stats

class ConfidenceAwareClassifier:
    """
    Advanced classifier yang mengintegrasikan uncertainty quantification
    dengan calibrated confidence estimation untuk robust decision making
    dalam production environments.
    """
    
    def __init__(self, model_name='MobileNetV2', confidence_threshold=0.7):
        """
        Initialize classifier dengan advanced confidence calibration
        
        Args:
            model_name: String identifier untuk pre-trained model
            confidence_threshold: Minimum confidence untuk positive classification
        """
        self.model = MobileNetV2(weights='imagenet', include_top=True)
        self.confidence_threshold = confidence_threshold
        self.calibration_temperature = 1.0  # For temperature scaling
        self.prediction_history = []
        
    def temperature_scaling(self, logits, temperature):
        """
        Implements temperature scaling untuk calibrating confidence estimates.
        
        Temperature scaling adalah post-processing technique yang proven effective
        untuk improving calibration tanpa mengubah model accuracy.
        
        Args:
            logits: Raw model outputs sebelum softmax activation
            temperature: Scaling parameter (T > 1 reduces confidence, T < 1 increases)
            
        Returns:
            Calibrated probability distribution
        """
        return tf.nn.softmax(logits / temperature, axis=-1)
    
    def monte_carlo_dropout_prediction(self, image, n_samples=100):
        """
        Implements Monte Carlo Dropout untuk estimating epistemic uncertainty.
        
        Technique ini enables uncertainty estimation pada deterministic models
        dengan repeatedly applying dropout during inference untuk generating
        diverse predictions yang representative dari model uncertainty.
        
        Args:
            image: Input image tensor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Mean prediction, variance, epistemic uncertainty estimates
        """
        predictions = []
        
        # Enable dropout during inference untuk Monte Carlo sampling
        for layer in self.model.layers:
            if hasattr(layer, 'training'):
                layer.training = True  # Force dropout to be active
        
        for _ in range(n_samples):
            pred = self.model(image, training=True)  # Keep dropout active
            predictions.append(pred.numpy())
        
        # Reset to normal inference mode
        for layer in self.model.layers:
            if hasattr(layer, 'training'):
                layer.training = False
        
        predictions = np.array(predictions)
        
        # Calculate statistics across samples
        mean_prediction = np.mean(predictions, axis=0)
        prediction_variance = np.var(predictions, axis=0)
        epistemic_uncertainty = np.mean(prediction_variance, axis=1)
        
        return mean_prediction, prediction_variance, epistemic_uncertainty
    
    def ensemble_prediction(self, image, models_list=None):
        """
        Implements ensemble prediction untuk improving robustness dan accuracy.
        
        Ensemble methods combine predictions dari multiple models untuk reducing
        individual model biases dan improving generalization performance.
        """
        if models_list is None:
            # Create simple ensemble dengan different random seeds
            models_list = [self.model]
        
        ensemble_predictions = []
        
        for model in models_list:
            pred = model.predict(image, verbose=0)
            ensemble_predictions.append(pred)
        
        # Calculate ensemble statistics
        ensemble_predictions = np.array(ensemble_predictions)
        mean_prediction = np.mean(ensemble_predictions, axis=0)
        prediction_std = np.std(ensemble_predictions, axis=0)
        
        return mean_prediction, prediction_std
    
    def calculate_prediction_entropy(self, predictions):
        """
        Calculate Shannon entropy sebagai measure dari prediction uncertainty.
        
        Higher entropy indicates higher uncertainty dalam prediction,
        yang useful untuk identifying out-of-distribution samples atau
        cases yang memerlukan human review.
        """
        # Avoid log(0) dengan small epsilon
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        entropy = -np.sum(predictions * np.log(predictions), axis=-1)
        return entropy
    
    def comprehensive_prediction(self, image_path, analysis_methods=['standard', 'mc_dropout', 'entropy']):
        """
        Performs comprehensive prediction analysis dengan multiple uncertainty
        quantification methods untuk robust classification decision.
        
        Args:
            image_path: Path ke input image
            analysis_methods: List of analysis methods to apply
            
        Returns:
            Dictionary berisi comprehensive analysis results
        """
        # Load dan preprocess image
        from tensorflow.keras.preprocessing import image
        
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        results = {
            'image_path': image_path,
            'analysis_timestamp': tf.timestamp().numpy(),
            'methods_used': analysis_methods
        }
        
        # Standard prediction
        if 'standard' in analysis_methods:
            standard_pred = self.model.predict(img_array, verbose=0)
            decoded_pred = decode_predictions(standard_pred, top=5)[0]
            
            results['standard_prediction'] = {
                'raw_predictions': standard_pred,
                'decoded_predictions': decoded_pred,
                'max_confidence': float(np.max(standard_pred)),
                'prediction_confident': float(np.max(standard_pred)) > self.confidence_threshold
            }
        
        # Monte Carlo Dropout analysis
        if 'mc_dropout' in analysis_methods:
            mc_mean, mc_variance, epistemic_uncertainty = self.monte_carlo_dropout_prediction(img_array)
            mc_decoded = decode_predictions(mc_mean, top=5)[0]
            
            results['monte_carlo_analysis'] = {
                'mean_prediction': mc_mean,
                'prediction_variance': mc_variance,
                'epistemic_uncertainty': float(np.mean(epistemic_uncertainty)),
                'decoded_predictions': mc_decoded,
                'uncertainty_level': 'high' if np.mean(epistemic_uncertainty) > 0.1 else 'low'
            }
        
        # Entropy analysis
        if 'entropy' in analysis_methods:
            entropy = self.calculate_prediction_entropy(standard_pred if 'standard' in analysis_methods else self.model.predict(img_array, verbose=0))
            
            results['entropy_analysis'] = {
                'prediction_entropy': float(entropy[0]),
                'uncertainty_interpretation': self._interpret_entropy(entropy[0]),
                'recommended_action': self._recommend_action_based_on_entropy(entropy[0])
            }
        
        # Overall decision synthesis
        results['final_decision'] = self._synthesize_decision(results)
        
        return results
    
    def _interpret_entropy(self, entropy_value):
        """Interpret entropy value dalam context of classification confidence"""
        if entropy_value < 0.5:
            return "Very confident prediction"
        elif entropy_value < 1.0:
            return "Moderately confident prediction"
        elif entropy_value < 2.0:
            return "Low confidence prediction"
        else:
            return "Very uncertain prediction"
    
    def _recommend_action_based_on_entropy(self, entropy_value):
        """Recommend action berdasarkan entropy analysis"""
        if entropy_value < 0.5:
            return "Accept prediction automatically"
        elif entropy_value < 1.0:
            return "Accept with monitoring"
        elif entropy_value < 2.0:
            return "Flag for manual review"
        else:
            return "Reject - seek alternative classification method"
    
    def _synthesize_decision(self, analysis_results):
        """
        Synthesize final decision berdasarkan multiple analysis methods
        """
        decision_factors = []
        
        if 'standard_prediction' in analysis_results:
            if analysis_results['standard_prediction']['prediction_confident']:
                decision_factors.append('high_confidence_standard')
            else:
                decision_factors.append('low_confidence_standard')
        
        if 'monte_carlo_analysis' in analysis_results:
            if analysis_results['monte_carlo_analysis']['uncertainty_level'] == 'low':
                decision_factors.append('low_epistemic_uncertainty')
            else:
                decision_factors.append('high_epistemic_uncertainty')
        
        if 'entropy_analysis' in analysis_results:
            entropy = analysis_results['entropy_analysis']['prediction_entropy']
            if entropy < 1.0:
                decision_factors.append('low_entropy')
            else:
                decision_factors.append('high_entropy')
        
        # Decision logic
        positive_factors = ['high_confidence_standard', 'low_epistemic_uncertainty', 'low_entropy']
        positive_count = sum(1 for factor in decision_factors if factor in positive_factors)
        
        if positive_count >= 2:
            return {
                'classification_decision': 'ACCEPT',
                'confidence_level': 'HIGH',
                'reasoning': f"Multiple positive indicators: {positive_count}/3"
            }
        elif positive_count == 1:
            return {
                'classification_decision': 'REVIEW',
                'confidence_level': 'MEDIUM',
                'reasoning': "Mixed indicators - manual review recommended"
            }
        else:
            return {
                'classification_decision': 'REJECT',
                'confidence_level': 'LOW',
                'reasoning': "Multiple uncertainty indicators detected"
            }

# Demonstration of advanced classification system
def demonstrate_advanced_classification():
    """
    Comprehensive demonstration dari advanced classification capabilities
    dengan real-world application scenarios.
    """
    print("Advanced Classification System Demonstration")
    print("=" * 55)
    
    # Initialize advanced classifier
    classifier = ConfidenceAwareClassifier(confidence_threshold=0.75)
    
    # Simulate dengan synthetic image data untuk demonstration
    synthetic_image = np.random.rand(1, 224, 224, 3) * 255
    
    # Perform standard prediction untuk comparison
    standard_pred = classifier.model.predict(synthetic_image, verbose=0)
    standard_decoded = decode_predictions(standard_pred, top=3)[0]
    
    print("\nStandard Prediction Results:")
    print("-" * 30)
    for i, (class_id, class_name, score) in enumerate(standard_decoded):
        print(f"{i+1}. {class_name:<20} {score:.4f} ({score*100:.1f}%)")
    
    # Calculate entropy untuk uncertainty assessment
    entropy = classifier.calculate_prediction_entropy(standard_pred)
    print(f"\nPrediction Entropy: {entropy[0]:.4f}")
    print(f"Uncertainty Level: {classifier._interpret_entropy(entropy[0])}")
    print(f"Recommended Action: {classifier._recommend_action_based_on_entropy(entropy[0])}")
    
    # Monte Carlo Dropout analysis
    print("\nMonte Carlo Dropout Analysis:")
    print("-" * 35)
    mc_mean, mc_variance, epistemic_uncertainty = classifier.monte_carlo_dropout_prediction(synthetic_image, n_samples=50)
    
    print(f"Epistemic Uncertainty: {np.mean(epistemic_uncertainty):.6f}")
    print(f"Prediction Variance (mean): {np.mean(mc_variance):.6f}")
    
    mc_decoded = decode_predictions(mc_mean, top=3)[0]
    print("\nMC Dropout Top Predictions:")
    for i, (class_id, class_name, score) in enumerate(mc_decoded):
        variance_for_class = mc_variance[0][np.argmax(mc_mean[0])]
        std_dev = np.sqrt(variance_for_class)
        print(f"{i+1}. {class_name:<20} {score:.4f} ± {std_dev:.4f}")
    
    return classifier

# Execute advanced demonstration
advanced_classifier = demonstrate_advanced_classification()
```

#### 2. **Production-Grade Deployment Pipeline dengan MLOps Integration**

Implementasi production-ready system memerlukan integration dengan MLOps practices, monitoring, logging, dan automated model management:

```python
import logging
import json
import time
from datetime import datetime
from pathlib import Path
import joblib
import tensorflow as tf

class ProductionClassificationPipeline:
    """
    Enterprise-grade classification pipeline dengan comprehensive
    monitoring, logging, model versioning, dan automated fallback mechanisms.
    """
    
    def __init__(self, config_path='classification_config.json'):
        """
        Initialize production pipeline dengan configuration management
        """
        self.config = self._load_configuration(config_path)
        self.logger = self._setup_logging()
        self.model = self._load_model_with_versioning()
        self.metrics_collector = self._initialize_metrics_collection()
        self.performance_monitor = self._setup_performance_monitoring()
        
    def _load_configuration(self, config_path):
        """Load configuration dengan environment-specific settings"""
        default_config = {
            'model_version': '1.0.0',
            'confidence_threshold': 0.75,
            'max_inference_time_ms': 1000,
            'batch_size': 32,
            'enable_monitoring': True,
            'enable_caching': True,
            'fallback_model_path': None,
            'logging_level': 'INFO',
            'metrics_endpoint': None
        }
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self):
        """Setup comprehensive logging untuk production monitoring"""
        logger = logging.getLogger('ProductionClassifier')
        logger.setLevel(getattr(logging, self.config['logging_level']))
        
        # Console handler untuk development
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler untuk production logs
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f'classification_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_model_with_versioning(self):
        """Load model dengan version management dan fallback support"""
        try:
            # Primary model loading
            model = MobileNetV2(weights='imagenet', include_top=True)
            self.logger.info(f"Successfully loaded MobileNetV2 version {self.config['model_version']}")
            
            # Model validation
            self._validate_model(model)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load primary model: {e}")
            
            # Fallback model loading
            if self.config['fallback_model_path']:
                try:
                    fallback_model = tf.keras.models.load_model(self.config['fallback_model_path'])
                    self.logger.warning("Loaded fallback model")
                    return fallback_model
                except Exception as fallback_error:
                    self.logger.critical(f"Fallback model loading failed: {fallback_error}")
            
            raise RuntimeError("All model loading attempts failed")
    
    def _validate_model(self, model):
        """Validate model integrity dan performance"""
        # Test dengan synthetic input
        test_input = tf.random.normal((1, 224, 224, 3))
        
        start_time = time.time()
        test_output = model(test_input, training=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Validation checks
        assert test_output.shape == (1, 1000), f"Unexpected output shape: {test_output.shape}"
        assert inference_time < self.config['max_inference_time_ms'], f"Inference too slow: {inference_time}ms"
        assert tf.reduce_sum(test_output).numpy() > 0, "Model producing zero outputs"
        
        self.logger.info(f"Model validation passed. Inference time: {inference_time:.2f}ms")
    
    def _initialize_metrics_collection(self):
        """Initialize metrics collection untuk monitoring"""
        return {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_confidence': 0.0,
            'average_inference_time_ms': 0.0,
            'last_update_timestamp': time.time(),
            'daily_prediction_count': 0,
            'high_confidence_predictions': 0,
            'low_confidence_predictions': 0
        }
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring dengan automated alerts"""
        return {
            'inference_times': [],
            'confidence_scores': [],
            'error_counts': {'last_hour': 0, 'last_day': 0},
            'alert_thresholds': {
                'max_inference_time_ms': 2000,
                'min_average_confidence': 0.5,
                'max_error_rate': 0.05
            }
        }
    
    def predict_with_monitoring(self, image_input, request_id=None):
        """
        Core prediction method dengan comprehensive monitoring dan error handling
        
        Args:
            image_input: Input image (path, array, atau tensor)
            request_id: Unique identifier untuk request tracking
            
        Returns:
            Dictionary berisi prediction results dan metadata
        """
        request_id = request_id or f"req_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Input preprocessing dan validation
            processed_input = self._preprocess_input(image_input)
            
            # Core prediction dengan error handling
            prediction_start = time.time()
            raw_predictions = self.model.predict(processed_input, verbose=0)
            inference_time = (time.time() - prediction_start) * 1000
            
            # Post-processing dan decoding
            decoded_predictions = decode_predictions(raw_predictions, top=5)[0]
            
            # Extract key metrics
            max_confidence = float(np.max(raw_predictions))
            prediction_entropy = self._calculate_entropy(raw_predictions[0])
            
            # Update monitoring metrics
            self._update_metrics(inference_time, max_confidence, success=True)
            
            # Construct comprehensive response
            response = {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'predictions': [
                    {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'rank': rank + 1
                    }
                    for rank, (class_id, class_name, confidence) in enumerate(decoded_predictions)
                ],
                'metadata': {
                    'inference_time_ms': round(inference_time, 2),
                    'max_confidence': round(max_confidence, 4),
                    'prediction_entropy': round(prediction_entropy, 4),
                    'model_version': self.config['model_version'],
                    'high_confidence': max_confidence > self.config['confidence_threshold']
                },
                'quality_indicators': {
                    'confidence_level': self._categorize_confidence(max_confidence),
                    'uncertainty_level': self._categorize_uncertainty(prediction_entropy),
                    'recommendation': self._generate_recommendation(max_confidence, prediction_entropy)
                }
            }
            
            # Logging
            self.logger.info(f"Successful prediction for {request_id}: {decoded_predictions[0][1]} ({max_confidence:.3f})")
            
            return response
            
        except Exception as e:
            # Error handling dengan detailed logging
            total_time = (time.time() - start_time) * 1000
            self._update_metrics(total_time, 0.0, success=False)
            
            error_response = {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error_message': str(e),
                'error_type': type(e).__name__,
                'total_time_ms': round(total_time, 2)
            }
            
            self.logger.error(f"Prediction failed for {request_id}: {e}")
            
            return error_response
    
    def _preprocess_input(self, image_input):
        """Robust image preprocessing dengan validation"""
        if isinstance(image_input, str):
            # Load dari file path
            from tensorflow.keras.preprocessing import image
            img = image.load_img(image_input, target_size=(224, 224))
            img_array = image.img_to_array(img)
        elif isinstance(image_input, np.ndarray):
            # Handle numpy array
            if image_input.shape != (224, 224, 3):
                # Resize jika perlu
                img_array = tf.image.resize(image_input, [224, 224]).numpy()
            else:
                img_array = image_input
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")
        
        # Normalize dan add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def _calculate_entropy(self, probabilities):
        """Calculate Shannon entropy untuk uncertainty quantification"""
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        return -np.sum(probabilities * np.log(probabilities))
    
    def _categorize_confidence(self, confidence):
        """Categorize confidence level untuk user interpretation"""
        if confidence > 0.9:
            return "VERY_HIGH"
        elif confidence > 0.75:
            return "HIGH"
        elif confidence > 0.5:
            return "MEDIUM"
        elif confidence > 0.25:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _categorize_uncertainty(self, entropy):
        """Categorize uncertainty level berdasarkan entropy"""
        if entropy < 0.5:
            return "VERY_LOW"
        elif entropy < 1.0:
            return "LOW"
        elif entropy < 2.0:
            return "MEDIUM"
        elif entropy < 3.0:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def _generate_recommendation(self, confidence, entropy):
        """Generate actionable recommendation berdasarkan metrics"""
        if confidence > 0.8 and entropy < 1.0:
            return "Accept prediction with high confidence"
        elif confidence > 0.6 and entropy < 1.5:
            return "Accept prediction with moderate confidence"
        elif confidence > 0.4:
            return "Review prediction manually"
        else:
            return "Reject prediction - seek alternative method"
    
    def _update_metrics(self, inference_time, confidence, success=True):
        """Update monitoring metrics untuk performance tracking"""
        self.metrics_collector['total_predictions'] += 1
        
        if success:
            self.metrics_collector['successful_predictions'] += 1
            
            # Update running averages
            n = self.metrics_collector['successful_predictions']
            current_avg_confidence = self.metrics_collector['average_confidence']
            self.metrics_collector['average_confidence'] = (
                (current_avg_confidence * (n - 1) + confidence) / n
            )
            
            current_avg_time = self.metrics_collector['average_inference_time_ms']
            self.metrics_collector['average_inference_time_ms'] = (
                (current_avg_time * (n - 1) + inference_time) / n
            )
            
            # Track confidence distribution
            if confidence > self.config['confidence_threshold']:
                self.metrics_collector['high_confidence_predictions'] += 1
            else:
                self.metrics_collector['low_confidence_predictions'] += 1
                
        else:
            self.metrics_collector['failed_predictions'] += 1
        
        # Update performance monitoring
        self.performance_monitor['inference_times'].append(inference_time)
        if success:
            self.performance_monitor['confidence_scores'].append(confidence)
        
        # Keep only recent data (last 1000 predictions)
        if len(self.performance_monitor['inference_times']) > 1000:
            self.performance_monitor['inference_times'] = self.performance_monitor['inference_times'][-1000:]
        if len(self.performance_monitor['confidence_scores']) > 1000:
            self.performance_monitor['confidence_scores'] = self.performance_monitor['confidence_scores'][-1000:]
        
        # Check untuk performance alerts
        self._check_performance_alerts()
    
    def _check_performance_alerts(self):
        """Check for performance issues dan trigger alerts jika perlu"""
        recent_times = self.performance_monitor['inference_times'][-100:]  # Last 100 predictions
        recent_confidences = self.performance_monitor['confidence_scores'][-100:]
        
        if len(recent_times) > 10:  # Minimal data untuk meaningful analysis
            avg_recent_time = np.mean(recent_times)
            avg_recent_confidence = np.mean(recent_confidences) if recent_confidences else 0
            
            # Performance degradation alerts
            if avg_recent_time > self.performance_monitor['alert_thresholds']['max_inference_time_ms']:
                self.logger.warning(f"Performance alert: Average inference time {avg_recent_time:.2f}ms exceeds threshold")
            
            if avg_recent_confidence < self.performance_monitor['alert_thresholds']['min_average_confidence']:
                self.logger.warning(f"Confidence alert: Average confidence {avg_recent_confidence:.3f} below threshold")
            
            # Error rate monitoring
            error_rate = self.metrics_collector['failed_predictions'] / self.metrics_collector['total_predictions']
            if error_rate > self.performance_monitor['alert_thresholds']['max_error_rate']:
                self.logger.critical(f"Error rate alert: {error_rate:.3f} exceeds maximum threshold")
    
    def get_system_health_report(self):
        """Generate comprehensive system health report"""
        total_predictions = self.metrics_collector['total_predictions']
        
        if total_predictions == 0:
            return {"status": "No predictions yet", "metrics": self.metrics_collector}
        
        success_rate = self.metrics_collector['successful_predictions'] / total_predictions
        
        health_report = {
            'system_status': 'healthy' if success_rate > 0.95 else 'degraded' if success_rate > 0.8 else 'critical',
            'uptime_metrics': {
                'total_predictions': total_predictions,
                'success_rate': round(success_rate, 4),
                'average_inference_time_ms': round(self.metrics_collector['average_inference_time_ms'], 2),
                'average_confidence': round(self.metrics_collector['average_confidence'], 4)
            },
            'quality_metrics': {
                'high_confidence_rate': round(
                    self.metrics_collector['high_confidence_predictions'] / max(self.metrics_collector['successful_predictions'], 1), 4
                ),
                'recent_performance': {
                    'avg_inference_time_last_100': round(np.mean(self.performance_monitor['inference_times'][-100:]), 2) if self.performance_monitor['inference_times'] else 0,
                    'avg_confidence_last_100': round(np.mean(self.performance_monitor['confidence_scores'][-100:]), 4) if self.performance_monitor['confidence_scores'] else 0
                }
            },
            'configuration': {
                'model_version': self.config['model_version'],
                'confidence_threshold': self.config['confidence_threshold'],
                'max_inference_time_ms': self.config['max_inference_time_ms']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return health_report

# Demonstration of production pipeline
def demonstrate_production_pipeline():
    """
    Comprehensive demonstration dari production-ready classification pipeline
    dengan complete monitoring dan error handling capabilities.
    """
    print("Production Classification Pipeline Demonstration")
    print("=" * 60)
    
    # Initialize production pipeline
    pipeline = ProductionClassificationPipeline()
    
    # Generate synthetic test cases
    test_cases = [
        np.random.rand(224, 224, 3) * 255,  # Random image
        np.ones((224, 224, 3)) * 128,       # Gray image
        np.zeros((224, 224, 3)),            # Black image
    ]
    
    print("\nProcessing test cases:")
    print("-" * 25)
    
    for i, test_image in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        result = pipeline.predict_with_monitoring(test_image, request_id=f"test_{i+1}")
        
        if result['status'] == 'success':
            top_prediction = result['predictions'][0]
            metadata = result['metadata']
            quality = result['quality_indicators']
            
            print(f"  ✓ Prediction: {top_prediction['class_name']}")
            print(f"  ✓ Confidence: {top_prediction['confidence']:.4f} ({quality['confidence_level']})")
            print(f"  ✓ Inference Time: {metadata['inference_time_ms']:.2f}ms")
            print(f"  ✓ Recommendation: {quality['recommendation']}")
        else:
            print(f"  ✗ Error: {result['error_message']}")
    
    # Generate system health report
    print(f"\n{'='*40}")
    print("System Health Report")
    print(f"{'='*40}")
    
    health_report = pipeline.get_system_health_report()
    
    print(f"System Status: {health_report['system_status'].upper()}")
    print(f"Total Predictions: {health_report['uptime_metrics']['total_predictions']}")
    print(f"Success Rate: {health_report['uptime_metrics']['success_rate']*100:.1f}%")
    print(f"Average Inference Time: {health_report['uptime_metrics']['average_inference_time_ms']:.2f}ms")
    print(f"Average Confidence: {health_report['uptime_metrics']['average_confidence']:.4f}")
    print(f"High Confidence Rate: {health_report['quality_metrics']['high_confidence_rate']*100:.1f}%")
    
    return pipeline

# Execute production demonstration
production_pipeline = demonstrate_production_pipeline()
```

Implementasi ini mendemonstrasikan production-grade classification system yang ready untuk deployment dalam enterprise environments, complete dengan monitoring, logging, error handling, dan automated performance management.

---

## Kesimpulan Bab 3

Dalam bab ini, kita telah melakukan eksplorasi mendalam tentang MobileNetV2 architecture yang melampaui sekedar pemahaman surface-level. Pembahasan meliputi:

### Aspek Teknis yang Telah Dikuasai:

1. **Evolusi Architectural**: Pemahaman komprehensif tentang progression dari MobileNetV1 ke V2, termasuk innovations dalam depthwise separable convolutions dan inverted residual blocks.

2. **Mathematical Foundations**: Analisis detailed dari mathematical operations, computational complexity analysis, dan parameter efficiency yang menjadi foundation dari mobile-optimized architectures.

3. **ImageNet Pre-training System**: Understanding mendalam tentang training process, WordNet hierarchy, dan semantic relationships yang memungkinkan effective transfer learning.

4. **Production Implementation**: Practical implementation strategies untuk real-world applications, termasuk uncertainty quantification, monitoring systems, dan MLOps integration.

### Key Technical Insights:

- **Inverted Residual Design**: Architecture innovation yang memungkinkan efficient information flow sambil maintaining computational efficiency
- **Linear Bottlenecks**: Strategic use dari linear activations untuk preserving information dalam low-dimensional representations  
- **Depth Multiplier Strategy**: Flexible scaling mechanism untuk adapting model complexity berdasarkan deployment constraints
- **Transfer Learning Effectiveness**: Systematic approach untuk leveraging pre-trained representations dalam domain-specific applications

### Advanced Applications Covered:

- Confidence-aware classification dengan uncertainty quantification
- Monte Carlo dropout untuk epistemic uncertainty estimation
- Production-ready deployment pipelines dengan comprehensive monitoring
- MLOps integration dengan automated performance management

Dengan foundation yang solid dalam MobileNetV2 architecture, kita sekarang siap untuk melanjutkan ke implementasi praktis dalam bab berikutnya, dimana concepts ini akan diaplikasikan dalam building complete image classification applications menggunakan Streamlit dan modern deployment practices.

---

*Bab selanjutnya akan fokus pada practical implementation dari konsep-konsep advanced yang telah dipelajari dalam konteks end-to-end application development.*
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
