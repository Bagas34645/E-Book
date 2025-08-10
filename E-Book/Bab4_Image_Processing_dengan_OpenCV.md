# BAB 4: Image Processing dengan OpenCV

## Pendahuluan

Dalam pengembangan sistem klasifikasi gambar menggunakan deep learning, tahap preprocessing atau pemrosesan awal gambar merupakan komponen yang sangat krusial untuk mencapai akurasi yang optimal. OpenCV (Open Source Computer Vision Library) adalah library yang paling populer dan powerful untuk melakukan berbagai operasi computer vision dan image processing. Pada bab ini, kita akan mempelajari secara mendalam bagaimana menggunakan OpenCV untuk mempersiapkan data gambar agar sesuai dengan requirements input model MobileNetV2.

Preprocessing yang tepat tidak hanya memastikan bahwa gambar memiliki format dan dimensi yang sesuai dengan arsitektur model, tetapi juga dapat meningkatkan performa model secara signifikan. Setiap langkah preprocessing yang akan kita pelajari memiliki alasan teknis yang solid dan telah terbukti efektif dalam berbagai aplikasi computer vision di industri.

## 4.1 Pengenalan OpenCV untuk Computer Vision

### 4.1.1 Konsep Dasar OpenCV

OpenCV adalah library open-source yang dikembangkan oleh Intel pada tahun 1999 dan telah menjadi standar industri untuk aplikasi computer vision. Library ini menyediakan lebih dari 2500 algoritma yang dioptimasi untuk berbagai task computer vision, mulai dari operasi dasar seperti loading dan manipulasi gambar hingga algoritma kompleks seperti object detection dan face recognition.

Keunggulan utama OpenCV terletak pada performanya yang sangat optimal karena core library-nya ditulis dalam C/C++ dengan optimasi untuk berbagai arsitektur processor. Wrapper Python yang tersedia memungkinkan developer untuk menggunakan performa tinggi C/C++ dengan sintaks Python yang mudah dipahami. Hal ini membuat OpenCV ideal untuk aplikasi real-time yang membutuhkan processing speed tinggi.

### 4.1.2 Instalasi dan Setup OpenCV

OpenCV dapat diinstal dengan mudah menggunakan package manager pip. Untuk aplikasi machine learning, kita biasanya menggunakan opencv-python yang merupakan distribusi pre-compiled OpenCV untuk Python:

```bash
pip install opencv-python
pip install opencv-contrib-python  # untuk fitur tambahan
```

Package `opencv-contrib-python` berisi modul tambahan yang tidak termasuk dalam distribusi standar, seperti SIFT, SURF, dan berbagai algoritma experimental yang sering berguna untuk research dan development aplikasi computer vision yang lebih advanced.

### 4.1.3 Struktur Data Image dalam OpenCV

Dalam OpenCV, gambar direpresentasikan sebagai NumPy array dengan struktur multi-dimensional. Untuk gambar berwarna (RGB), struktur array adalah (height, width, channels) dimana channels biasanya 3 untuk Red, Green, Blue. Namun perlu diperhatikan bahwa OpenCV menggunakan format BGR (Blue, Green, Red) sebagai default, bukan RGB seperti library lainnya.

Pemahaman tentang struktur data ini sangat penting karena mempengaruhi bagaimana kita melakukan manipulasi pixel-level dan memastikan konsistensi format data ketika berinteraksi dengan library lain seperti TensorFlow atau PIL. Misalnya, jika kita load gambar dengan OpenCV dan kemudian ingin menampilkannya dengan matplotlib, kita perlu melakukan konversi BGR ke RGB terlebih dahulu.

### 4.1.4 Color Space dan Channel Management

OpenCV mendukung berbagai color space seperti BGR, RGB, HSV, LAB, dan GRAY. Pemilihan color space yang tepat sangat penting tergantung pada aplikasi spesifik. Untuk deep learning dengan pre-trained model seperti MobileNetV2, kita biasanya menggunakan RGB format karena model tersebut di-train menggunakan dataset dengan format RGB.

Konversi antar color space dapat dilakukan dengan fungsi `cv2.cvtColor()`. Operasi ini sangat efisien dan dioptimasi untuk berbagai kombinasi konversi. Pemahaman tentang karakteristik masing-masing color space membantu dalam memilih preprocessing yang optimal untuk specific use case.

## 4.2 Loading dan Manipulasi Gambar

### 4.2.1 Loading Gambar dari File

Proses loading gambar adalah langkah pertama dalam pipeline image processing. OpenCV menyediakan fungsi `cv2.imread()` yang sangat robust dan mendukung berbagai format gambar populer seperti JPEG, PNG, BMP, TIFF, dan WebP. Fungsi ini memiliki beberapa flag yang memungkinkan kita mengontrol bagaimana gambar di-load:

```python
import cv2
import numpy as np

# Load gambar dalam format BGR (default)
image = cv2.imread('path/to/image.jpg')

# Load gambar dalam grayscale
image_gray = cv2.imread('path/to/image.jpg', cv2.IMREAD_GRAYSCALE)

# Load gambar dengan alpha channel (RGBA)
image_rgba = cv2.imread('path/to/image.png', cv2.IMREAD_UNCHANGED)
```

Parameter kedua dalam `cv2.imread()` sangat penting untuk dikontrol. `cv2.IMREAD_COLOR` (default) akan load gambar dalam 3 channel BGR, `cv2.IMREAD_GRAYSCALE` akan convert gambar menjadi single channel grayscale, dan `cv2.IMREAD_UNCHANGED` akan preserve format original termasuk alpha channel jika ada.

### 4.2.2 Error Handling dan Validation

Dalam aplikasi production, sangat penting untuk melakukan proper error handling saat loading gambar. File gambar bisa corrupt, path bisa salah, atau format tidak didukung. OpenCV akan return `None` jika gagal load gambar, sehingga kita perlu selalu melakukan validation:

```python
def load_image_safely(image_path):
    """
    Load gambar dengan error handling yang proper
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Gagal load gambar dari: {image_path}")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
```

Implementasi error handling yang robust memastikan aplikasi kita tidak crash ketika menghadapi edge case dan memberikan feedback yang jelas untuk debugging.

### 4.2.3 Basic Image Properties dan Metadata

Setelah berhasil loading gambar, langkah berikutnya adalah memahami properties dasar gambar tersebut. Informasi seperti dimensi, data type, dan jumlah channel sangat penting untuk preprocessing selanjutnya:

```python
def analyze_image_properties(image):
    """
    Analyze basic properties dari gambar
    """
    if image is None:
        return None
    
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    dtype = image.dtype
    
    print(f"Dimensi: {width}x{height}")
    print(f"Channels: {channels}")
    print(f"Data type: {dtype}")
    print(f"Min pixel value: {image.min()}")
    print(f"Max pixel value: {image.max()}")
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'dtype': dtype
    }
```

Informasi ini membantu kita memahami karakteristik data input dan merencanakan preprocessing pipeline yang sesuai.

### 4.2.4 Basic Image Manipulations

OpenCV menyediakan berbagai fungsi untuk manipulasi dasar gambar. Operasi-operasi ini sering digunakan sebagai preprocessing step atau untuk data augmentation:

```python
# Flip gambar horizontal
flipped_horizontal = cv2.flip(image, 1)

# Flip gambar vertical
flipped_vertical = cv2.flip(image, 0)

# Rotate gambar
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, rotation_matrix, (w, h))

# Crop gambar
x, y, w, h = 100, 100, 200, 200  # x, y, width, height
cropped = image[y:y+h, x:x+w]
```

Setiap operasi manipulasi memiliki computational cost yang berbeda. Operasi seperti flip relatif murah karena hanya mengubah urutan pixel, sedangkan rotation membutuhkan interpolasi yang lebih computational expensive.

## 4.3 Resizing Image ke 224x224 pixels

### 4.3.1 Pentingnya Standardisasi Dimensi Input

Dalam arsitektur deep learning, terutama Convolutional Neural Networks (CNN), dimensi input harus konsisten untuk semua gambar yang akan diproses. MobileNetV2, seperti kebanyakan model CNN modern, di-design untuk menerima input dengan dimensi fixed yaitu 224x224 pixels. Standardisasi ini bukan hanya requirement teknis, tetapi juga mempengaruhi efisiensi computational dan memory usage.

Dimensi 224x224 dipilih berdasarkan trade-off antara detail informasi yang dapat dipertahankan dan computational efficiency. Dimensi ini cukup besar untuk mempertahankan fitur-fitur penting dalam gambar, namun tidak terlalu besar sehingga menyebabkan computational overhead yang signifikan. Penelitian menunjukkan bahwa dimensi ini optimal untuk berbagai task computer vision.

### 4.3.2 Algoritma Interpolasi untuk Resizing

Ketika melakukan resize gambar, kita perlu memilih algoritma interpolasi yang tepat. OpenCV menyediakan beberapa metode interpolasi, masing-masing dengan karakteristik dan use case yang berbeda:

```python
def resize_image_with_interpolation(image, target_size=(224, 224)):
    """
    Resize gambar dengan berbagai metode interpolasi
    """
    methods = {
        'nearest': cv2.INTER_NEAREST,      # Fastest, lowest quality
        'linear': cv2.INTER_LINEAR,        # Good balance, default
        'cubic': cv2.INTER_CUBIC,          # Better quality, slower
        'area': cv2.INTER_AREA,            # Best for downsampling
        'lanczos': cv2.INTER_LANCZOS4      # Highest quality, slowest
    }
    
    results = {}
    for name, method in methods.items():
        resized = cv2.resize(image, target_size, interpolation=method)
        results[name] = resized
    
    return results
```

**INTER_NEAREST** menggunakan nearest neighbor interpolation, dimana setiap pixel baru mengambil nilai dari pixel terdekat. Metode ini paling cepat tetapi menghasilkan kualitas paling rendah dengan efek aliasing yang terlihat.

**INTER_LINEAR** (bilinear interpolation) menggunakan interpolasi linear dari 4 pixel terdekat. Ini adalah default method yang memberikan balance baik antara speed dan quality, cocok untuk kebanyakan aplikasi.

**INTER_CUBIC** (bicubic interpolation) menggunakan interpolasi dari 16 pixel terdekat, menghasilkan kualitas lebih baik terutama untuk edges dan detail halus, tetapi membutuhkan computational cost lebih tinggi.

**INTER_AREA** optimal untuk downsampling (mengurangi ukuran gambar) karena menggunakan area interpolation yang mengurangi moiré effects dan aliasing.

**INTER_LANCZOS4** menggunakan Lanczos kernel dengan window size 4, memberikan kualitas terbaik terutama untuk preserving sharp edges, tetapi paling lambat.

### 4.3.3 Aspect Ratio Preservation vs Direct Resize

Salah satu challenge utama dalam resizing adalah menangani perbedaan aspect ratio antara gambar original dan target size. Ada beberapa strategi yang dapat diimplementasikan:

#### Direct Resize (Non-Uniform Scaling)
```python
def direct_resize(image, target_size=(224, 224)):
    """
    Resize gambar langsung ke target size tanpa mempertahankan aspect ratio
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
```

Metode ini paling simple dan fast, tetapi dapat menyebabkan distorsi pada gambar jika aspect ratio original berbeda signifikan dengan target.

#### Aspect Ratio Preserving dengan Padding
```python
def resize_with_padding(image, target_size=(224, 224), pad_color=(0, 0, 0)):
    """
    Resize gambar dengan mempertahankan aspect ratio dan menambahkan padding
    """
    old_height, old_width = image.shape[:2]
    target_width, target_height = target_size
    
    # Hitung scaling factor
    scale = min(target_width / old_width, target_height / old_height)
    
    # Hitung dimensi baru
    new_width = int(old_width * scale)
    new_height = int(old_height * scale)
    
    # Resize dengan aspect ratio preserved
    resized = cv2.resize(image, (new_width, new_height), 
                        interpolation=cv2.INTER_LINEAR)
    
    # Buat canvas dengan target size
    canvas = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)
    
    # Hitung posisi untuk center placement
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized image pada canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas
```

Metode ini mempertahankan aspect ratio original tetapi menambahkan padding (biasanya hitam) untuk mencapai target size. Ini mencegah distorsi tetapi mengurangi area yang berisi informasi actual.

#### Aspect Ratio Preserving dengan Center Crop
```python
def resize_with_center_crop(image, target_size=(224, 224)):
    """
    Resize gambar dengan mempertahankan aspect ratio dan melakukan center crop
    """
    old_height, old_width = image.shape[:2]
    target_width, target_height = target_size
    
    # Hitung scaling factor untuk memastikan minimal dimension match target
    scale = max(target_width / old_width, target_height / old_height)
    
    # Hitung dimensi setelah scaling
    new_width = int(old_width * scale)
    new_height = int(old_height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), 
                        interpolation=cv2.INTER_LINEAR)
    
    # Hitung crop coordinates untuk center crop
    y_start = (new_height - target_height) // 2
    x_start = (new_width - target_width) // 2
    
    # Crop ke target size
    cropped = resized[y_start:y_start+target_height, 
                     x_start:x_start+target_width]
    
    return cropped
```

Metode ini mempertahankan aspect ratio dan menggunakan seluruh area target, tetapi menghilangkan bagian gambar di edges.

### 4.3.4 Performance Optimization untuk Batch Processing

Dalam aplikasi real-world, kita sering perlu memproses banyak gambar sekaligus. Optimasi performance menjadi sangat penting:

```python
def batch_resize_optimized(image_paths, target_size=(224, 224), 
                          interpolation=cv2.INTER_LINEAR):
    """
    Optimized batch resizing untuk multiple images
    """
    resized_images = []
    
    for path in image_paths:
        try:
            # Load image
            image = cv2.imread(path)
            if image is None:
                continue
                
            # Resize
            resized = cv2.resize(image, target_size, interpolation=interpolation)
            resized_images.append(resized)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    return np.array(resized_images) if resized_images else None
```

## 4.4 Preprocessing untuk MobileNetV2

### 4.4.1 Understanding MobileNetV2 Input Requirements

MobileNetV2 adalah arsitektur CNN yang didesign khusus untuk mobile dan embedded devices dengan fokus pada efficiency tanpa mengorbankan akurasi secara signifikan. Model ini memiliki specific requirements untuk input data yang harus dipenuhi agar dapat berfungsi optimal.

Input yang diharapkan oleh MobileNetV2 adalah tensor dengan shape (batch_size, 224, 224, 3) dimana:
- **batch_size**: jumlah gambar yang diproses secara bersamaan
- **224, 224**: dimensi spatial (height, width) dalam pixels
- **3**: jumlah channels (Red, Green, Blue)

Model ini di-training menggunakan ImageNet dataset dengan specific preprocessing pipeline, sehingga untuk mencapai performa optimal, kita perlu meniru preprocessing yang sama saat inference.

### 4.4.2 Color Space Conversion untuk MobileNetV2

Salah satu langkah krusial dalam preprocessing adalah memastikan color space yang konsisten. OpenCV default menggunakan BGR format, sedangkan MobileNetV2 (dan kebanyakan pre-trained model) mengharapkan RGB format:

```python
def convert_bgr_to_rgb(image):
    """
    Convert BGR (OpenCV default) ke RGB format untuk MobileNetV2
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR ke RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    else:
        raise ValueError("Input harus berupa gambar BGR 3-channel")

def preprocess_color_space(image):
    """
    Comprehensive color space preprocessing
    """
    # Validasi input
    if image is None:
        raise ValueError("Input image tidak boleh None")
    
    # Handle grayscale conversion ke RGB jika diperlukan
    if len(image.shape) == 2:
        # Convert grayscale ke RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA ke RGB (remove alpha channel)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Convert BGR ke RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image format: {image.shape}")
    
    return image
```

Konversi color space ini penting karena channel order yang berbeda dapat menyebabkan model menginterpretasi warna secara incorrect, yang pada akhirnya mempengaruhi akurasi klasifikasi.

### 4.4.3 Data Type Conversion dan Range Adjustment

MobileNetV2 mengharapkan input dalam format floating-point dengan range nilai tertentu. OpenCV default load gambar sebagai uint8 dengan range 0-255, sehingga kita perlu melakukan konversi:

```python
def convert_to_float32(image):
    """
    Convert image dari uint8 ke float32 dengan range 0-1
    """
    if image.dtype != np.uint8:
        print(f"Warning: Input bukan uint8, current dtype: {image.dtype}")
    
    # Convert ke float32 dan normalize ke range 0-1
    float_image = image.astype(np.float32) / 255.0
    
    # Validasi range
    assert float_image.min() >= 0.0 and float_image.max() <= 1.0, \
           f"Range tidak valid: {float_image.min()} - {float_image.max()}"
    
    return float_image
```

### 4.4.4 Comprehensive Preprocessing Pipeline

Menggabungkan semua langkah preprocessing dalam satu pipeline yang efisien dan reusable:

```python
class MobileNetV2Preprocessor:
    """
    Comprehensive preprocessing pipeline untuk MobileNetV2
    """
    
    def __init__(self, target_size=(224, 224), interpolation=cv2.INTER_LINEAR):
        self.target_size = target_size
        self.interpolation = interpolation
    
    def __call__(self, image_path_or_array):
        """
        Main preprocessing function
        """
        # Handle input (path atau array)
        if isinstance(image_path_or_array, str):
            image = self._load_image(image_path_or_array)
        else:
            image = image_path_or_array.copy()
        
        # Apply preprocessing steps
        image = self._resize_image(image)
        image = self._convert_color_space(image)
        image = self._convert_data_type(image)
        
        return image
    
    def _load_image(self, image_path):
        """Load image dengan error handling"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Gagal load gambar: {image_path}")
        return image
    
    def _resize_image(self, image):
        """Resize dengan method yang dipilih"""
        return cv2.resize(image, self.target_size, interpolation=self.interpolation)
    
    def _convert_color_space(self, image):
        """Convert color space ke RGB"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported image format: {image.shape}")
    
    def _convert_data_type(self, image):
        """Convert ke float32 dengan range 0-1"""
        return image.astype(np.float32) / 255.0
    
    def preprocess_batch(self, image_paths):
        """Preprocess multiple images sekaligus"""
        processed_images = []
        
        for path in image_paths:
            try:
                processed = self(path)
                processed_images.append(processed)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        return np.array(processed_images) if processed_images else None
```

## 4.5 Normalisasi dan Dimensi Expansion

### 4.5.1 Konsep Normalisasi dalam Deep Learning

Normalisasi adalah proses standardisasi distribusi nilai input agar memiliki karakteristik statistik yang konsisten. Dalam konteks deep learning, normalizasi yang tepat sangat krusial untuk:

1. **Stabilitas Training**: Gradient yang stable dan tidak exploding/vanishing
2. **Convergence Speed**: Model converge lebih cepat ke optimal solution
3. **Generalization**: Model perform better pada data yang belum pernah dilihat
4. **Numerical Stability**: Mencegah overflow/underflow dalam floating-point operations

MobileNetV2, seperti kebanyakan model yang di-training pada ImageNet, menggunakan specific normalization scheme yang harus kita tiru saat inference.

### 4.5.2 ImageNet Normalization Standards

Model pre-trained pada ImageNet dataset menggunakan normalization dengan mean dan standard deviation yang dihitung dari seluruh training dataset:

```python
# ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_imagenet(image):
    """
    Normalize image menggunakan ImageNet statistics
    
    Formula: (image - mean) / std
    """
    if image.dtype != np.float32:
        raise ValueError("Input harus float32 dengan range 0-1")
    
    # Ensure image dalam range 0-1
    if image.max() > 1.0 or image.min() < 0.0:
        raise ValueError(f"Image range tidak valid: {image.min()} - {image.max()}")
    
    # Apply normalization per channel
    normalized = (image - IMAGENET_MEAN) / IMAGENET_STD
    
    return normalized
```

Nilai mean dan std ini dihitung dari jutaan gambar dalam ImageNet dataset dan merepresentasikan karakteristik statistik umum dari natural images. Channel-wise normalization ini membantu model memahami distribusi warna yang konsisten.

### 4.5.3 Understanding Statistical Properties

Untuk memahami lebih dalam mengapa normalization penting, mari kita analyze statistical properties sebelum dan sesudah normalization:

```python
def analyze_normalization_effect(image_before, image_after):
    """
    Analyze statistical properties sebelum dan sesudah normalization
    """
    print("=== BEFORE NORMALIZATION ===")
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        channel_data = image_before[:, :, i]
        print(f"{channel} Channel:")
        print(f"  Mean: {channel_data.mean():.4f}")
        print(f"  Std:  {channel_data.std():.4f}")
        print(f"  Min:  {channel_data.min():.4f}")
        print(f"  Max:  {channel_data.max():.4f}")
    
    print("\n=== AFTER NORMALIZATION ===")
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        channel_data = image_after[:, :, i]
        print(f"{channel} Channel:")
        print(f"  Mean: {channel_data.mean():.4f}")
        print(f"  Std:  {channel_data.std():.4f}")
        print(f"  Min:  {channel_data.min():.4f}")
        print(f"  Max:  {channel_data.max():.4f}")

def visualize_normalization_distribution(images_before, images_after):
    """
    Visualize distribusi pixel values sebelum dan sesudah normalization
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    channels = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    
    # Plot histogram sebelum normalization
    for i, (channel, color) in enumerate(zip(channels, colors)):
        channel_data = images_before[:, :, :, i].flatten()
        axes[0, i].hist(channel_data, bins=50, alpha=0.7, color=color)
        axes[0, i].set_title(f'{channel} - Before Normalization')
        axes[0, i].set_xlabel('Pixel Value')
        axes[0, i].set_ylabel('Frequency')
    
    # Plot histogram sesudah normalization
    for i, (channel, color) in enumerate(zip(channels, colors)):
        channel_data = images_after[:, :, :, i].flatten()
        axes[1, i].hist(channel_data, bins=50, alpha=0.7, color=color)
        axes[1, i].set_title(f'{channel} - After Normalization')
        axes[1, i].set_xlabel('Normalized Value')
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
```

### 4.5.4 Dimensi Expansion untuk Batch Processing

Deep learning models biasanya didesign untuk memproses multiple gambar sekaligus (batch processing) untuk efisiensi computational. Single image dengan shape (224, 224, 3) perlu di-expand menjadi (1, 224, 224, 3) untuk batch dimension:

```python
def expand_dimensions(image):
    """
    Expand dimensions untuk menambahkan batch dimension
    
    Input:  (height, width, channels)
    Output: (batch_size=1, height, width, channels)
    """
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D array, got {len(image.shape)}D")
    
    expanded = np.expand_dims(image, axis=0)
    print(f"Shape transformation: {image.shape} -> {expanded.shape}")
    
    return expanded

def batch_expand_dimensions(images):
    """
    Expand dimensions untuk array of images
    
    Input:  (num_images, height, width, channels)
    Output: (num_images, height, width, channels) - sudah dalam batch format
    """
    if len(images.shape) == 3:
        # Single image, tambahkan batch dimension
        return np.expand_dims(images, axis=0)
    elif len(images.shape) == 4:
        # Already in batch format
        return images
    else:
        raise ValueError(f"Unsupported array shape: {images.shape}")
```

### 4.5.5 Memory-Efficient Batch Processing

Untuk aplikasi yang memproses banyak gambar, memory management menjadi penting:

```python
def create_batch_generator(image_paths, batch_size=32, preprocessor=None):
    """
    Generator untuk batch processing yang memory-efficient
    """
    if preprocessor is None:
        preprocessor = MobileNetV2Preprocessor()
    
    num_images = len(image_paths)
    
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch_paths = image_paths[start_idx:end_idx]
        
        batch_images = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                # Preprocess single image
                processed = preprocessor(path)
                
                # Normalize
                normalized = normalize_imagenet(processed)
                
                batch_images.append(normalized)
                valid_paths.append(path)
                
            except Exception as e:
                print(f"Skipping {path}: {e}")
                continue
        
        if batch_images:
            # Convert ke numpy array dan expand batch dimension
            batch_array = np.array(batch_images)
            yield batch_array, valid_paths

# Contoh penggunaan
def process_images_in_batches(image_paths, model, batch_size=32):
    """
    Process images menggunakan batch generator
    """
    preprocessor = MobileNetV2Preprocessor()
    results = []
    
    batch_gen = create_batch_generator(image_paths, batch_size, preprocessor)
    
    for batch_images, batch_paths in batch_gen:
        # Predict menggunakan model
        predictions = model.predict(batch_images)
        
        # Store results
        for path, pred in zip(batch_paths, predictions):
            results.append({
                'path': path,
                'prediction': pred
            })
    
    return results
```

### 4.5.6 Complete Preprocessing Pipeline

Mari kita buat complete preprocessing pipeline yang menggabungkan semua concepts:

```python
class CompleteMobileNetV2Preprocessor:
    """
    Complete preprocessing pipeline untuk MobileNetV2 dengan semua optimizations
    """
    
    def __init__(self, 
                 target_size=(224, 224),
                 interpolation=cv2.INTER_LINEAR,
                 preserve_aspect_ratio=False,
                 apply_normalization=True):
        
        self.target_size = target_size
        self.interpolation = interpolation
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.apply_normalization = apply_normalization
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess_single(self, image_input):
        """
        Preprocess single image dengan complete pipeline
        """
        # Load image jika input adalah path
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Cannot load image: {image_input}")
        else:
            image = image_input.copy()
        
        # Resize
        if self.preserve_aspect_ratio:
            image = self._resize_with_aspect_ratio(image)
        else:
            image = cv2.resize(image, self.target_size, 
                             interpolation=self.interpolation)
        
        # Color space conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to 0-1
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization jika enabled
        if self.apply_normalization:
            image = self._normalize_imagenet(image)
        
        return image
    
    def preprocess_batch(self, image_inputs, add_batch_dim=True):
        """
        Preprocess multiple images
        """
        processed_images = []
        
        for img_input in image_inputs:
            try:
                processed = self.preprocess_single(img_input)
                processed_images.append(processed)
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        if not processed_images:
            raise ValueError("No images successfully processed")
        
        # Convert to numpy array
        batch_array = np.array(processed_images)
        
        # Add batch dimension jika single image
        if add_batch_dim and len(batch_array.shape) == 3:
            batch_array = np.expand_dims(batch_array, axis=0)
        
        return batch_array
    
    def _resize_with_aspect_ratio(self, image):
        """
        Resize dengan mempertahankan aspect ratio menggunakan center crop
        """
        old_height, old_width = image.shape[:2]
        target_width, target_height = self.target_size
        
        # Calculate scaling factor
        scale = max(target_width / old_width, target_height / old_height)
        
        # New dimensions
        new_width = int(old_width * scale)
        new_height = int(old_height * scale)
        
        # Resize
        resized = cv2.resize(image, (new_width, new_height), 
                           interpolation=self.interpolation)
        
        # Center crop
        y_start = (new_height - target_height) // 2
        x_start = (new_width - target_width) // 2
        
        cropped = resized[y_start:y_start+target_height, 
                         x_start:x_start+target_width]
        
        return cropped
    
    def _normalize_imagenet(self, image):
        """
        Apply ImageNet normalization
        """
        return (image - self.mean) / self.std
    
    def get_preprocessing_info(self):
        """
        Return informasi tentang preprocessing configuration
        """
        return {
            'target_size': self.target_size,
            'interpolation_method': self._get_interpolation_name(),
            'preserve_aspect_ratio': self.preserve_aspect_ratio,
            'apply_normalization': self.apply_normalization,
            'normalization_mean': self.mean.tolist(),
            'normalization_std': self.std.tolist()
        }
    
    def _get_interpolation_name(self):
        """
        Get human-readable interpolation method name
        """
        interpolation_names = {
            cv2.INTER_NEAREST: 'NEAREST',
            cv2.INTER_LINEAR: 'LINEAR',
            cv2.INTER_CUBIC: 'CUBIC',
            cv2.INTER_AREA: 'AREA',
            cv2.INTER_LANCZOS4: 'LANCZOS4'
        }
        return interpolation_names.get(self.interpolation, 'UNKNOWN')

# Contoh penggunaan
def demo_complete_preprocessing():
    """
    Demo penggunaan complete preprocessing pipeline
    """
    # Inisialisasi preprocessor
    preprocessor = CompleteMobileNetV2Preprocessor(
        target_size=(224, 224),
        interpolation=cv2.INTER_LINEAR,
        preserve_aspect_ratio=True,
        apply_normalization=True
    )
    
    # Print configuration
    print("Preprocessing Configuration:")
    import json
    print(json.dumps(preprocessor.get_preprocessing_info(), indent=2))
    
    # Example preprocessing single image
    try:
        # Ganti dengan path gambar yang valid
        preprocessed = preprocessor.preprocess_single('path/to/your/image.jpg')
        print(f"\nPreprocessed image shape: {preprocessed.shape}")
        print(f"Data type: {preprocessed.dtype}")
        print(f"Value range: {preprocessed.min():.4f} to {preprocessed.max():.4f}")
        
        # Add batch dimension untuk model inference
        batch_input = np.expand_dims(preprocessed, axis=0)
        print(f"Batch input shape: {batch_input.shape}")
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
```

## Kesimpulan

Dalam bab ini, kita telah mempelajari secara komprehensif tentang image processing menggunakan OpenCV untuk mempersiapkan data input yang optimal bagi model MobileNetV2. Setiap langkah preprocessing yang telah dibahas memiliki peran penting dalam memastikan model dapat bekerja dengan performa maksimal.

Pemahaman mendalam tentang konsep-konsep ini tidak hanya penting untuk implementasi yang benar, tetapi juga untuk debugging ketika model tidak memberikan hasil yang diharapkan. Preprocessing yang tidak tepat sering menjadi bottleneck dalam pipeline machine learning, dan dengan menguasai principles yang telah dibahas, kita dapat mengidentifikasi dan mengatasi masalah-masalah tersebut.

Langkah selanjutnya adalah mengintegrasikan preprocessing pipeline ini dengan model MobileNetV2 untuk melakukan klasifikasi gambar yang akan kita pelajari di bab berikutnya. Pipeline yang telah kita buat bersifat modular dan dapat dengan mudah diadaptasi untuk berbagai use case dan requirement spesifik.
