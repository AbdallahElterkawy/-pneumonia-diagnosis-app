from tensorflow.keras.preprocessing.image import ImageDataGenerator

# مسارات البيانات
train_path = "C:/Users/DARK/Desktop/pneumonia_project/chest_xray/train"
val_path = "C:/Users/DARK/Desktop/pneumonia_project/chest_xray/val"

# تجهيز الصور: إعادة تحجيم وتطبيع
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

# تحميل الصور من المجلدات
train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)