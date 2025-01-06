import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

# Load Fashion MNIST Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to include channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Dimensions
input_shape = x_train.shape[1:]  # (28, 28, 1)

# Build the Autoencoder Model
def build_autoencoder(input_shape):
    # Encoder
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = tf.keras.layers.Cropping2D(((2, 2), (2, 2)))(x)

    # Autoencoder model
    autoencoder = Model(inputs, decoded)

    # Encoder model for feature extraction
    encoder = Model(inputs, encoded)

    return autoencoder, encoder

# Compile the model
autoencoder, encoder = build_autoencoder(input_shape)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

# Model Checkpoint callback
# Model Checkpoint callback
checkpoint = ModelCheckpoint(
    filepath="autoencoder_best_model.keras",  # Changed to .keras extension
    monitor="val_loss",
    save_best_only=True,
    mode="min"
)


# Train the Autoencoder
history = autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test),
    callbacks=[checkpoint]  # Save the best model during training
)

# Save the final model
autoencoder.save("autoencoder_final_model.h5")

# Extract Features (Embeddings)
train_embeddings = encoder.predict(x_train)
test_embeddings = encoder.predict(x_test)

# Flatten embeddings for similarity search
train_embeddings_flat = train_embeddings.reshape(len(train_embeddings), -1)
test_embeddings_flat = test_embeddings.reshape(len(test_embeddings), -1)

# Similarity Search Function
def find_similar_images(query_image, num_results=5):
    # Get query embedding
    query_embedding = encoder.predict(query_image.reshape(1, 28, 28, 1)).reshape(1, -1)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, train_embeddings_flat)
    most_similar_indices = np.argsort(similarities[0])[::-1][:num_results]
    
    # Retrieve and display similar images
    similar_images = x_train[most_similar_indices]
    
    plt.figure(figsize=(10, 2))
    for i, img in enumerate(similar_images):
        plt.subplot(1, num_results, i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

    return most_similar_indices

# Evaluate Similarity Search with Precision, Recall, and Retrieval Accuracy
def evaluate_similarity_search():
    precision_list = []
    recall_list = []
    total_retrievals = 0
    correct_retrievals = 0

    for i in range(len(x_test)):  # Loop over test samples
        query_image = x_test[i]
        true_label = y_test[i]
        
        most_similar_indices = find_similar_images(query_image)
        similar_labels = y_train[most_similar_indices]
        
        # Calculate Precision and Recall
        precision = precision_score([true_label] * len(similar_labels), similar_labels, average='micro')
        recall = recall_score([true_label] * len(similar_labels), similar_labels, average='micro')
        
        precision_list.append(precision)
        recall_list.append(recall)
        
        # Calculate Retrieval Accuracy (check if true label is among the retrieved)
        if true_label in similar_labels:
            correct_retrievals += 1
        total_retrievals += 1

    # Calculate mean precision, recall, and retrieval accuracy
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    retrieval_accuracy = correct_retrievals / total_retrievals

    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print(f"Retrieval Accuracy: {retrieval_accuracy:.4f}")

# Test Similarity Search
random_test_image = x_test[0]
plt.imshow(random_test_image.squeeze(), cmap='gray')
plt.title("Query Image")
plt.axis('off')
plt.show()

# Evaluate metrics
evaluate_similarity_search()
