import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.metrics import Precision, Recall
from tqdm.notebook import tqdm


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * (
            (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        )

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name="dice_coefficient", **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice_coef = self.add_weight(name="dc", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)  # Cast to float32
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)  # Cast to float32
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        # Use backend.epsilon() to avoid division by zero
        self.dice_coef.assign(
            2.0
            * intersection
            / (
                tf.reduce_sum(y_true_f)
                + tf.reduce_sum(y_pred_f)
                + tf.keras.backend.epsilon()
            )
        )

    def result(self):
        return self.dice_coef

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.dice_coef.assign(0.0)


class MetricsHistory(Callback):
    def __init__(self):
        super(MetricsHistory, self).__init__()
        self.metrics = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "dice_coefficient": [],
            "iou": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1_score": [],
            "val_iou": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.metrics["loss"].append(logs.get("loss"))
        self.metrics["accuracy"].append(logs.get("accuracy"))
        self.metrics["precision"].append(logs.get("precision"))
        self.metrics["recall"].append(logs.get("recall"))
        self.metrics["f1_score"].append(logs.get("f1_score"))
        self.metrics["dice_coefficient"].append(logs.get("dice_coefficient"))
        self.metrics["iou"].append(logs.get("iou"))
        self.metrics["val_loss"].append(logs.get("val_loss"))
        self.metrics["val_accuracy"].append(logs.get("val_accuracy"))
        self.metrics["val_precision"].append(logs.get("val_precision"))
        self.metrics["val_recall"].append(logs.get("val_recall"))
        self.metrics["val_f1_score"].append(logs.get("val_f1_score"))
        self.metrics["val_iou"].append(logs.get("val_iou"))


class TQDMProgressBar(Callback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def on_train_begin(self, logs=None):
        total = self.params.get("steps", 0) * self.params.get("epochs", 0)
        self.progress_bar = tqdm(total=total, position=0, desc="Training", leave=True)

    def on_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)
        self.progress_bar.set_postfix(logs, refresh=True)

    def on_train_end(self, logs=None):
        self.progress_bar.close()
        self.progress_bar = None


class CustomUNet:
    def __init__(
        self,
        img_height=512,
        img_width=512,
        batch_size=32,
        start_filters=32,
        depth=4,
        conv_kernel_size=(3, 3),
        pool_size=(2, 2),
        activation="relu",
        output_activation="sigmoid",
        padding="same",
        initializer="he_normal",
        dropout=0.5,
        batch_norm=False,
        num_classes=1,
        upsample_size=(2, 2),
        optimizer="adam",
        loss="binary_crossentropy",
        save_model_path=None,
        patience=10,
        min_delta=0.001,
    ):
        self.metrics = None
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.start_filters = start_filters
        self.depth = depth
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.activation = activation
        self.output_activation = output_activation
        self.padding = padding
        self.initializer = initializer
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.num_classes = num_classes
        self.upsample_size = upsample_size
        self.optimizer = optimizer
        self.loss = loss
        self.save_model_path = save_model_path
        self.patience = patience
        self.min_delta = min_delta

        self.model = self.build_unet((self.img_height, self.img_width, 1))

    @staticmethod
    def custom_data_generator(
        image_folder, mask_folder, batch_size, img_height, img_width
    ):
        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith(".npz")]
        )
        # Create a list of mask files or None if mask_folder is not provided
        mask_files = (
            sorted([f for f in os.listdir(mask_folder) if f.endswith("_mask.npz")])
            if mask_folder is not None
            else [None] * len(image_files)
        )

        total_files = len(image_files)

        while True:  # Loop forever so the generator never terminates
            for i in tqdm(range(0, total_files, batch_size)):
                batch_image_files = image_files[i : i + batch_size]
                # Define batch_mask_files only if mask_folder is not None
                batch_mask_files = (
                    mask_files[i : i + batch_size]
                    if mask_folder is not None
                    else [None] * batch_size
                )

                batch_images = []
                batch_masks = []

                for img_file in batch_image_files:
                    # Load image
                    with np.load(os.path.join(image_folder, img_file)) as data:
                        image = data["data"]
                    image = np.resize(
                        image, (img_height, img_width, 1)
                    )  # Add channel dimension
                    image = image / 255.0  # Normalize
                    batch_images.append(image)

                # If mask_folder is provided, load masks
                if mask_folder is not None:
                    for mask_file in batch_mask_files:
                        # Load mask
                        with np.load(os.path.join(mask_folder, mask_file)) as data:
                            mask = data["data"]
                        mask = np.resize(
                            mask, (img_height, img_width, 1)
                        )  # Add channel dimension
                        batch_masks.append(mask)
                    yield np.array(batch_images), np.array(batch_masks)
                else:
                    yield np.array(batch_images)

    def build_unet(self, input_shape):
        inputs = Input(input_shape)
        x = inputs

        skip_connections = []

        # Encoder
        for i in range(self.depth):
            x = Conv2D(
                filters=self.start_filters * (2**i),
                kernel_size=self.conv_kernel_size,
                activation=self.activation,
                padding=self.padding,
                kernel_initializer=self.initializer,
            )(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Conv2D(
                filters=self.start_filters * (2**i) * 2,
                kernel_size=self.conv_kernel_size,
                activation=self.activation,
                padding=self.padding,
                kernel_initializer=self.initializer,
            )(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            skip_connections.append(x)
            x = MaxPooling2D(pool_size=self.pool_size)(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)

        # Middle
        x = Conv2D(
            filters=self.start_filters * (2**self.depth),
            kernel_size=self.conv_kernel_size,
            activation=self.activation,
            padding=self.padding,
            kernel_initializer=self.initializer,
        )(x)
        if self.batch_norm:
            x = BatchNormalization()(x)

        # Decoder
        for i in reversed(range(self.depth)):
            x = UpSampling2D(size=self.upsample_size)(x)
            x = concatenate([x, skip_connections[i]])
            x = Conv2D(
                filters=self.start_filters * (2**i),
                kernel_size=self.conv_kernel_size,
                activation=self.activation,
                padding=self.padding,
                kernel_initializer=self.initializer,
            )(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Conv2D(
                filters=self.start_filters * (2**i),
                kernel_size=self.conv_kernel_size,
                activation=self.activation,
                padding=self.padding,
                kernel_initializer=self.initializer,
            )(x)
            if self.batch_norm:
                x = BatchNormalization()(x)

        # Output
        outputs = Conv2D(
            filters=self.num_classes,
            kernel_size=(1, 1),
            activation=self.output_activation,
        )(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model

    def compile_and_train(
        self,
        train_image_folder,
        train_mask_folder,
        val_image_folder,
        val_mask_folder,
        epochs=50,
    ):
        train_generator = self.custom_data_generator(
            train_image_folder,
            train_mask_folder,
            self.batch_size,
            self.img_height,
            self.img_width,
        )
        val_generator = self.custom_data_generator(
            val_image_folder,
            val_mask_folder,
            self.batch_size,
            self.img_height,
            self.img_width,
        )

        steps_per_epoch = len(os.listdir(train_image_folder)) // self.batch_size
        validation_steps = len(os.listdir(val_image_folder)) // self.batch_size

        # Initialize tqdm callback
        tqdm_callback = TQDMProgressBar()

        f1_score_metric = F1Score(name="f1_score")

        dice_coef_metric = DiceCoefficient(name="dice_coefficient")

        metrics_history = MetricsHistory()

        # Add the Dice coefficient metric and MetricsHistory callback when compiling the model
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
                f1_score_metric,
                dice_coef_metric,
                MeanIoU(num_classes=2, name="iou"),
            ],
        )

        # Initialize EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor="val_loss",  # Change this to 'loss' if you don't have a validation set
            min_delta=self.min_delta,  # The minimum change to qualify as an improvement
            patience=self.patience,  # How many epochs to wait before stopping
            restore_best_weights=True,  # This will keep the best weights once stopped
            verbose=1,  # To print the log of when the training stops
        )

        # Train the model and save the history
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=0,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[tqdm_callback, early_stopping, metrics_history],
        )

        self.metrics = metrics_history.metrics

        return history

    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
        )

    def evaluate(
        self,
        test_image_folder,
        test_mask_folder,
        batch_size=32,
        img_height=512,
        img_width=512,
    ):
        test_generator = self.custom_data_generator(
            test_image_folder, test_mask_folder, batch_size, img_height, img_width
        )
        results = self.model.evaluate(
            test_generator, steps=len(os.listdir(test_image_folder)) // self.batch_size
        )
        print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

    def predict(self, image_folder, batch_size=32, img_height=512, img_width=512):
        print("Generating predictions...")
        print("image_folder: ", image_folder)
        # Create a generator for the images
        predict_generator = self.custom_data_generator(
            image_folder, None, batch_size, img_height, img_width
        )

        total_images = len(os.listdir(image_folder))
        steps = (
            total_images + batch_size - 1
        ) // batch_size  # This ensures you cover all images
        _predictions = self.model.predict(predict_generator, steps=steps)
        return _predictions

    def plot_training_history(self):
        plt.figure(figsize=(15, 5))

        # Plot Accuracy and Loss
        plt.subplot(1, 3, 1)
        if "accuracy" in self.metrics:
            plt.plot(self.metrics["accuracy"], label="Accuracy")
        plt.plot(self.metrics["loss"], label="Loss")
        plt.title("Accuracy and Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        # Plot Precision and Recall
        plt.subplot(1, 3, 2)
        if "precision" in self.metrics:
            plt.plot(self.metrics["precision"], label="Precision")
        if "recall" in self.metrics:
            plt.plot(self.metrics["recall"], label="Recall")
        plt.title("Precision and Recall over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        # Plot F1-Score and Dice Coefficient
        plt.subplot(1, 3, 3)
        if "f1_score" in self.metrics:
            plt.plot(self.metrics["f1_score"], label="F1 Score")
        if "dice_coefficient" in self.metrics:
            plt.plot(self.metrics["dice_coefficient"], label="Dice Coefficient")
        plt.title("F1 Score and Dice Coefficient over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        plt.show()

    def visualize_predictions(
        self, image_folder, mask_folder, num_images=5, threshold=0.5
    ):
        """
        Visualizes the original images, ground truth masks, predicted masks,
        and original images with bounding boxes for a given number of images.
        """
        print("Generating predictions...")
        print("image_folder: ", image_folder)
        print("mask_folder: ", mask_folder)

        generator = self.custom_data_generator(
            image_folder, mask_folder, self.batch_size, self.img_height, self.img_width
        )

        with tqdm(total=num_images, desc="Visualizing Predictions") as pbar:
            for i in range(num_images):
                # Get next batch of images and masks
                images, masks = next(generator)
                predictions = self.model.predict(images, verbose=0)

                # Convert predictions to binary masks
                binary_predictions = (predictions > threshold).astype(np.uint8)

                for j in range(len(images)):
                    plt.figure(figsize=(20, 5))

                    # Plot original image
                    plt.subplot(1, 4, 1)
                    plt.title("Original Image")
                    plt.imshow(np.squeeze(images[j]), cmap="gray")
                    plt.axis("off")

                    # Plot ground truth mask
                    plt.subplot(1, 4, 2)
                    plt.title("Ground Truth Mask")
                    plt.imshow(np.squeeze(masks[j]), cmap="gray")
                    plt.axis("off")

                    # Plot predicted mask
                    plt.subplot(1, 4, 3)
                    plt.title("Predicted Mask")
                    plt.imshow(np.squeeze(binary_predictions[j]), cmap="gray")
                    plt.axis("off")

                    # Plot original image with bounding boxes
                    plt.subplot(1, 4, 4)
                    plt.title("Original Image with Bounding Boxes")
                    boxed_image = self.draw_bounding_boxes(
                        np.squeeze(images[j]), np.squeeze(binary_predictions[j])
                    )
                    plt.imshow(boxed_image, cmap="gray")
                    plt.axis("off")

                    plt.show()

                pbar.update(1)

    @staticmethod
    def draw_bounding_boxes(image, mask, min_area=1000):
        """
        Draws bounding boxes on the image based on the provided mask.
        """
        _, thresh = cv2.threshold(mask, 40, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Make a copy of the image to draw on
        image_with_boxes = image.copy().astype(np.uint8)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Draw rectangle on the copy of the image
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image_with_boxes


if __name__ == "__main__":
    # Initialize the UNet model
    unet = CustomUNet(img_height=512, img_width=512, batch_size=32)

    # Define paths to your training, validation, and test data
    train_image_folder = (
        "/hpcstor6/scratch01/r/ryan.zurrin001/omama_2d/2d_512_small/train/images"
    )
    train_mask_folder = (
        "/hpcstor6/scratch01/r/ryan.zurrin001/omama_2d/2d_512_small/train/masks"
    )
    val_image_folder = (
        "/hpcstor6/scratch01/r/ryan.zurrin001/omama_2d/2d_512_small/val/images"
    )
    val_mask_folder = (
        "/hpcstor6/scratch01/r/ryan.zurrin001/omama_2d/2d_512_small/val/masks"
    )
    test_image_folder = (
        "/hpcstor6/scratch01/r/ryan.zurrin001/omama_2d/2d_512_small/test/images"
    )
    test_mask_folder = (
        "/hpcstor6/scratch01/r/ryan.zurrin001/omama_2d/2d_512_small/test/masks"
    )

    # Compile the model and start the training process
    history = unet.compile_and_train(
        train_image_folder,
        train_mask_folder,
        val_image_folder,
        val_mask_folder,
        epochs=50,
    )

    # After training, plot the training history
    unet.plot_training_history()

    # Optionally, you can save the model weights after training
    if unet.save_model_path is not None:
        try:
            unet.model.save_weights(unet.save_model_path)
            print(f"Saved model weights to {unet.save_model_path}")
        except Exception as e:
            print(f"Error saving model weights: {e}")

    # If you want to evaluate the model on a test set, you can do so like this
    unet.evaluate(test_image_folder, test_mask_folder)

    # Predictions folder
    prediction_folder = "/hpcstor6/scratch01/r/ryan.zurrin001/omama_2d/predictions"

    # Create prediction folder if it doesn't exist
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)

    # Make predictions using the trained model
    predictions = unet.predict(test_image_folder)

    # Save predictions as images
    for i, pred in enumerate(predictions):
        # Assuming that the prediction is a single-channel image with pixel values as probabilities
        pred_image = (pred.squeeze() * 255).astype(
            np.uint8
        )  # Convert from float to uint8
        cv2.imwrite(os.path.join(prediction_folder, f"prediction_{i}.png"), pred_image)
