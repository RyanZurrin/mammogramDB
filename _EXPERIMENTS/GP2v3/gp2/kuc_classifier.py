import os
import pickle
from .classifier import Classifier

from .util import Util

from tensorflow.keras import callbacks


class KUC_Classifier(Classifier):
    def __init__(self, verbose, workingdir, **kwargs):
        super().__init__(verbose, workingdir)
        self.name = None
        self.metric = None
        self.loss = None
        self.optimizer = None
        self.model = None

    def build(self):
        """Build the model."""
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metric
        )

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        patience_counter=2,
        batch_size=64,
        epochs=100,
        call_backs=None,
    ):
        """Train the model.
        Parameters
        ----------
        X_train : numpy.ndarray
            The training images.
        y_train : numpy.ndarray
            The training masks.
        X_val : numpy.ndarray
            The validation images.
        y_val : numpy.ndarray
            The validation masks.
        patience_counter : int
            The number of epochs to wait before early stopping.
        batch_size : int
            The batch size to use.
        epochs : int
            The number of epochs to train for.
        call_backs : list
            The list of callbacks to use.
        """

        super().train(X_train, y_train, X_val, y_val)
        checkpoint_file = os.path.join(self.workingdir, self.name)
        checkpoint_file = Util.create_numbered_file(
            checkpoint_file, f"{self.name}_model"
        )

        if call_backs is None:
            call_backs = [
                callbacks.EarlyStopping(
                    patience=patience_counter, monitor="loss", verbose=0
                ),
                callbacks.ModelCheckpoint(
                    checkpoint_file,
                    save_weights_only=False,
                    monitor="val_loss",
                    mode="min",
                    verbose=0,
                    save_best_only=True,
                ),
            ]
        else:
            call_backs = call_backs

        history = self.model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=call_backs,
        )

        history_file = os.path.join(self.workingdir, f"{self.name}_history")
        history_file = Util.create_numbered_file(history_file, ".pkl")
        with open(history_file, "wb") as f:
            pickle.dump(history.history, f)

        print(f"Model saved to: {checkpoint_file}")
        print(f"History saved to: {history_file}")

        return history

    def predict(self, X_test, y_pred, threshold=0.5):
        """Predict the masks for the given images.
        Parameters
        ----------
        X_text : numpy.ndarray
            The images to predict the masks for.
        y_pred : numpy.ndarray
            The predicted masks.
        threshold : float
            The threshold to use for the predictions.
        """
        predictions = self.model.predict(X_test)

        predictions[predictions >= threshold] = 1.0
        predictions[predictions < threshold] = 0.0

        scores = self.model.evaluate(X_test, y_pred, verbose=0)

        return predictions, scores

    # print all the classes attributes using vars() function
    def __str__(self):
        return str(vars(self))

    @staticmethod
    def dice_coeff(y_true, y_pred, smooth=1e-9):
        """Calculate the dice coefficient.

        Parameters
        ----------
        y_true : numpy.ndarray
            The true masks.
        y_pred : numpy.ndarray
            The predicted masks.
        smooth : float
            The smoothing factor.

        Returns
        -------
        float
            The dice coefficient.
        """
        import tensorflow as tf

        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        return (2.0 * intersection + smooth) / (union + smooth)

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        """Calculate the loss.
        Parameters
        ----------
        y_true : numpy.ndarray
            The true masks.
        y_pred : numpy.ndarray
            The predicted masks.
        Returns
        -------
        float
            The loss.
        """
        import tensorflow as tf

        return tf.keras.losses.binary_crossentropy(y_true, y_pred) + (
            1 - KUC_Classifier.dice_coeff(y_true, y_pred)
        )

    @staticmethod
    def hybrid_loss(y_true, y_pred):
        """Calculate the loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            The true masks.
        y_pred : numpy.ndarray
            The predicted masks.

        Returns
        -------
        float
            The loss.
        """
        import tensorflow as tf
        from keras_unet_collection import losses

        # check that the types of both y_true and y_pred are the same
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4 / 3)
        loss_iou = losses.iou_seg(y_true, y_pred)

        # (x)
        # loss_ssim = losses.ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)

        return loss_focal + loss_iou  # +loss_ssim

    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1):
        """Calculate the dice loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            The true masks.
        y_pred : numpy.ndarray
            The predicted masks.
        smooth : float
            The smoothing factor.

        Returns
        -------
        float
            The dice loss.
        """
        return 1 - KUC_Classifier.dice_coeff(y_true, y_pred, smooth)
