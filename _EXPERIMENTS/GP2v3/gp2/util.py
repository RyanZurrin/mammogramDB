import numpy as np
import os, logging

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class Util:
    @staticmethod
    def disable_tensorflow_logging():
        """ """
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        logging.getLogger("tensorflow").disabled = True

    @staticmethod
    def create_A_B_Z_split(images, labels, dataset_size=1000, weights=None):
        """ """

        #
        # We split as follows
        # A 0.4*dataset_size
        # B 0.4*dataset_size
        # Z 0.2*dataset_size
        #

        A_split = int(0.4 * dataset_size)
        B_split = int(0.2 * dataset_size)
        Z_split = int(0.4 * dataset_size)

        if weights:
            A_split = int(weights["A"] * dataset_size)
            B_split = int(weights["B"] * dataset_size)
            Z_split = int(weights["Z"] * dataset_size)

        # machine data
        A = images[0:A_split, :, :, 0]
        A_labels = labels[0:A_split, :, :, 0]

        A = np.stack((A, A_labels), axis=-1)

        # human data
        B = images[A_split : A_split + B_split, :, :, 0]
        B_labels = labels[A_split : A_split + B_split, :, :, 0]

        B = np.stack((B, B_labels), axis=-1)

        # we will also provide a Z array that
        # can be used to funnel additional data later

        # funnel data
        Z = images[A_split + B_split : A_split + B_split + Z_split, :, :, 0]
        Z_labels = labels[A_split + B_split : A_split + B_split + Z_split, :, :, 0]

        Z = np.stack((Z, Z_labels), axis=-1)

        return A, B, Z

    @staticmethod
    def create_train_val_test_split(
        dataset, train_count=200, val_count=300, test_count=250, shuffle=True
    ):
        if shuffle:
            np.random.shuffle(dataset)

        train = dataset[0:train_count]
        val = dataset[train_count : train_count + val_count]
        test = dataset[train_count + val_count : train_count + val_count + test_count]

        return train, val, test

    @staticmethod
    def create_numbered_file(filename, extension):
        number = 0000
        numbered_file = filename + "_" + str(number) + extension
        while os.path.exists(numbered_file):
            number += 1
            numbered_file = filename + "_" + str(number) + extension

        return numbered_file

    @staticmethod
    def plot_accuracies(x, y1, y2):
        # Plot Line1 (Left Y Axis)
        fig, ax1 = plt.subplots(1, 1, figsize=(3, 3), dpi=80)
        (line1,) = ax1.plot(x, y1, color="tab:red", label="Classifier")
        (line2,) = ax1.plot(x, y2, color="tab:blue", label="Discriminator")
        ax1.legend(handles=[line1, line2])
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # # Plot Line2 (Right Y Axis)
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        # ax2.plot(x, y2, color='tab:blue')

        # Decorations
        # ax1 (left Y axis)
        ax1.set_xlabel("Cycle", color="tab:gray", fontsize=14)
        ax1.tick_params(axis="x", rotation=0, labelsize=12, labelcolor="tab:gray")
        ax1.set_ylabel("Accuracy", color="tab:red", fontsize=14)
        ax1.tick_params(axis="y", rotation=0, labelcolor="tab:red")
        ax1.grid(alpha=0.4)

        # ax2 (right Y axis)
        # ax2.set_ylabel("# Unemployed (1000's)", color='tab:blue', fontsize=20)
        # ax2.tick_params(axis='y', labelcolor='tab:blue')
        # ax2.set_xticks(np.arange(0, len(x), 60))
        # ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
        # ax2.set_title("Personal Savings Rate vs Unemployed: Plotting in Secondary Y Axis", fontsize=22)
        fig.tight_layout()
        plt.show()


    @staticmethod
    def plot_combined(classifier_scores_list, discriminator_scores_list=None, title="", initial_classifier_score=None):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Check if discriminator scores are provided
        has_discriminator_scores = discriminator_scores_list is not None

        colors = ["tab:red", "tab:green", "tab:blue"]
        line_styles = ["-", "--", "-."]
        for i, c_scores in enumerate(classifier_scores_list):

            # If initial_classifier_score is provided, prepend it to c_scores
            if initial_classifier_score:
                c_scores = [initial_classifier_score] + c_scores

            x = range(len(c_scores))
            y1 = [v[1] for v in c_scores]
            ax1.plot(
                x, y1, color=colors[i], linestyle=line_styles[0], label=f"Classifier Run {i+1}"
            )

            # Plot discriminator scores if they are provided
            if has_discriminator_scores:
                y2 = [v[1] for v in discriminator_scores_list[i]]
                ax1.plot(
                    x,
                    y2,
                    color=colors[i],
                    linestyle=line_styles[1],
                    label=f"Discriminator Run {i+1}",
                )

        ax1.set_xlabel("Cycle", color="tab:gray", fontsize=14)
        ax1.tick_params(axis="x", rotation=0, labelsize=12, labelcolor="tab:gray")
        ax1.set_ylabel("Accuracy", color="tab:red", fontsize=14)
        ax1.tick_params(axis="y", rotation=0, labelcolor="tab:red")
        ax1.grid(alpha=0.4)
        ax1.legend()
        ax1.set_title(title)

        # Set the x-axis ticks to be integer values
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()
        plt.show()
        

    @staticmethod
    def plot_combined_d(classifier_scores_list, discriminator_scores_list=None, 
                        title="Accuracy Plot", initial_classifier_score=None):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        colors = ["tab:red", "tab:green", "tab:blue"]
        line_styles = ["-", "--", "-."]

        has_discriminator_scores = discriminator_scores_list is not None

        # Check if the initial_classifier_score is provided
        if initial_classifier_score:
            for c_scores in classifier_scores_list:
                c_scores.insert(0, initial_classifier_score)

            if has_discriminator_scores:
                # Insert a placeholder for the discriminator scores
                for d_scores in discriminator_scores_list:
                    d_scores.insert(0, (0, None))

        for i, c_scores in enumerate(classifier_scores_list):
            x = range(len(c_scores))
            y1 = [v[1] for v in c_scores]
            ax1.plot(x, y1, color=colors[i], linestyle=line_styles[0], 
                     label=f"Classifier Run {i+1}")

            if has_discriminator_scores:
                y2 = [v[1] for v in discriminator_scores_list[i]]
                ax1.plot(x, y2, color=colors[i], linestyle=line_styles[1], 
                         label=f"Discriminator Run {i+1}")

        ax1.set_xlabel("Cycle", color="tab:gray", fontsize=14)
        ax1.tick_params(axis="x", rotation=0, labelsize=12, labelcolor="tab:gray")
        ax1.set_ylabel("Accuracy", color="tab:red", fontsize=14)
        ax1.tick_params(axis="y", rotation=0, labelcolor="tab:red")
        ax1.grid(alpha=0.4)
        ax1.legend()
        ax1.set_title(title)

        # Set the x-axis ticks to be integer values
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()
        plt.show()


    @staticmethod
    def visualize_predictions(model, images, masks, num_samples=100):
        for i in range(num_samples):
            x_val, y_val = images[i], masks[i]
            y_pred, _ = model.predict(x_val[np.newaxis, ...], y_val[np.newaxis, ...])

            # Draw the input image, true mask, and predicted mask
            fig, ax = plt.subplots(nrows=1, ncols=3)

            ax[0].imshow(x_val)
            ax[0].title.set_text("Input Image")

            ax[1].imshow(y_val.squeeze(), cmap="gray")
            ax[1].title.set_text("True Mask")

            ax[2].imshow(y_pred.squeeze(), cmap="gray")
            ax[2].title.set_text("Predicted Mask")

            plt.show()
            


    def display_image_and_mask(image, mask):
        """
        Display an image and its mask side-by-side.

        Parameters:
        -----------
        image : ndarray
            The image to be displayed.
        mask : ndarray
            The mask to be displayed.

        Returns:
        --------
        None
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Display image
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title('Image')

        # Display mask
        axes[1].imshow(mask)
        axes[1].axis('off')
        axes[1].set_title('Mask')

        plt.tight_layout()
        plt.show()
