from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from math import atan2

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :-1], data[:, -1].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        X, y = load_dataset(f"../datasets/{f}")
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(fit: Perceptron, x: np.ndarray, val: int):
            losses.append(fit.loss(X, y))

        cur_perceptron = Perceptron(callback=callback).fit(X, y)
        # Plot figure
        px.line(losses, title=f"Training Loss Values as a Function of Training Iterations - {n} Data",
                x=np.array(range(len(losses))), y=losses, labels={'x': 'Iterations', 'y': 'Loss'}).show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/gaussian1.npy")
        # Fit models and predict over training set
        gnb_model = GaussianNaiveBayes().fit(X, y)
        lda_model = LDA().fit(X, y)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy

        gnb_prediction = gnb_model.predict(X)
        gnb_accuracy = round(accuracy(y, gnb_prediction), 3)
        lda_prediction = lda_model.predict(X)
        lda_accuracy = round(accuracy(y, lda_prediction), 3)

        fig = make_subplots(rows=1, cols=2, subplot_titles=[f" Naive Gaussian Bayes model, accuracy = {gnb_accuracy}",
                                                            f"LDA model, accuracy = {lda_accuracy}"]) \
            .add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(size=20, color=gnb_prediction,
                                                                                    symbol=y, line_width=1)), row=1,
                       col=1) \
            .add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(size=20, color=lda_prediction,
                                                                                    symbol=y, line_width=1)), row=1,
                       col=2) \
            .add_trace(go.Scatter(x=gnb_model.mu_[:, 0], y=gnb_model.mu_[:, 1], mode='markers',
                                  marker=dict(size=20, color='black', symbol='x')), row=1, col=1) \
            .add_trace(go.Scatter(x=lda_model.mu_[:, 0], y=lda_model.mu_[:, 1], mode='markers', marker=dict(size=20,
                                                                                                            color='black',
                                                                                                            symbol='x')),
                       row=1, col=2)

        # add ellipses for gnb model:

        for k in gnb_model.classes_:
            cov_k = np.diag(gnb_model.vars_[k])
            l1, l2 = tuple(np.linalg.eigvalsh(cov_k)[::-1])
            theta = atan2(l1 - cov_k[0, 0], cov_k[0, 1]) if cov_k[0, 1] != 0 else (np.pi / 2 if
                                                                                   cov_k[0, 0] < cov_k[1, 1] else 0)
            t = np.linspace(0, 2 * np.pi, 100)
            xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
            ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))
            fig.add_trace(go.Scatter(x=gnb_model.mu_[k][0] + xs, y=gnb_model.mu_[k][1] + ys, mode="lines",
                                     marker_color="black"), row=1, col=1)

        # add ellipses for lda model:

        for k in lda_model.classes_:
            l1, l2 = tuple(np.linalg.eigvalsh(lda_model.cov_)[::-1])
            theta = atan2(l1 - lda_model.cov_[0, 0], lda_model.cov_[0, 1]) if lda_model.cov_[0, 1] != 0 \
                else (np.pi / 2 if lda_model.cov_[0, 0] < lda_model.cov_[1, 1] else 0)
            t = np.linspace(0, 2 * np.pi, 100)
            xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
            ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))
            fig.add_trace(go.Scatter(x=lda_model.mu_[k][0] + xs, y=lda_model.mu_[k][1] + ys, mode="lines",
                                     marker_color="black"), row=1, col=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
