import torch
import torchmetrics


def create_metrics():
    """
    Create and return the metrics for denoising.

    Returns:
        metrics_denoise (MetricsDenoise): The metrics for denoising.
    """
    metrics_denoise = MetricsDenoise(
        loss=torchmetrics.MeanMetric(),
        miou=torchmetrics.classification.BinaryJaccardIndex(),
    )
    metrics_denoise.to(torch.device("cuda"))

    return metrics_denoise


class MetricsDenoise():
    """
    Class for computing metrics for denoising tasks.

    Args:
        **kwargs: Additional keyword arguments representing the metrics to be computed.

    Attributes:
        metrics (dict): A dictionary containing the metrics to be computed.

    Methods:
        apply_threshold: Applies a threshold to the predicted values.
        update: Updates the metrics with the given loss, predicted values, and ground truth values.
        reset: Resets the metrics to their initial state.
        compute: Computes the metrics and returns the results.
        to: Moves the metrics to the specified device.
    """

    def __init__(self, **kwargs):
        self.metrics = {k: v for k, v in kwargs.items()}

    def apply_threshold(self, y, threshold=0.5):
        """
        Applies a threshold to the predicted values.

        Args:
            y (torch.Tensor): The predicted values.
            threshold (float): The threshold value. Default is 0.5.

        Returns:
            torch.Tensor: The thresholded values.
        """
        return (y > threshold).to(torch.uint8)

    def update(self, loss, pred, y):
        """
        Updates the metrics with the given loss, predicted values, and ground truth values.

        Args:
            loss (torch.Tensor): The loss value.
            pred (torch.Tensor): The predicted values.
            y (torch.Tensor): The ground truth values.
        """
        pred_th = self.apply_threshold(pred)
        y_th = self.apply_threshold(y)

        for metric_name in self.metrics.keys():
            if metric_name == "loss":
                self.metrics["loss"].update(loss)
            elif metric_name == "miou":
                self.metrics["miou"].update(pred_th, y_th)

    def reset(self):
        """
        Resets the metrics to their initial state.
        """
        for metric in self.metrics.values():
            metric.reset()

    def compute(self):
        """
        Computes the metrics and returns the results.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        return {k: v.compute().item() for k, v in self.metrics.items()}

    def to(self, device):
        """
        Moves the metrics to the specified device.

        Args:
            device (torch.device): The device to move the metrics to.
        """
        self.metrics = {k: v.to(device) for k, v in self.metrics.items()}
