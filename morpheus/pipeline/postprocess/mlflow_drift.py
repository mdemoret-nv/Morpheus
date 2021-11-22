import logging
import typing

import cupy as cp
import mlflow
import neo

from morpheus.config import Config
from morpheus.pipeline.messages import MultiResponseMessage
from morpheus.pipeline.messages import MultiResponseProbsMessage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair

logger = logging.getLogger(__name__)


class MLFlowDriftStage(SinglePortStage):
    """
    Caculates model drift over time and reports the information to ML Flow

    Parameters
    ----------
    c : Config
        The global config.
    tracking_uri : str, optional
        The ML Flow tracking URI to connect to the tracking backend. If not speficied, MF Flow will use 'file:///mlruns'
        relative to the current directory, by default None
    experiment_name : str, optional
        The experiement name to use in ML Flow, by default "Morpheus"
    run_id : str, optional
        The ML Flow Run ID to report metrics to. If unspecified, Morpheus will attempt to reuse any previously created
        runs that are still active. Otherwise, a new run will be created. By default, runs are left in an active state.
    labels : typing.List[str], optional
        Converts probability indexes into labels for the ML Flow UI. If no labels are specified, the probability labels
        are determined by the pipeline mode., by default None
    batch_size : int, optional
        The batch size to calculate model drift statistics. Allows for increasing or decreasing how much data is
        reported to MLFlow. Default is -1 which will use the pipeline batch_size., by default -1
    force_new_run : bool, optional
        Whether or not to reuse the most recent run ID in ML Flow or create a new one each time the pipeline is run, by
        default False
    """
    def __init__(self,
                 c: Config,
                 tracking_uri: str = None,
                 experiment_name: str = "Morpheus",
                 run_id: str = None,
                 labels: typing.List[str] = None,
                 batch_size: int = -1,
                 force_new_run: bool = False):
        super().__init__(c)

        self._feature_length = c.feature_length
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_id = run_id
        self._labels = c.class_labels if labels is None or len(labels) == 0 else labels

        if (batch_size == -1):
            self._batch_size = c.pipeline_batch_size
        else:
            self._batch_size = batch_size

        if (self._batch_size > c.pipeline_batch_size):
            logger.warning(("Warning: MLFlowDriftStage batch_size (%d) is greater than pipeline_batch_size (%d). "
                            "Reducing stage batch_size to pipeline_batch_size"),
                           self._batch_size,
                           c.pipeline_batch_size)
            self._batch_size = c.pipeline_batch_size

        # Set the active run up
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        if (run_id is None and not force_new_run):
            # Get the current experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)

            # Find all active runs
            active_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                                             order_by=["attribute.start_time"])

            if (len(active_runs) > 0 and "tags.morpheus.type" in active_runs):
                morpheus_runs = active_runs[active_runs["tags.morpheus.type"] == "drift"]

                if (len(morpheus_runs) > 0):
                    run_id = morpheus_runs.head(n=1)["run_id"].iloc[0]

        mlflow.start_run(run_id, run_name="Model Drift", tags={"morpheus.type": "drift"})

    @property
    def name(self) -> str:
        return "mlflow_drift"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[MultiResponseMessage, ]
            Accepted input types

        """
        return (MultiResponseProbsMessage, )

    def _calc_drift(self, x: MultiResponseProbsMessage):

        # All probs in a batch will be calculated
        shifted = cp.abs(x.probs - 0.5) + 0.5

        # Make sure the labels list is long enough
        for x in range(len(self._labels), shifted.shape[1]):
            self._labels.append(str(x))

        for i in list(range(0, x.count, self._batch_size)):
            start = i
            end = min(start + self._batch_size, x.count)
            mean = cp.mean(shifted[start:end, :], axis=0, keepdims=True)

            # For each column, report the metric
            metrics = {self._labels[y]: mean[0, y].item() for y in range(mean.shape[1])}

            metrics["total"] = cp.mean(mean).item()

            mlflow.log_metrics(metrics)

        return x

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        # Convert the messages to rows of strings
        node = seg.make_node(self.unique_name, self._calc_drift)
        seg.make_edge(input_stream[0], node)
        stream = node

        # Return input unchanged
        return stream, MultiResponseMessage
