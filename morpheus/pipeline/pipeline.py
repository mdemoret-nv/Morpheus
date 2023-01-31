# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import signal
import time
import typing
from collections import defaultdict
from functools import partial

import mrc
import networkx
from tqdm import tqdm

import cudf

from morpheus.config import Config
from morpheus.pipeline.receiver import Receiver
from morpheus.pipeline.sender import Sender
from morpheus.pipeline.source_stage import SourceStage
from morpheus.pipeline.stage import Stage
from morpheus.pipeline.stream_wrapper import StreamWrapper
from morpheus.utils.type_utils import pretty_print_type_name

logger = logging.getLogger(__name__)

StageT = typing.TypeVar("StageT", bound=StreamWrapper)


class Pipeline():
    """
    Class for building your pipeline. A pipeline for your use case can be constructed by first adding a
    `Source` via `set_source` then any number of downstream `Stage` classes via `add_stage`. The order stages
    are added with `add_stage` determines the order in which stage executions are carried out. You can use
    stages included within Morpheus or your own custom-built stages.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.

    """

    def __init__(self, c: Config):
        self._source_count: int = None  # Maximum number of iterations for progress reporting. None = Unknown/Unlimited

        self._id_counter = 0

        # Complete set of nodes across segments in this pipeline
        self._stages: typing.Set[Stage] = set()

        # Complete set of sources across segments in this pipeline
        self._sources: typing.Set[SourceStage] = set()

        # Dictionary containing segment information for this pipeline
        self._segments: typing.Dict = defaultdict(lambda: {"nodes": set(), "ingress_ports": [], "egress_ports": []})

        self._exec_options = mrc.Options()
        self._exec_options.topology.user_cpuset = "0-{}".format(c.num_threads - 1)
        self._exec_options.engine_factories.default_engine_type = mrc.core.options.EngineType.Thread

        # Set the default channel size
        mrc.Config.default_channel_size = c.edge_buffer_size

        self.batch_size = c.pipeline_batch_size

        self._segment_graphs = defaultdict(lambda: networkx.DiGraph())

        self._is_built = False
        self._is_build_complete = False
        self._is_started = False

        self._mrc_executor: mrc.Executor = None
        self._mrc_pipeline: mrc.Pipeline = None

    @property
    def is_built(self) -> bool:
        return self._is_built

    def _add_id_col(self, x: cudf.DataFrame):

        # Data in stream is cudf Dataframes at this point. We need an ID column before continuing
        x.insert(0, 'ID', range(self._id_counter, self._id_counter + len(x)))
        self._id_counter += len(x)

        return x

    def add_stage(self, stage: StageT, segment_id: str = "main") -> StageT:
        """
        Add a stage to a segment in the pipeline.

        Parameters
        ----------
        stage : Stage
            The stage object to add. It cannot be already added to another `Pipeline` object.

        segment_id : str
            ID indicating what segment the stage should be added to.
        """

        assert stage._pipeline is None or stage._pipeline is self, "A stage can only be added to one pipeline at a time"

        segment_nodes = self._segments[segment_id]["nodes"]
        segment_graph = self._segment_graphs[segment_id]

        # Add to list of stages if it's a stage, not a source
        if (isinstance(stage, Stage)):
            segment_nodes.add(stage)
            self._stages.add(stage)
        elif (isinstance(stage, SourceStage)):
            segment_nodes.add(stage)
            self._sources.add(stage)
        else:
            raise NotImplementedError("add_stage() failed. Unknown node type: {}".format(type(stage)))

        stage._pipeline = self

        segment_graph.add_node(stage)

        return stage

    def add_edge(self,
                 start: typing.Union[StreamWrapper, Sender],
                 end: typing.Union[Stage, Receiver],
                 segment_id: str = "main"):
        """
        Create an edge between two stages and add it to a segment in the pipeline.

        Parameters
        ----------
        start : typing.Union[StreamWrapper, Sender]
            The start of the edge or parent stage.

        end : typing.Union[Stage, Receiver]
            The end of the edge or child stage.

        segment_id : str
            ID indicating what segment the edge should be added to.
        """

        if (isinstance(start, StreamWrapper)):
            start_port = start.output_ports[0]
        elif (isinstance(start, Sender)):
            start_port = start

        if (isinstance(end, Stage)):
            end_port = end.input_ports[0]
        elif (isinstance(end, Receiver)):
            end_port = end

        start_port._output_receivers.append(end_port)
        end_port._input_senders.append(start_port)

        segment_graph = self._segment_graphs[segment_id]
        segment_graph.add_edge(start_port.parent,
                               end_port.parent,
                               start_port_idx=start_port.port_number,
                               end_port_idx=end_port.port_number)

    def add_segment_edge(self,
                         egress_stage: Stage,
                         egress_segment: str,
                         ingress_stage: Stage,
                         ingress_segment: str,
                         port_pair: typing.Union[str, typing.Tuple[str, typing.Type, bool]]):
        """
        Create an edge between two segments in the pipeline.

        Parameters
        ----------

        egress_stage : Stage
            The egress stage of the parent segment

        egress_segment : str
            Segment ID of the parent segment

        ingress_stage : Stage
            The ingress stage of the child segment

        ingress_segment : str
            Segment ID of the child segment

        port_pair : typing.Union[str, typing.Tuple]
            Either the ID of the egress segment, or a tuple with the following three elements:
                * str: ID of the egress segment
                * class: type being sent (typically `object`)
                * bool: If the type is a shared pointer (typically should be `False`)
        """
        egress_edges = self._segments[egress_segment]["egress_ports"]
        egress_edges.append({
            "port_pair": port_pair,
            "input_sender": egress_stage.unique_name,
            "output_receiver": ingress_stage.unique_name,
            "receiver_segment": ingress_segment
        })

        ingress_edges = self._segments[ingress_segment]["ingress_ports"]
        ingress_edges.append({
            "port_pair": port_pair,
            "input_sender": egress_stage.unique_name,
            "sender_segment": egress_segment,
            "output_receiver": ingress_stage.unique_name
        })

    def build(self):
        """
        This function sequentially activates all the Morpheus pipeline stages passed by the users to execute a
        pipeline. For the `Source` and all added `Stage` objects, `StreamWrapper.build` will be called sequentially to
        construct the pipeline.

        Once the pipeline has been constructed, this will start the pipeline by calling `Source.start` on the source
        object.
        """
        assert not self._is_built, "Pipeline can only be built once!"
        assert len(self._sources) > 0, "Pipeline must have a source stage"

        logger.info("====Registering Pipeline====")

        self._mrc_executor = mrc.Executor(self._exec_options)

        self._mrc_pipeline = mrc.Pipeline()

        def inner_build(builder: mrc.Builder, segment_id: str):
            segment_graph = self._segment_graphs[segment_id]

            # This should be a BFS search from each source nodes; but, since we don't have source stage loops
            # topo_sort provides a reasonable approximation.
            for stage in networkx.topological_sort(segment_graph):
                if (stage.can_build()):
                    stage.build(builder)

            if (not all([x.is_built for x in segment_graph.nodes()])):
                # raise NotImplementedError("Circular pipelines are not yet supported!")
                logger.warning("Cyclic pipeline graph detected! Building with reduced constraints")

                for stage in segment_graph.nodes():
                    if (stage.can_build(check_ports=True)):
                        stage.build()

            if (not all(x.is_built for x in segment_graph.nodes())):
                raise RuntimeError("Could not build pipeline. Ensure all types can be determined")

            # Finally, execute the link phase (only necessary for circular pipelines)
            # for s in source_and_stages:
            for stage in segment_graph.nodes():
                for port in stage.input_ports:
                    port.link()

        logger.info("====Building Pipeline====")
        for segment_id in self._segments.keys():
            logger.info(f"====Building Segment: {segment_id}====")
            segment_ingress_ports = self._segments[segment_id]["ingress_ports"]
            segment_egress_ports = self._segments[segment_id]["egress_ports"]
            segment_inner_build = partial(inner_build, segment_id=segment_id)

            self._mrc_pipeline.make_segment(segment_id, [port_info["port_pair"] for port_info in segment_ingress_ports],
                                            [port_info["port_pair"] for port_info in segment_egress_ports],
                                            segment_inner_build)
            logger.info("====Building Segment Complete!====")

        logger.info("====Building Pipeline Complete!====")
        self._is_build_complete = True

        # Finally call _on_start
        self._on_start()

        self._mrc_executor.register_pipeline(self._mrc_pipeline)

        self._is_built = True

        logger.info("====Registering Pipeline Complete!====")

    def _start(self):
        assert self._is_built, "Pipeline must be built before starting"

        logger.info("====Starting Pipeline====")

        self._mrc_executor.start()

        logger.info("====Pipeline Started====")

    def stop(self):
        """
        Stops all running stages and the underlying MRC pipeline.
        """

        logger.info("====Stopping Pipeline====")
        for s in list(self._sources) + list(self._stages):
            s.stop()

        self._mrc_executor.stop()

        logger.info("====Pipeline Stopped====")

    async def join(self):
        """
        Suspend execution all currently running stages and the MRC pipeline.
        Typically called after `stop`.
        """
        try:
            await self._mrc_executor.join_async()
        except Exception:
            logger.exception("Exception occurred in pipeline. Rethrowing")
            raise
        finally:
            # Make sure these are always shut down even if there was an error
            for s in list(self._sources):
                s.stop()

            # First wait for all sources to stop. This only occurs after all messages have been processed fully
            for s in list(self._sources):
                await s.join()

            # Now that there is no more data, call stop on all stages to ensure shutdown (i.e., for stages that have
            # their own worker loop thread)
            for s in list(self._stages):
                s.stop()

            # Now call join on all stages
            for s in list(self._stages):
                await s.join()

    async def _build_and_start(self):

        if (not self.is_built):
            try:
                self.build()
            except Exception:
                logger.exception("Error occurred during Pipeline.build(). Exiting.", exc_info=True)
                return

        await self._async_start()

        self._start()

    async def _async_start(self):

        # Loop over all stages and call on_start if it exists
        for s in self._stages:
            await s.start_async()

    def _on_start(self):

        # Only execute this once
        if (self._is_started):
            return

        # Stop from running this twice
        self._is_started = True

        logger.debug("Starting! Time: {}".format(time.time()))

        # Loop over all stages and call on_start if it exists
        for s in self._stages:
            s.on_start()

    def visualize(self, filename: str = None, **graph_kwargs):
        """
        Output a pipeline diagram to `filename`. The file format of the diagrame is inferred by the extension of
        `filename`. If the directory path leading to `filename` does not exist it will be created, if `filename` already
        exists it will be overwritten.  Requires the graphviz library.
        """

        # Mimic the streamz visualization
        # 1. Create graph (already done since we use networkx under the hood)
        # 2. Readable graph
        # 3. To graphviz
        # 4. Draw
        import graphviz

        # Default graph attributes
        graph_attr = {
            "nodesep": "1",
            "ranksep": "1",
            "pad": "0.5",
        }

        # Allow user to overwrite defaults
        graph_attr.update(graph_kwargs)

        gv_graph = graphviz.Digraph(graph_attr=graph_attr)
        gv_graph.attr(compound="true")
        gv_subgraphs = {}

        # Need a little different functionality for left/right vs vertical
        is_lr = graph_kwargs.get("rankdir", None) == "LR"

        start_def_port = ":e" if is_lr else ":s"
        end_def_port = ":w" if is_lr else ":n"

        def has_ports(n: StreamWrapper, is_input):
            if (is_input):
                return len(n.input_ports) > 0
            else:
                return len(n.output_ports) > 0

        if not self._is_build_complete:
            raise RuntimeError("Pipeline.visualize() requires that the Pipeline has been started before generating "
                               "the visualization. Please call Pipeline.start(), Pipeline.build_and_start() or "
                               "Pipeline.run() before calling Pipeline.visualize(). This is a known issue and will "
                               "be fixed in a future release.")

        # Now build up the nodes
        for idx, segment_id in enumerate(self._segments):
            gv_subgraphs[segment_id] = graphviz.Digraph(f"cluster_{segment_id}")
            gv_subgraph = gv_subgraphs[segment_id]
            gv_subgraph.attr(label=segment_id)
            for n, attrs in typing.cast(typing.Mapping[StreamWrapper, dict],
                                        self._segment_graphs[segment_id].nodes).items():
                node_attrs = attrs.copy()

                label = ""

                show_in_ports = has_ports(n, is_input=True)
                show_out_ports = has_ports(n, is_input=False)

                # Build the ports for the node. Only show ports if there are any
                # (Would like to have this not show for one port, but the lines get all messed up)
                if (show_in_ports):
                    in_port_label = " {{ {} }} | ".format(" | ".join(
                        [f"<u{x.port_number}> input_port: {x.port_number}" for x in n.input_ports]))
                    label += in_port_label

                label += n.unique_name

                if (show_out_ports):
                    out_port_label = " | {{ {} }}".format(" | ".join(
                        [f"<d{x.port_number}> output_port: {x.port_number}" for x in n.output_ports]))
                    label += out_port_label

                if (show_in_ports or show_out_ports):
                    label = f"{{ {label} }}"

                node_attrs.update({
                    "label": label,
                    "shape": "record",
                    "fillcolor": "white",
                })
                # TODO: Eventually allow nodes to have different attributes based on type
                # node_attrs.update(n.get_graphviz_attrs())
                gv_subgraph.node(n.unique_name, **node_attrs)

        # Build up edges
        for segment_id in self._segments:
            gv_subgraph = gv_subgraphs[segment_id]
            for e, attrs in typing.cast(typing.Mapping[typing.Tuple[StreamWrapper, StreamWrapper], dict],
                                        self._segment_graphs[segment_id].edges()).items():  # noqa: E501

                edge_attrs = {}

                start_name = e[0].unique_name

                # Append the port if necessary
                if (has_ports(e[0], is_input=False)):
                    start_name += f":d{attrs['start_port_idx']}"
                else:
                    start_name += start_def_port

                end_name = e[1].unique_name

                if (has_ports(e[1], is_input=True)):
                    end_name += f":u{attrs['end_port_idx']}"
                else:
                    end_name += end_def_port

                # Now we only want to show the type label in some scenarios:
                # 1. If there is only one edge between two nodes, draw type in middle "label"
                # 2. For port with an edge, only draw that port's type once (using index 0 of the senders/receivers)
                start_port_idx = int(attrs['start_port_idx'])
                end_port_idx = int(attrs['end_port_idx'])

                out_port = e[0].output_ports[start_port_idx]
                in_port = e[1].input_ports[end_port_idx]

                # Check for situation #1
                if (len(in_port._input_senders) == 1 and len(out_port._output_receivers) == 1
                        and (in_port.in_type == out_port.out_type)):

                    edge_attrs["label"] = pretty_print_type_name(in_port.in_type)
                else:
                    rec_idx = out_port._output_receivers.index(in_port)
                    sen_idx = in_port._input_senders.index(out_port)

                    # Add type labels if available
                    if (rec_idx == 0 and out_port.out_type is not None):
                        edge_attrs["taillabel"] = pretty_print_type_name(out_port.out_type)

                    if (sen_idx == 0 and in_port.in_type is not None):
                        edge_attrs["headlabel"] = pretty_print_type_name(in_port.in_type)

                gv_subgraph.edge(start_name, end_name, **edge_attrs)

            for egress_port in self._segments[segment_id]["egress_ports"]:
                gv_graph.edge(egress_port["input_sender"],
                              egress_port["output_receiver"],
                              style="dashed",
                              label=f"Segment Port: {egress_port['port_pair'][0]}")

        for key, gv_subgraph in gv_subgraphs.items():
            gv_graph.subgraph(gv_subgraph)

        file_format = os.path.splitext(filename)[-1].replace(".", "")

        viz_binary = gv_graph.pipe(format=file_format)
        # print(gv_graph.source)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "wb") as f:
            f.write(viz_binary)

    async def run_async(self):
        """
        This function sets up the current asyncio loop, builds the pipeline, and awaits on it to complete.
        """
        loop = asyncio.get_running_loop()

        def error_handler(_, context: dict):

            msg = "Unhandled exception in async loop! Exception: \n{}".format(context["message"])
            exception = context.get("exception", Exception())

            logger.critical(msg, exc_info=exception)

        loop.set_exception_handler(error_handler)

        exit_count = 0

        # Handles Ctrl+C for graceful shutdown
        def term_signal():

            nonlocal exit_count
            exit_count = exit_count + 1

            if (exit_count == 1):
                tqdm.write("Stopping pipeline. Please wait... Press Ctrl+C again to kill.")
                self.stop()
            else:
                tqdm.write("Killing")
                exit(1)

        for s in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(s, term_signal)

        try:
            await self._build_and_start()

            # Wait for completion
            await self.join()

        except KeyboardInterrupt:
            tqdm.write("Stopping pipeline. Please wait...")

            # Stop the pipeline
            self.stop()

            # Wait again for nice completion
            await self.join()

        finally:
            # Shutdown the async generator sources and exit
            logger.info("====Pipeline Complete====")

    def run(self):
        """
        This function makes use of asyncio features to keep the pipeline running indefinitely.
        """

        # Use asyncio.run() to launch the pipeline. This creates and destroys an event loop so re-running a pipeline in
        # the same process wont fail
        asyncio.run(self.run_async())
