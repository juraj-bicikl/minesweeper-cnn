import os
import numpy as np
import tensorflow as tf
from tf_graph import TfGraph


class TfModelRunner:

    def __init__(self,
                 graph: TfGraph,
                 saved_model_path: str,
                 extra_train_inputs: {} = {},
                 extra_inference_inputs: {} = {}):

        self.graph = graph
        self.saved_model_path = saved_model_path
        self.extra_train_inputs = extra_train_inputs
        self.extra_inference_inputs = extra_inference_inputs
        self.session = None

    def __run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        assert self.session, "session is not active"
        return self.session.run(fetches, feed_dict, options, run_metadata)

    def run_inference(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return self.__run(fetches, {**feed_dict, **self.extra_inference_inputs}, options, run_metadata)

    def init(self):
        self.__run(fetches=self.graph.init)

    def run_train_step(self, feed_dict):
        self.__run(self.graph.train, {**feed_dict, **self.extra_train_inputs})

    def evaluate(self, input: np.ndarray) -> np.ndarray:
        return self.__run(self.graph.output, {**{self.graph.input: input}, **self.extra_inference_inputs})

    def save(self, path=None):
        assert self.session, "session is not active"
        the_path = path if path else self.saved_model_path
        parent_dir = os.path.split(the_path)[0]
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        self.graph.saver.save(self.session, the_path)

    def load(self, path=None):
        assert self.session, "session is not active"
        the_path = path if path else self.saved_model_path
        self.graph.saver.restore(self.session, the_path)

    def start_session(self):
        if not self.session:
            self.session = tf.Session(graph=self.graph.graph)

    def end_session(self, save=False):
        if self.session:
            if save:
                self.save()
            self.session.close()
            self.session = None
