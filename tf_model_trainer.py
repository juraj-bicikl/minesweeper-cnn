import time
import tensorflow as tf
from tf_model_runner import TfModelRunner
from tf_data import TfData


class TfTrainer:

    def __init__(self,
                 model_runner: TfModelRunner,
                 learn_data: TfData,
                 validation_data: TfData,
                 batch_size: int,
                 interval: int,
                 max_num_intervals: int,
                 tensorboard_dir: str):

        self.model_runner = model_runner
        self.learn_data = learn_data
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.interval = interval
        self.max_num_intervals = max_num_intervals
        self.tensorboard_dir = tensorboard_dir
        self.graph = model_runner.graph
        self.writer = None

    def train(self, model_path=None):

        self.model_runner.start_session()
        if model_path:
            self.model_runner.load(model_path)
        else:
            self.model_runner.init()

        self.writer = tf.summary.FileWriter(self.tensorboard_dir, graph=self.graph.graph)
        validation_tgt, validation_ins = self.validation_data.all_rows()

        step = 0
        for i in range(self.max_num_intervals):
            data_time = 0.0
            train_time = 0.0
            for _ in range(self.interval):
                t0 = time.time()
                target, inputs = self.learn_data.random_batch(self.batch_size)
                t1 = time.time()
                self.model_runner.run_train_step(feed_dict={self.graph.input: inputs, self.graph.target: target})
                t2 = time.time()
                data_time += (t1 - t0)
                train_time += (t2 - t1)
                step += 1

            summary = self.model_runner.run_inference(self.graph.summary, {self.graph.input: validation_ins, self.graph.target: validation_tgt})
            self.writer.add_summary(summary, global_step=step)
            print("interval %d done, data time: %.2f s, train time: %.2f s" % (i + 1, data_time, train_time))

        self.writer.close()
        self.model_runner.end_session(save=True)


class TfTrainerWithRefreshingData:

    def __init__(self,
                 model_runner: TfModelRunner,
                 learn_data_generator,
                 validation_data: TfData,
                 batch_size: int,
                 interval: int,
                 max_num_intervals: int,
                 learn_data_refresh_period: int,
                 tensorboard_dir: str):

        self.model_runner = model_runner
        self.learn_data_generator = learn_data_generator
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.interval = interval
        self.max_num_intervals = max_num_intervals
        self.learn_data_refresh_period = learn_data_refresh_period
        self.tensorboard_dir = tensorboard_dir
        self.graph = model_runner.graph
        self.writer = None

    def train(self, model_path=None):

        self.model_runner.start_session()
        if model_path:
            self.model_runner.load(model_path)
        else:
            self.model_runner.init()

        self.writer = tf.summary.FileWriter(self.tensorboard_dir, graph=self.graph.graph)
        validation_tgt, validation_ins = self.validation_data.all_rows()

        step = 0
        for i in range(self.max_num_intervals):
            if i % self.learn_data_refresh_period == 0:
                learn_data = self.learn_data_generator()

            data_time = 0.0
            train_time = 0.0
            for _ in range(self.interval):
                t0 = time.time()
                target, inputs = learn_data.random_batch(self.batch_size)
                t1 = time.time()
                self.model_runner.run_train_step(feed_dict={self.graph.input: inputs, self.graph.target: target})
                t2 = time.time()
                data_time += (t1 - t0)
                train_time += (t2 - t1)
                step += 1

            t0 = time.time()
            summary = self.model_runner.run_inference(self.graph.summary, {self.graph.input: validation_ins, self.graph.target: validation_tgt})
            summary_time = time.time() - t0
            self.writer.add_summary(summary, global_step=step)
            print("interval %d done, data time: %.2f s, train time: %.2f s, summary time: %.2f" % (i + 1, data_time, train_time, summary_time))

        self.writer.close()
        self.model_runner.end_session(save=True)