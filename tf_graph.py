import tensorflow as tf


class TfGraph:

    def __init__(self,
                 graph: tf.Graph,
                 saver: tf.train.Saver,
                 train_name='train',
                 init_name='init',
                 input_name='input:0',
                 target_name='target:0',
                 output_name='output:0',
                 error_cost_name='error_cost:0',
                 total_cost_name='total_cost:0',
                 summary_name='summary'):

        self.graph = graph
        self.saver = saver
        self.train = train_name
        self.init = init_name
        self.input = input_name
        self.target = target_name
        self.output = output_name
        self.error_cost = error_cost_name
        self.total_cost = total_cost_name
        self.summary = summary_name

    def write_as_graph_def(self, dir, name):
        graph_def = self.graph.as_graph_def()
        tf.train.write_graph(graph_def, dir, name, as_text=True)
