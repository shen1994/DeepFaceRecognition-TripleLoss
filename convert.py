# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:18:01 2018

@author: shen1994
"""

import os
import tensorflow as tf
from keras import backend as K
from n_model import inception_v2
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework import graph_io

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    image_shape = (192, 192, 3)
    embedding_size = 512

    network = inception_v2(image_shape, embedding_size=embedding_size)
    network.load_weights('model/weights.70.hdf5', by_name=True)

    print('input name is: ', network.input.name)
    print('output name is: ', network.output.name)
    
    K.set_learning_phase(0)
    frozen_graph = freeze_session(K.get_session(), output_names=[network.output.op.name])
    graph_io.write_graph(frozen_graph, "model/", "pico_FaceVector_model.pb", as_text=False)
   
    
    
    