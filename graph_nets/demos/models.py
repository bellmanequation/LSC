# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt

NUM_LAYERS = 2  # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 16  # Hard-code latent layer sizes for demos.


def make_mlp_model():
    return snt.Sequential([
        snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
        snt.LayerNorm()
    ])


def make_edge_model():
    return snt.Sequential([
        snt.nets.MLP([64, 32, 3], activate_final=True),
    ])


def make_node_model():
    return snt.Sequential([
        snt.nets.MLP([256], activate_final=True),
        snt.LayerNorm()
    ])


def make_Hnode_model():
    return snt.Sequential([
        snt.nets.MLP([64], activate_final=True),
        snt.LayerNorm()
    ])


def make_Lnode_model():
    return snt.Sequential([
        snt.nets.MLP([64, 32], activate_final=True),
        snt.LayerNorm()
    ])


def make_conv_model():
    return snt.Sequential([
        snt.nets.ConvNet2D(output_channels=[32, 32], kernel_shapes=[3, 3],
                           strides=[1, 1], paddings=['VALID', 'VALID'], activate_final=True),
        snt.BatchFlatten(),
        snt.nets.MLP([256], activate_final=True),
    ])


def get_q_model():
    return snt.Sequential([
        snt.nets.MLP([128, 64, 13], activate_final=False)
    ])


class GCrpNetworkTiny(snt.AbstractModule):
    def __init__(self, name="GCrpNetworkTiny"):
        super(GCrpNetworkTiny, self).__init__(name=name)
        with self._enter_variable_scope():
            self._obsEncoder = modules.obsEncoder(encoder_fn=make_conv_model)
            self._network = modules.CommNet(
                edge_model_fn=make_edge_model,
                node_model_fn=make_node_model)

            self._hnetwork = modules.HCommNet(
                edge_model_fn=make_edge_model,
                node_model_fn=make_Hnode_model)

            self._Lnetwork = modules.LCommNet(
                edge_model_fn=make_edge_model,
                node_model_fn=make_Lnode_model)

            self._qnet = modules.qEncoder(mlp_fn=get_q_model)

    def _build(self, inputs):
        return self._qnet(self._Lnetwork(
            self._hnetwork(self._network(self._obsEncoder(inputs)))))
