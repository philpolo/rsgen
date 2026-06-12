#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:27:06 2026

@author: polo
"""

import os
import sys
import pickle as pkl
import numpy as np
import tensorflow as tf
from time import time

sys.path.append("..")
sys.path.append("../Communication")

import roadscene2vec
from roadscene2vec.data.dataset import SceneGraphDataset
from roadscene2vec.scene_graph.extraction import image_extractor as RealEx

from Communication.e2emodel import MIMOE2EModel
from sgautoencoder.sg_autoencoder import sg_autoencoder

sys.modules["util"] = roadscene2vec.util


class GBSED:
    def __init__(self, config, com_model_endpoint, batch_size=16, ebno=tf.constant(20, tf.float32)):
        self.config = config
        self.data = self._load_()
        self.batch_size = batch_size
        self.ebno = ebno
        self.sg_ae = sg_autoencoder(self.config)
        self._load_communicator_(com_model_endpoint)

    def _set_ebno(self, ebno):
        self.ebno = ebno

    def _load_communicator_(self, endpoint):
        self.model = MIMOE2EModel()
        self.model(1, tf.constant(0.0, tf.float32))
        self.model_weights_path = endpoint

        with open(self.model_weights_path, "rb") as f:
            weights = pkl.load(f)

        for i, w in enumerate(weights):
            self.model.neural_rx.weights[i].assign(w)

    def _load_(self):
        scenegraph_dataset = SceneGraphDataset()
        scenegraph_dataset.dataset_save_path = self.config.location_data["input_path"]

        if self.config.loading_type == "pickle":
            return scenegraph_dataset.load()

        elif self.config.loading_type == "folder":
            sg_extraction_object = RealEx.RealExtractor(self.config)
            sg_extraction_object.load()
            return sg_extraction_object.getDataSet()

    def _format_storage_(self, labels, feature_nodes, L, comp_T):
        to_serialize = []
        to_serialize.append(len(labels))
        to_serialize.extend(labels)

        to_serialize.append(feature_nodes.size)
        to_serialize.extend(feature_nodes.ravel())

        to_serialize.append(L.size)
        to_serialize.extend(L)

        to_serialize.append(comp_T.size)
        to_serialize.extend(comp_T.ravel())

        return np.asarray(to_serialize, dtype=np.float16)

    def _format_loading_(self, to_read):
        cur_idx = 0

        nb = int(to_read[cur_idx])
        cur_idx += 1
        labels = to_read[cur_idx:cur_idx + nb]
        cur_idx += nb

        nb = int(to_read[cur_idx])
        cur_idx += 1
        features = to_read[cur_idx:cur_idx + nb]
        cur_idx += nb

        nb = int(to_read[cur_idx])
        cur_idx += 1
        L = to_read[cur_idx:cur_idx + nb]
        cur_idx += nb

        nb = int(to_read[cur_idx])
        cur_idx += 1
        comp_T = to_read[cur_idx:]

        labels = [int(i) for i in labels]
        feature_nodes = features.reshape((len(labels), -1))
        L = [int(i) for i in L]
        comp_T = comp_T.reshape((len(L), len(labels), len(labels)))

        return labels, feature_nodes, L, comp_T

    def _prepare_bits_for_model_(self, input_bits_1d):
        model_k = self.model.k.numpy() if hasattr(self.model.k, "numpy") else self.model.k

        expected_bits = (
            self.batch_size
            * self.model.num_ut
            * self.model.num_tx_ant
            * model_k
        )

        if len(input_bits_1d) > expected_bits:
            prepared_bits = input_bits_1d[:expected_bits]
        elif len(input_bits_1d) < expected_bits:
            padding_size = expected_bits - len(input_bits_1d)
            prepared_bits = np.pad(input_bits_1d, (0, padding_size), constant_values=0)
        else:
            prepared_bits = input_bits_1d

        return tf.constant(prepared_bits, dtype=tf.int32)

    def _recover_original_bits_(self, decoded_bits, original_length):
        decoded_bits_flat = tf.reshape(decoded_bits, [-1])
        return decoded_bits_flat[:original_length]

    def _to_bits_array_(self, np_array):
        b = np_array.tobytes()
        return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

    def _to_float_array_(self, bits):
        b = np.packbits(bits)
        return np.frombuffer(b.tobytes(), np.float16)

    def _process_sg_(self, sg):
        labels, feat_nodes_mat, T = self.sg_ae.encode(sg)
        comp_T, L = self.sg_ae.sem_compression(T)

        to_serialize = self._format_storage_(labels, feat_nodes_mat, L, comp_T)
        input_bits = self._to_bits_array_(to_serialize)

        return {
            "labels": labels,
            "feature_nodes_matrix": feat_nodes_mat,
            "compressed_Tensor": comp_T,
            "indexes": L,
            "input_bits": input_bits,
            "prepared_bits": self._prepare_bits_for_model_(input_bits),
        }

    def _sg_reconstruction_(self, received_bits, initial_processed_sg):
        to_read = self._to_float_array_(received_bits)

        labels = initial_processed_sg["labels"]
        comp_T = initial_processed_sg["compressed_Tensor"]
        feat_nodes_mat = initial_processed_sg["feature_nodes_matrix"]
        L = initial_processed_sg["indexes"]

        try:
            rec_labels, rec_feature_nodes, rec_L, rec_comp_T = self._format_loading_(to_read)

            if (
                np.allclose(comp_T, rec_comp_T)
                and np.allclose(labels, rec_labels)
                and np.allclose(feat_nodes_mat, rec_feature_nodes)
                and np.allclose(L, rec_L)
            ):
                return self.sg_ae.decode(rec_labels, rec_feature_nodes, rec_L, rec_comp_T)

        except Exception:
            print("\nTruncated file", file=sys.stderr)

    def sg_transmission(self, sg):
        initial_processed_sg = self._process_sg_(sg)

        start = time()
        b, b_hat = self.model(
            self.batch_size,
            self.ebno,
            initial_processed_sg["prepared_bits"],
        )
        end = time()

        received_bits = self._recover_original_bits_(
            b_hat,
            initial_processed_sg["input_bits"].size,
        ).numpy()

        received_bits = np.asarray(received_bits, dtype=np.uint8)

        return self._sg_reconstruction_(received_bits, initial_processed_sg), end - start