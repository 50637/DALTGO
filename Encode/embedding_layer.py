# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf  # pylint: disable=g-bad-import-order

#from official.transformer.model import model_utils
#from official.r1.utils import tpu as tpu_utils


class EmbeddingSharedWeights(tf.compat.v1.layers.Layer):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size, method="gather"):
    super(EmbeddingSharedWeights, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    if method not in ("gather", "matmul"):
      raise ValueError("method {} must be 'gather' or 'matmul'".format(method))
    self.method = method

  def build(self, _):
    with tf.compat.v1.variable_scope("embedding_and_softmax", reuse=tf.compat.v1.AUTO_REUSE):
      # Create and initialize weights. The random normal initializer was chosen
      # randomly, and works well.
      self.shared_weights = tf.compat.v1.get_variable(
          "weights", [self.vocab_size, self.hidden_size],
          initializer=tf.random_normal_initializer(
              0., self.hidden_size ** -0.5))

    self.built = True

  def call(self, x):
    """Get token embeddings of x using word2vec.

    Args:
      x: An int64 tensor with shape [batch_size, length]
    Returns:
      embeddings: float32 tensor with shape [batch_size, length, embedding_size]
      padding: float32 tensor with shape [batch_size, length] indicating the
        locations of the padding tokens in x.
    """
    with tf.name_scope("embedding"):
      from gensim.models.word2vec import Word2Vec
      from gensim.models import KeyedVectors
      model = KeyedVectors.load_word2vec_format('mf80.txt', binary=False)
      vocab = list(model.vocab.keys())
      embedding_matrix = model.vectors
      embedding_var = tf.Variable(embedding_matrix)
      embedding_lookup = tf.nn.embedding_lookup(embedding_var, [i for i in range(len(vocab))])
      embeddings = tf.nn.embedding_lookup(embedding_lookup, x)
      embeddings *= self.hidden_size ** 0.5
      return embeddings


  def linear(self, x):
    """Computes logits by running x through a linear layer.

    Args:
      x: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    with tf.name_scope("presoftmax_linear"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      x = tf.reshape(x, [-1, self.hidden_size])
      logits = tf.matmul(x, self.shared_weights, transpose_b=True)

      return tf.reshape(logits, [batch_size, length, self.vocab_size])
