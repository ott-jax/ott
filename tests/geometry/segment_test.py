# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax.numpy as jnp
import numpy as np

from ott.geometry import segment


class TestSegmentPointCloud:

  def test_default_weights_unsorted_segment_ids(self):
    x = jnp.arange(6.0).reshape(3, 2)
    segment_ids = jnp.array([1, 0, 0])

    _, a, num_per_segment = segment.segment_point_cloud(
        x, segment_ids=segment_ids, num_segments=2, max_measure_size=2
    )

    np.testing.assert_array_equal(num_per_segment, [2, 1])
    np.testing.assert_allclose(a, [[0.5, 0.5], [1.0, 0.0]])
