import optax

from ott.neural.flow_matching import FlowMatching
from ott.neural.flows import ConstantNoiseFlow
from ott.neural.models import NeuralVectorField
from ott.solvers.linear import sinkhorn


class TestFlowMatching:

  def test_flow_matching(self, data_loader_gaussian_1, data_loader_gaussian_2):
    neural_vf = NeuralVectorField(
        input_dim=2, hidden_dims=[32, 32], output_dim=2, activation="relu"
    )
    ot_solver = sinkhorn.SinkhornSolver()
    flow = ConstantNoiseFlow(sigma=0)
    optimizer = optax.adam(learning_rate=1e-3)
    fm = FlowMatching(
        neural_vf,
        input_dim=2,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        flow=flow,
        optimizer=optimizer
    )
    fm(data_loader_gaussian_1, data_loader_gaussian_2)
