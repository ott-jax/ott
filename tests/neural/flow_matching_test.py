import optax

from ott.neural.models.models import NeuralVectorField
from ott.neural.solvers.flow_matching import FlowMatching
from ott.neural.solvers.flows import ConstantNoiseFlow
from ott.solvers.linear import sinkhorn


class TestFlowMatching:

  def test_flow_matching(self, data_loader_gaussian):
    neural_vf = NeuralVectorField(
        output_dim=2,
        condition_dim=0,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    flow = ConstantNoiseFlow(sigma=0)
    optimizer = optax.adam(learning_rate=1e-3)
    fm = FlowMatching(
        neural_vf,
        input_dim=2,
        cond_dim=0,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        flow=flow,
        optimizer=optimizer
    )
    fm(data_loader_gaussian, data_loader_gaussian)
