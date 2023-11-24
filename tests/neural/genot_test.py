import jax
import jax.numpy as jnp
import optax

from ott.neural.models.models import NeuralVectorField
from ott.neural.solvers.genot import GENOT
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein


class TestGENOT:

  def test_genot_linear_unconditional(self, genot_data_loader_linear):
    neural_vf = NeuralVectorField(
        output_dim=2,
        condition_dim=0,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=2,
        output_dim=2,
        cond_dim=0,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        optimizer=optimizer
    )
    genot(genot_data_loader_linear, genot_data_loader_linear)

    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_linear
    )
    result_forward = genot.transport(
        source_lin, condition=condition, forward=True
    )
    assert isinstance(result_forward, jax.Array)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  def test_genot_quad_unconditional(self, genot_data_loader_quad):
    neural_vf = NeuralVectorField(
        output_dim=2,
        condition_dim=0,
        latent_embed_dim=5,
    )
    ot_solver = gromov_wasserstein.GromovWasserstein(epsilon=1e-2)
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=1,
        output_dim=2,
        cond_dim=0,
        epsilon=None,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        optimizer=optimizer
    )
    genot(genot_data_loader_quad, genot_data_loader_quad)

    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_quad
    )
    result_forward = genot.transport(
        source_quad, condition=condition, forward=True
    )
    assert isinstance(result_forward, jax.Array)
    assert jnp.sum(jnp.isnan(result_forward)) == 0

  def test_genot_linear_conditional(self, genot_data_loader_linear_conditional):
    neural_vf = NeuralVectorField(
        output_dim=2,
        condition_dim=4,
        latent_embed_dim=5,
    )
    ot_solver = sinkhorn.Sinkhorn()
    optimizer = optax.adam(learning_rate=1e-3)
    genot = GENOT(
        neural_vf,
        input_dim=2,
        output_dim=2,
        cond_dim=4,
        iterations=3,
        valid_freq=2,
        ot_solver=ot_solver,
        optimizer=optimizer
    )
    genot(
        genot_data_loader_linear_conditional,
        genot_data_loader_linear_conditional
    )

    source_lin, source_quad, target_lin, target_quad, condition = next(
        genot_data_loader_linear_conditional
    )
    result_forward = genot.transport(
        source_lin, condition=condition, forward=True
    )
    assert isinstance(result_forward, jax.Array)
    assert jnp.sum(jnp.isnan(result_forward)) == 0
