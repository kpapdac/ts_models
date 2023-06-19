import math
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.examples.bart import load_bart_od
from pyro.contrib.forecast import ForecastingModel, Forecaster, eval_crps
from pyro.infer.reparam import LocScaleReparam, SymmetricStableReparam
from pyro.ops.tensor_utils import periodic_repeat

class linearModel(ForecastingModel):
    # We then implement the .model() method. Since this is a generative model, it shouldn't
    # look at data; however it is convenient to see the shape of data we're supposed to
    # generate, so this inputs a zeros_like(data) tensor instead of the actual data.
    def model(self, zero_data, covariates):
        data_dim = zero_data.size(-1)  # Should be 1 in this univariate tutorial.
        feature_dim = covariates.size(-1)

        # The first part of the model is a probabilistic program to create a prediction.
        # We use the zero_data as a template for the shape of the prediction.
        bias = pyro.sample("bias", dist.Normal(0, 10).expand([data_dim]).to_event(1))
        weight = pyro.sample("weight", dist.Normal(0, 0.1).expand([feature_dim]).to_event(1))
        prediction = bias + (weight * covariates).sum(-1, keepdim=True)
        # The prediction should have the same shape as zero_data (duration, obs_dim),
        # but may have additional sample dimensions on the left.
        assert prediction.shape[-2:] == zero_data.shape

        # The next part of the model creates a likelihood or noise distribution.
        # Again we'll be Bayesian and write this as a probabilistic program with
        # priors over parameters.
        noise_scale = pyro.sample("noise_scale", dist.LogNormal(-5, 5).expand([1]).to_event(1))
        noise_dist = dist.Normal(0, noise_scale)

        # The final step is to call the .predict() method.
        self.predict(noise_dist, prediction)

class levyStable(ForecastingModel):
    def model(self, zero_data, covariates):
        duration, data_dim = zero_data.shape

        # Let's model each time series as a Levy stable process, and share process parameters
        # across time series. To do that in Pyro, we'll declare the shared random variables
        # outside of the "origin" plate:
        drift_stability = pyro.sample("drift_stability", dist.Uniform(1, 2))
        drift_scale = pyro.sample("drift_scale", dist.LogNormal(-20, 5))
        with pyro.plate("origin", data_dim, dim=-2):
            # Now inside of the origin plate we sample drift and seasonal components.
            # All the time series inside the "origin" plate are independent,
            # given the drift parameters above.
            with self.time_plate:
                # We combine two different reparameterizers: the inner SymmetricStableReparam
                # is needed for the Stable site, and the outer LocScaleReparam is optional but
                # appears to improve inference.
                with poutine.reparam(config={"drift": LocScaleReparam()}):
                    with poutine.reparam(config={"drift": SymmetricStableReparam()}):
                        drift = pyro.sample("drift",
                                            dist.Stable(drift_stability, 0, drift_scale))

            with pyro.plate("hour_of_week", 24 * 7, dim=-1):
                seasonal = pyro.sample("seasonal", dist.Normal(0, 5))

        # Now outside of the time plate we can perform time-dependent operations like
        # integrating over time. This allows us to create a motion with slow drift.
        seasonal = periodic_repeat(seasonal, duration, dim=-1)
        motion = drift.cumsum(dim=-1)  # A Levy stable motion to model shocks.
        prediction = motion + seasonal

        # Next we do some reshaping. Pyro's forecasting framework assumes all data is
        # multivariate of shape (duration, data_dim), but the above code uses an "origins"
        # plate that is left of the time_plate. Our prediction starts off with shape
        assert prediction.shape[-2:] == (data_dim, duration)
        # We need to swap those dimensions but keep the -2 dimension intact, in case Pyro
        # adds sample dimensions to the left of that.
        prediction = prediction.unsqueeze(-1).transpose(-1, -3)
        assert prediction.shape[-3:] == (1, duration, data_dim), prediction.shape

        # Finally we can construct a noise distribution.
        # We will share parameters across all time series.
        obs_scale = pyro.sample("obs_scale", dist.LogNormal(-5, 5))
        noise_dist = dist.Normal(0, obs_scale.unsqueeze(-1))
        self.predict(noise_dist, prediction)

class deepHr(ForecastingModel):
    def model(self, zero_data, covariates):
        num_stations, num_stations, duration, one = zero_data.shape

        # We construct plates once so we can reuse them later. We ensure they don't collide by
        # specifying different dim args for each: -3, -2, -1. Note the time_plate is dim=-1.
        origin_plate = pyro.plate("origin", num_stations, dim=-3)
        destin_plate = pyro.plate("destin", num_stations, dim=-2)
        hour_of_week_plate = pyro.plate("hour_of_week", 24 * 7, dim=-1)

        # Let's model the time-dependent part with only O(num_stations * duration) many
        # parameters, rather than the full possible O(num_stations ** 2 * duration) data size.
        drift_stability = pyro.sample("drift_stability", dist.Uniform(1, 2))
        drift_scale = pyro.sample("drift_scale", dist.LogNormal(-20, 5))
        with origin_plate:
            with hour_of_week_plate:
                origin_seasonal = pyro.sample("origin_seasonal", dist.Normal(0, 5))
        with destin_plate:
            with hour_of_week_plate:
                destin_seasonal = pyro.sample("destin_seasonal", dist.Normal(0, 5))
            with self.time_plate:
                with poutine.reparam(config={"drift": LocScaleReparam()}):
                    with poutine.reparam(config={"drift": SymmetricStableReparam()}):
                        drift = pyro.sample("drift",
                                            dist.Stable(drift_stability, 0, drift_scale))
        # Additionally we can model a static pairwise station->station affinity, which e.g.
        # can compensate for the fact that people tend not to travel from a station to itself.
        with origin_plate, destin_plate:
            pairwise = pyro.sample("pairwise", dist.Normal(0, 1))

        # Outside of the time plate we can now form the prediction.
        seasonal = origin_seasonal + destin_seasonal  # Note this broadcasts.
        seasonal = periodic_repeat(seasonal, duration, dim=-1)
        motion = drift.cumsum(dim=-1)  # A Levy stable motion to model shocks.
        prediction = motion + seasonal + pairwise

        # We will decompose the noise scale parameter into
        # an origin-local and a destination-local component.
        with origin_plate:
            origin_scale = pyro.sample("origin_scale", dist.LogNormal(-5, 5))
        with destin_plate:
            destin_scale = pyro.sample("destin_scale", dist.LogNormal(-5, 5))
        scale = origin_scale + destin_scale

        # At this point our prediction and scale have shape (50, 50, duration) and (50, 50, 1)
        # respectively, but we want them to have shape (50, 50, duration, 1) to satisfy the
        # Forecaster requirements.
        scale = scale.unsqueeze(-1)
        prediction = prediction.unsqueeze(-1)

        # Finally we construct a noise distribution and call the .predict() method.
        # Note that predict must be called inside the origin and destination plates.
        noise_dist = dist.Normal(0, scale)
        with origin_plate, destin_plate:
            self.predict(noise_dist, prediction)