import numpy as np
import pytest
import torch

from rrnet.lighting import ParameterDenormalizer, default_parameter_statistics, unpack_lights
from rrnet.lprm import LPRM
from rrnet.physical_reference_data import (
    MEADPhysicalReferencePairs, REFERENCE_LIGHTING_MODES, _stable_seed,
    sample_physical_theta, sample_reference_theta, sample_diverse_source_thetas)


def statistics(tmp_path):
    mean, std = default_parameter_statistics(9)
    path = tmp_path / 'stats.npz'
    np.savez(path, mean=mean.numpy(), std=std.numpy())
    return str(path)


def test_affine_exact_and_state_compatibility():
    old = ParameterDenormalizer(9)
    new = ParameterDenormalizer(9, parameterization='affine')
    new.load_state_dict(old.state_dict(), strict=True)
    z = torch.randn(4, 93)
    assert torch.equal(new(z), old.std * z + old.mean)
    assert set(new.state_dict()) == {'mean', 'std'}


def test_smooth_ranges_center_and_gradients(tmp_path):
    decoder = ParameterDenormalizer(9, statistics(tmp_path), 'smooth_physical')
    center = decoder(torch.zeros(1, 93))
    assert torch.allclose(center, decoder.mean, atol=2e-6)
    z = torch.randn(8, 93, requires_grad=True)
    result = decoder(z)
    fields = unpack_lights(result, 9)
    for name in ('color', 'attenuation', 'ambient'):
        assert (fields[name] > 0).all()
    assert ((fields['position'] > 0) & (fields['position'] < 1)).all()
    assert torch.allclose(fields['direction'].norm(dim=-1), torch.ones(8, 9), atol=1e-6)
    result.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert (z.grad.reshape(8, -1)[:, :3] > 0).all()
    assert torch.isfinite(decoder(torch.full((1, 93), 100.))).all()


def test_statistics_required_and_validated(tmp_path):
    with pytest.raises(ValueError):
        ParameterDenormalizer(9, parameterization='smooth_physical')
    path = tmp_path / 'invalid.npz'
    np.savez(path, mean=np.full(93, np.nan), std=np.ones(93))
    with pytest.raises(ValueError):
        ParameterDenormalizer(9, str(path))


def test_lprm_returns_physical_normalized_theta(tmp_path):
    model = LPRM(shorter_side=64, statistics_path=statistics(tmp_path),
                 light_parameterization='smooth_physical').eval()
    with torch.no_grad():
        prediction = model(torch.rand(2, 3, 64, 64))
    reconstructed = prediction['theta_normalized'] * model.denormalize.std + model.denormalize.mean
    assert torch.allclose(reconstructed, prediction['theta'], atol=1e-6)


@pytest.mark.parametrize('group_size,fraction', [(1, 0), (4, 0), (4, .2), (4, 1)])
def test_extracted_sampler_preserves_original_random_sequence(group_size, fraction):
    ds = MEADPhysicalReferencePairs.__new__(MEADPhysicalReferencePairs)
    ds.samples = [{}] * 6
    ds.people = ['A', 'A', 'B', 'B', 'C', 'C']
    ds.seed, ds.split, ds.epoch = 20260929, 'train', 2
    ds.num_lights, ds.variants_per_source = 9, 4
    ds.group_size, ds.reference_sensitivity_fraction = group_size, fraction
    ds.reference_lighting_probabilities = np.array([.5, .5, 0])
    for index in range(24):
        source_index, variant = divmod(index, 4)
        rng = np.random.default_rng(_stable_seed(ds.seed, ds.split, ds.epoch, source_index, variant))
        ref_index = ds._different_person(source_index, rng)
        mode = str(rng.choice(REFERENCE_LIGHTING_MODES, p=ds.reference_lighting_probabilities))
        if group_size == 1:
            sources = [sample_physical_theta(rng, 9)]
            references = [sample_reference_theta(rng, 9, mode)]
            modes, sensitivity = [mode], False
        else:
            sensitivity = bool(rng.random() < fraction)
            if sensitivity:
                sources = [sample_physical_theta(rng, 9)] * 4
                modes = ['dark', 'normal', 'dark', 'normal']
                rng.shuffle(modes)
                references = [sample_reference_theta(rng, 9, m) for m in modes]
            else:
                sources = sample_diverse_source_thetas(rng, 9, 4)
                references = [sample_reference_theta(rng, 9, mode)] * 4
                modes = [mode] * 4
        result = ds.sample_parameters(index)
        assert result['reference_index'] == ref_index
        assert result['uniform_group'] == (not sensitivity)
        assert tuple(modes) == result['reference_lighting_mode']
        assert torch.equal(torch.stack(sources), result['source_theta_target'])
        assert torch.equal(torch.stack(references), result['reference_theta_target'])
