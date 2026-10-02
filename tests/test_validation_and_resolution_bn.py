import torch
from torch import nn

from rrnet.lprm import LPRM, ResolutionBatchNorm2d
from rrnet.validation_sampling import balanced_validation_indices


def test_resolution_statistics_are_independent_and_keep_shared_affine():
    bn = ResolutionBatchNorm2d(nn.BatchNorm2d(3, momentum=1.0))
    bn.train()
    bn.coarse = True
    bn(torch.ones(2, 3, 4, 4) * 2)
    bn.coarse = False
    bn(torch.ones(2, 3, 8, 8) * 7)
    assert torch.allclose(bn.coarse_running_mean, torch.full((3,), 2.0))
    assert torch.allclose(bn.running_mean, torch.full((3,), 7.0))
    assert len(list(bn.parameters())) == 2


def test_legacy_checkpoint_upgrade_preserves_initial_eval_predictions():
    torch.manual_seed(7)
    old = LPRM(shorter_side=64).eval()
    new = LPRM(shorter_side=64, split_resolution_bn=True).eval()
    new.load_state_dict(old.state_dict(), strict=True)
    image = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(old(image)['theta'], new(image)['theta'], atol=1e-6)
    roundtrip = LPRM(shorter_side=64, split_resolution_bn=True).eval()
    roundtrip.load_state_dict(new.state_dict(), strict=True)


def test_validation_covers_people_and_frame_positions():
    people = ['a'] * 100 + ['b'] * 100 + ['c'] * 100
    indices = balanced_validation_indices(people, 1, 30)
    assert len(indices) == len(set(indices)) == 30
    assert [people[i] for i in indices[:3]] == ['a', 'b', 'c']
    assert sum(people[i] == 'a' for i in indices) == 10
    assert max(indices) > 280


def test_small_validation_budget_has_unique_valid_indices():
    indices = balanced_validation_indices(['a', 'b', 'c'], 2, 2)
    assert len(indices) == len(set(indices)) == 2
    assert all(0 <= i < 6 for i in indices)
