import warnings
import pytest


def test_autocast_no_futurewarning():
    """Verify torch.amp.autocast(device_type='cuda') can be used on CUDA without emitting FutureWarning.
    Skips test if CUDA isn't available.
    """
    try:
        import torch
    except Exception:
        pytest.skip('torch not installed')

    if not torch.cuda.is_available():
        pytest.skip('CUDA not available; skipping GPU autocast test')

    # run a small op inside autocast and assert no FutureWarning occurred
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda'):
                a = torch.randn((2, 2), device='cuda')
                b = a * 2.0
                # move result to CPU to ensure ops executed
                _ = b.cpu().numpy()

        # Ensure no FutureWarning in the warnings list
        future_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
        assert len(future_warnings) == 0, f'Found FutureWarnings during autocast: {future_warnings}'
