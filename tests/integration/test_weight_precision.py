"""Run in the SkyRL environment; ordinary lightweight checks skip this module."""

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('skyrl')


def test_meta_conversion_preserves_tied_parameters_and_snapshot():
    from simple_scibuddy.training.weights import preserve_conversion_aliases

    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(8, 4)
    model.head = torch.nn.Linear(4, 8, bias=False)
    model.head.weight = model.embedding.weight
    snapshot = model.state_dict()
    with preserve_conversion_aliases(model):
        model.to('meta')
        assert model.head.weight is model.embedding.weight
        assert model.head.weight.is_meta
        model.load_state_dict(snapshot, assign=True)
        assert model.head.weight is model.embedding.weight
        assert not model.head.weight.is_meta
    assert 'to' not in model.__dict__
    assert 'load_state_dict' not in model.__dict__
    assert not snapshot['head.weight'].is_meta
    assert torch.equal(snapshot['head.weight'], snapshot['embedding.weight'])


def test_receiver_fingerprint_detects_parameter_and_buffer_changes():
    from simple_scibuddy.inference.weight_audit import fingerprint

    model = torch.nn.Linear(2, 2)
    model.register_buffer('counter', torch.tensor(1))
    before = fingerprint(model)
    assert fingerprint(model) == before
    with torch.no_grad():
        model.weight[0, 0] += 1
        model.counter += 1
    after = fingerprint(model)
    assert before['parameters']['weight']['sha256'] != after['parameters']['weight']['sha256']
    assert before['parameters']['bias'] == after['parameters']['bias']
    assert before['buffers']['counter']['sha256'] != after['buffers']['counter']['sha256']

@pytest.mark.parametrize('bucketing', [False, True])
@pytest.mark.parametrize('prefix', ['', 'language_model.'])
def test_recurrent_precision_survives_extraction(bucketing, prefix):
    from simple_scibuddy.training.weights import RecurrentWeightExtractor

    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.register_parameter('A_log', torch.nn.Parameter(torch.tensor([0.123456, 0.234567])))
    model.layer.proj = torch.nn.Linear(2, 2, bias=False)
    extractor = RecurrentWeightExtractor(model, enable_bucketing=bucketing, weight_prefix=prefix)
    # Exercise the real upstream extraction/grouping on CPU; distributed gathering
    # is verified separately by the explicit GPU integration run.
    extractor._gather_tensor = lambda tensor: tensor
    chunks = list(extractor.extract_weights(torch.bfloat16))
    values = {name: tensor for chunk in chunks
              for name, tensor in zip(chunk.names, chunk.tensors, strict=True)}
    assert values[prefix + 'layer.A_log'].dtype == torch.float32
    assert torch.equal(values[prefix + 'layer.A_log'], model.layer.A_log)
    assert values[prefix + 'layer.proj.weight'].dtype == torch.bfloat16
    assert torch.equal(values[prefix + 'layer.proj.weight'], model.layer.proj.weight.bfloat16())
    metadata = extractor.get_weight_metadata(torch.bfloat16)
    assert dict(zip(metadata['names'], metadata['dtype_names'], strict=True)) == {
        name: str(value.dtype).removeprefix('torch.') for name, value in values.items()
    }
    for chunk in chunks:
        assert chunk.dtypes == [str(t.dtype) for t in chunk.tensors]
        assert chunk.total_size_bytes == sum(t.numel() * t.element_size() for t in chunk.tensors)
