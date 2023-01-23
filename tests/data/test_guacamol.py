import pytest

from pcmr.data.guacamol import GuacaMolDataModule


@pytest.fixture(params=["loGP", "AmloDipINE_mPo", "meDiaN 1"])
def dataset(request):
    return request.param.lower()


@pytest.fixture(params=[0, 1, 2])
def seed(request):
    return request.param


@pytest.mark.parametrize("dataset", ['LOGP', 'AMLODIPINE_MPO', 'SCAFFOLD HOP', 'MEDIAN 1', 'RANOLAZINE_MPO', 'ARIPIPRAZOLE_SIMILARITY', 'FEXOFENADINE_MPO', 'ZALEPLON_MPO', 'VALSARTAN_SMARTS', 'CELECOXIB_REDISCOVERY', 'PERINDOPRIL_MPO', 'OSIMERTINIB_MPO', 'QED'])
def test_dataset_in(dataset):
    assert dataset.upper() in GuacaMolDataModule.datasets


def test_dataset_no_tasks(dataset):
    assert GuacaMolDataModule.get_tasks(dataset) == list()


@pytest.mark.parametrize("dataset", ["foo", "bar", "scaffold_hop"])
def test_invalid_dataset_tasks(dataset):
    with pytest.raises(ValueError):
        GuacaMolDataModule.get_tasks(dataset)


@pytest.mark.parametrize("task", ["foo", "bar", "baz"])
def test_get_all_data_invalid_task(dataset, task):
    with pytest.raises(ValueError):
        GuacaMolDataModule.get_all_data(dataset, task)


def test_get_all_data_same_seed(dataset):
    df1 = GuacaMolDataModule.get_all_data(dataset, None)
    df2 = GuacaMolDataModule.get_all_data(dataset, None)

    assert df1.equals(df2)


@pytest.mark.parametrize("dataset", ["logp"])
def test_get_all_data_reseed(dataset, seed):
    df1 = GuacaMolDataModule.get_all_data(dataset, None)
    GuacaMolDataModule.seed(seed)
    df2 = GuacaMolDataModule.get_all_data(dataset, None)

    assert not df1.equals(df2)