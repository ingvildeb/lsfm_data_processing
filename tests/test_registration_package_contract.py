from importlib.resources import files

from lsfm_data_processing import registration_and_transforms
from lsfm_data_processing.registration_and_transforms._batch_register_core import (
    load_batch_register_settings_from_path,
)
from lsfm_data_processing.registration_and_transforms._sweep_register_core import (
    load_sweep_register_settings_from_path,
)


def test_registration_core_entry_points_import() -> None:
    assert callable(load_batch_register_settings_from_path)
    assert callable(load_sweep_register_settings_from_path)
    assert registration_and_transforms.batch_registration
    assert registration_and_transforms.sweep_registration


def test_hpc_resources_are_packaged() -> None:
    batch_hpc = files(
        "lsfm_data_processing.registration_and_transforms.batch_registration"
    )
    sweep_hpc = files(
        "lsfm_data_processing.registration_and_transforms.sweep_registration"
    )

    assert batch_hpc.joinpath("config_templates", "hpc.toml").is_file()
    assert batch_hpc.joinpath("hpc", "run_one_batch_job.sh").is_file()
    assert sweep_hpc.joinpath("config_templates", "hpc.toml").is_file()
    assert sweep_hpc.joinpath("hpc", "run_one_sweep_job.sh").is_file()
