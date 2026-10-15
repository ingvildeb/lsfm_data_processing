from pathlib import Path

from atlasspace.config.image_models import ImageConfig
from atlasspace.config.preset_models import RegistrationParametersConfig
from atlasspace.config.space_models import SpaceDefinition
from atlasspace.runtime.registration import RegistrationJob

from lsfm_data_processing.registration_and_transforms.batch_registration.hpc.submission_summary import (
    format_missing_job,
    format_ready_job,
    missing_registration_inputs,
)


def _image(image_id: str, path: Path) -> ImageConfig:
    return ImageConfig(
        image_id=image_id,
        image=path,
        space=SpaceDefinition(
            space_name=image_id,
            orientation="LAS",
            axis_labels=("x", "y", "z"),
            units="voxel",
            resolution_um=(20, 20, 20),
            shape=(10, 10, 10),
        ),
    )


def _job(tmp_path: Path) -> RegistrationJob:
    subject = tmp_path / "subjects" / "IEB0158" / "ch2_native20um_masked.nii.gz"
    template = tmp_path / "templates" / "T_P14_Sytox_20um.nii.gz"
    subject.parent.mkdir(parents=True)
    template.parent.mkdir(parents=True)
    subject.write_bytes(b"subject")
    template.write_bytes(b"template")
    return RegistrationJob(
        fixed_image_config=_image("lsfm_sytox_p14", template),
        moving_image_config=_image("IEB0158", subject),
        output_dir=tmp_path / "output",
        parameters=RegistrationParametersConfig.model_validate(
            {
                "name": "test",
                "preprocessing": {"fixed_padding_um": 500},
                "registration": {
                    "working_resolution_um": 20,
                    "syn_gradient_step": 0.05,
                },
            }
        ),
    )


def test_ready_summary_reports_effective_parameters(tmp_path: Path) -> None:
    job = _job(tmp_path)

    assert missing_registration_inputs(job) == ()
    assert format_ready_job(
        job,
        input_sha256_verified=True,
        mem_gb=128,
    ) == (
        "[ready] IEB0158 input found: ch2_native20um_masked.nii.gz "
        "(SHA256 verified); template=lsfm_sytox_p14; resolution=20 um; "
        "gradient=0.05; padding=500 um; memory=128 GB"
    )


def test_missing_summary_identifies_the_missing_input(tmp_path: Path) -> None:
    job = _job(tmp_path)
    Path(job.moving_image_config.image).unlink()

    missing = missing_registration_inputs(job)

    assert missing == (("subject image", Path(job.moving_image_config.image)),)
    assert format_missing_job(job, missing).startswith(
        "[error] IEB0158 missing subject image:"
    )
