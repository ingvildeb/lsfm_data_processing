from __future__ import annotations

from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

try:
    from atlasspace import ImageConfig, SpaceDefinition, registration, transforms
except ImportError as exc:
    if exc.name == "atlasspace":
        raise ImportError(
            "This script requires atlasspace to be importable. Install it first, "
            "for example with `pip install -e path/to/atlasspace`."
        ) from exc
    raise ImportError(
        "This script could import atlasspace itself, but one of atlasspace's own "
        f"dependencies is missing: {exc.name}. Install that dependency in the active environment "
        "and try again."
    ) from exc

from lsfm_data_processing.utils.io_helpers import (
    load_script_config,
    normalize_user_path,
    require_dir,
    require_file,
)
from lsfm_data_processing.utils.naming import get_underscore_token


# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(
    Path(__file__),
    "1_batch_register",
    test_mode=test_mode,
)


# -------------------------
# CONFIG PARAMETERS
# -------------------------
run_cfg = cfg["run"]
subject_defaults_cfg = cfg["subject_defaults"]
segmentation_transform_cfg = cfg.get("segmentation_transform", {})
templates_cfg = cfg["templates"]
folder_to_template_cfg = cfg["folder_to_template"]

registration_preset = run_cfg["registration_preset"]
template_role = run_cfg.get("template_role", "moving")
output_subdir = run_cfg.get("output_subdir", "_01_registration")
orientation_alignment = run_cfg.get("orientation_alignment", "none")
write_input_images = run_cfg.get("write_input_images", False)

subject_image_name = subject_defaults_cfg["image_name"]
subject_orientation = subject_defaults_cfg["orientation"]
subject_resolution_um = float(subject_defaults_cfg["resolution_um"])
underscores_to_id = subject_defaults_cfg["underscores_to_id"]

transform_segmentations = segmentation_transform_cfg.get("enabled", False)
segmentation_output_subdir = segmentation_transform_cfg.get(
    "output_subdir",
    "_02_template_segmentations",
)
segmentation_interpolation = segmentation_transform_cfg.get(
    "interpolation",
    "nearestNeighbor",
)
write_transform_intermediates = segmentation_transform_cfg.get(
    "write_intermediates",
    False,
)


# -------------------------
# HELPERS
# -------------------------
def _make_space_definition(
    *,
    space_name: str,
    orientation: str,
    resolution_um: float,
) -> SpaceDefinition:
    return SpaceDefinition(
        space_name=space_name,
        orientation=orientation,
        resolution_um=(resolution_um, resolution_um, resolution_um),
    )


def _parse_subject_id(subject_folder: Path, token_index: int) -> str:
    return get_underscore_token(subject_folder.name, token_index, "subject_id")


def _build_subject_image_config(subject_folder: Path) -> ImageConfig:
    subject_id = _parse_subject_id(subject_folder, underscores_to_id)
    subject_image_path = require_file(
        subject_folder / subject_image_name,
        f"Subject image for {subject_id}",
    )

    return ImageConfig(
        image_id=subject_id,
        image=subject_image_path,
        space=_make_space_definition(
            space_name=subject_id,
            orientation=subject_orientation,
            resolution_um=subject_resolution_um,
        ),
    )


def _build_template_image_config(template_key: str) -> ImageConfig:
    template_cfg = templates_cfg[template_key]
    template_space_name = template_cfg.get("space_name", template_key)
    template_resolution_um = float(template_cfg["resolution_um"])
    template_image_path = require_file(
        normalize_user_path(template_cfg["image"]),
        f"Template image for {template_key}",
    )

    return ImageConfig(
        image_id=template_key,
        image=template_image_path,
        space=_make_space_definition(
            space_name=template_space_name,
            orientation=template_cfg["orientation"],
            resolution_um=template_resolution_um,
        ),
    )


def _build_segmentation_image_config(
    template_key: str,
    segmentation_name: str,
    segmentation_path_value: str,
) -> ImageConfig:
    template_cfg = templates_cfg[template_key]
    template_space_name = template_cfg.get("space_name", template_key)
    template_resolution_um = float(template_cfg["resolution_um"])
    segmentation_path = require_file(
        normalize_user_path(segmentation_path_value),
        f"Template segmentation '{segmentation_name}' for {template_key}",
    )

    return ImageConfig(
        image_id=segmentation_name,
        image=segmentation_path,
        space=_make_space_definition(
            space_name=template_space_name,
            orientation=template_cfg["orientation"],
            resolution_um=template_resolution_um,
        ),
    )


def _build_registration_job(
    *,
    subject_folder: Path,
    subject_config: ImageConfig,
    template_config: ImageConfig,
    parameters,
) -> registration.RegistrationJob:
    output_dir = subject_folder / output_subdir

    if template_role == "moving":
        fixed_image_config = subject_config
        moving_image_config = template_config
    elif template_role == "fixed":
        fixed_image_config = template_config
        moving_image_config = subject_config
    else:
        raise RuntimeError(
            f"template_role must be 'moving' or 'fixed'. Got: {template_role}"
        )

    return registration.RegistrationJob(
        fixed_image_config=fixed_image_config,
        moving_image_config=moving_image_config,
        output_dir=output_dir,
        parameters=parameters,
        orientation_alignment=orientation_alignment,
    )


def _segmentation_transform_direction() -> str:
    if template_role == "moving":
        return "forward"
    if template_role == "fixed":
        return "inverse"
    raise RuntimeError(f"Unsupported template_role: {template_role}")


def _segmentation_output_name(segmentation_name: str) -> str:
    direction = _segmentation_transform_direction()
    suffix = "WarpedSegmentation" if direction == "forward" else "InverseWarpedSegmentation"
    return f"{segmentation_name}_{suffix}.nii.gz"


def _apply_template_segmentations(
    *,
    subject_folder: Path,
    subject_config: ImageConfig,
    template_key: str,
    result,
) -> None:
    template_cfg = templates_cfg[template_key]
    segmentations_cfg = template_cfg.get("segmentations", {})

    if not segmentations_cfg:
        print(f"  No template segmentations configured for {template_key}. Skipping segmentation transforms.")
        return

    output_dir = subject_folder / segmentation_output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    transform_sequence = transforms.TransformSequence.from_registration_result(result)
    direction = _segmentation_transform_direction()

    for segmentation_name, segmentation_path_value in segmentations_cfg.items():
        segmentation_config = _build_segmentation_image_config(
            template_key,
            segmentation_name,
            segmentation_path_value,
        )
        output_path = output_dir / _segmentation_output_name(segmentation_name)
        transforms.transform_segmentation(
            segmentation_config,
            transform_sequence,
            subject_config,
            direction=direction,
            interpolation=segmentation_interpolation,
            output_path=output_path,
            write_intermediates=write_transform_intermediates,
        )
        print(f"  Wrote transformed segmentation: {output_path.name}")


# -------------------------
# MAIN CODE
# -------------------------
parameters = registration.load_preset(registration_preset)
parameters.execution.write_input_images = write_input_images

print(f"Using registration preset: {registration_preset}")

successful_subjects: list[str] = []
failed_subjects: list[str] = []

for subject_folder_value, template_key in folder_to_template_cfg.items():
    subject_folder = require_dir(
        normalize_user_path(subject_folder_value),
        "Subject folder",
    )

    if template_key not in templates_cfg:
        raise RuntimeError(
            f"Template key '{template_key}' is not defined under [templates]."
        )

    subject_id = _parse_subject_id(subject_folder, underscores_to_id)
    print(f"Running registration for {subject_id} using template '{template_key}' ...")

    try:
        subject_config = _build_subject_image_config(subject_folder)
        template_config = _build_template_image_config(template_key)
        job = _build_registration_job(
            subject_folder=subject_folder,
            subject_config=subject_config,
            template_config=template_config,
            parameters=parameters,
        )

        result = registration.run_antspy_registration(job)

        if not result.success:
            print(f"  Registration failed for {subject_id}: {result.error_message}")
            failed_subjects.append(subject_id)
            continue

        print(f"  Registration finished: {job.output_dir}")

        if transform_segmentations:
            _apply_template_segmentations(
                subject_folder=subject_folder,
                subject_config=subject_config,
                template_key=template_key,
                result=result,
            )

        successful_subjects.append(subject_id)
    except Exception as exc:
        print(f"  Failed while processing {subject_id}: {exc}")
        failed_subjects.append(subject_id)

print()
print(f"Finished batch registration. Successes: {len(successful_subjects)}")
if successful_subjects:
    print("Successful subjects:")
    for subject_id in successful_subjects:
        print(f"  - {subject_id}")

print(f"Failures: {len(failed_subjects)}")
if failed_subjects:
    print("Failed subjects:")
    for subject_id in failed_subjects:
        print(f"  - {subject_id}")
