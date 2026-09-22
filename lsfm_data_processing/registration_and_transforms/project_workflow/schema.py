"""Canonical workbook table names and columns for project registration workflows."""

RUN_EVALUATIONS_SHEET = "Run evaluations"
RESCUE_REQUESTS_SHEET = "Rescue requests"
CATEGORIES_SHEET = "Categories"
WORKFLOW_LISTS_SHEET = "_Registration workflow lists"
LEGACY_RESCUE_COLUMN = "Proposed rescue"

RUN_EVALUATION_COLUMNS = (
    "Subject ID",
    "Run path",
    "Preset name",
    "Manifest status",
    "Result",
    "Comments",
    "Selected",
    "Evaluator",
    "Evaluation date",
)

RESCUE_REQUEST_COLUMNS = (
    "Subject ID",
    "Rescue strategy",
    "Source run",
    "Variant name",
    "Gradient step",
    "Working resolution (um)",
    "Padding (um)",
    "Notes",
    "Status",
)

RESULT_CATEGORIES = {
    "excellent": "No meaningful registration inaccuracy identified.",
    "minor_localized": "Small localized inaccuracy that may be acceptable.",
    "overwarping": "Anatomy is deformed beyond the expected subject-template difference.",
    "underwarping": "The transform does not sufficiently match the local anatomy.",
    "large_inaccuracy": "A larger registration inaccuracy requiring intervention.",
    "exclude": "The subject or run should not be used.",
}
