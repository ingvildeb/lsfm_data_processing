Repo-top local registration scripts create gitignored `*_local.toml` files in this
folder on first run.

Those local configs are bootstrapped from the canonical templates shipped inside
the installable `lsfm_data_processing` package:

- `lsfm_data_processing/registration_and_transforms/batch_registration/config_templates/batch_register.toml`
- `lsfm_data_processing/registration_and_transforms/sweep_registration/config_templates/sweep_register.toml`

If you delete a local registration config, rerunning the corresponding script will
recreate it here and ask you to edit it before continuing.
