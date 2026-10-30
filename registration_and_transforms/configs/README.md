Repo-top local registration scripts create gitignored `*_local.toml` files in this
folder on first run.

Those local configs are bootstrapped from canonical package templates:

- `atlasspace/src/atlasspace/config_templates/registration_batch_template.toml`
- `atlasspace/src/atlasspace/config_templates/registration_sweep_template.toml`

If you delete a local registration config, rerunning the corresponding script will
recreate it here and ask you to edit it before continuing.
