# Registration Presets

Registration presets are YAML files that define the preprocessing, registration, and execution settings used by the
`atlasspace` registration runner. Built-in presets currently live in `atlasspace`, while project-specific presets can
live beside your registration project configs.

## Starting Point

For a project-specific preset, start by copying a built-in preset into your project:

```bash
mkdir -p /path/to/your_registration_project/configs/registration_presets

python -c "from importlib.resources import files; from shutil import copyfile; copyfile(files('atlasspace.presets.registration').joinpath('tuned_syn_cc.yaml'), '/path/to/your_registration_project/configs/registration_presets/my_tuned_syn.yaml')"
```

Replace the project path with your own registration project directory. Then reference the custom YAML path in
`configs/batch_register.toml` or `configs/sweep_register.toml`:

```toml
[run]
registration_presets = ["/path/to/your_registration_project/configs/registration_presets/my_tuned_syn.yaml"]
```

Built-in preset names such as `"baseline_syn_kimlab"` and `"tuned_syn_cc"` can be used directly. Full file paths are
recommended for custom presets, especially on HPC.

## Minimal Preset Template

```yaml
name: my_tuned_syn
description: Short description of what changed and why.

preprocessing:
  intensity_normalization: null
  minmax_clip_percentiles: null
  histogram_match: false
  gaussian_sigma_vox: null

registration:
  working_resolution_um: 20
  transform_type: SyN

  aff_metric: mattes
  aff_sampling: 32
  aff_random_sampling_rate: 0.25
  aff_iterations: [1000, 1000, 1000]
  aff_shrink_factors: [12, 8, 4]
  aff_smoothing_sigmas: [4, 3, 2]

  syn_metric: mattes
  syn_sampling: 32
  syn_gradient_step: 0.1
  syn_flow_sigma: 3.0
  syn_total_sigma: 0.0
  syn_reg_iterations: [1000, 1000, 1000]

execution:
  threads: 32
  singleprecision: true
  use_legacy_histogram_matching: false
  verbose: true
  random_seed: 0
  write_input_images: false
```

## Parameter Reference

Baseline settings below are from the built-in `baseline_syn_kimlab` preset, which is the package-facing version of the
historical SyN_JP/Kim lab baseline.

<table>
  <thead>
    <tr>
      <th>Parameter family</th>
      <th>Parameter</th>
      <th>Baseline setting (<code>baseline_syn_kimlab</code>)</th>
      <th>What it controls / means</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Preset identity</td>
      <td><code>name</code></td>
      <td><code>baseline_syn_kimlab</code></td>
      <td>Internal preset name written into summaries and used to label outputs. Use a unique value for custom presets.</td>
    </tr>
    <tr>
      <td><code>description</code></td>
      <td><code>Historical Kim lab SyN baseline benchmark at 20 um.</code></td>
      <td>Human-readable note about the preset.</td>
    </tr>
    <tr>
      <td rowspan="4">Input preprocessing</td>
      <td><code>preprocessing.intensity_normalization</code></td>
      <td><code>null</code></td>
      <td>Optional intensity normalization before registration. Accepted values are <code>null</code>, <code>zscore</code>, and <code>robust_zscore</code>.</td>
    </tr>
    <tr>
      <td><code>preprocessing.minmax_clip_percentiles</code></td>
      <td><code>null</code></td>
      <td>Optional intensity clipping/rescaling before registration, for example <code>[1.0, 99.0]</code>.</td>
    </tr>
    <tr>
      <td><code>preprocessing.histogram_match</code></td>
      <td><code>false</code></td>
      <td>Histogram-match the moving image to the fixed image before registration preprocessing.</td>
    </tr>
    <tr>
      <td><code>preprocessing.gaussian_sigma_vox</code></td>
      <td><code>null</code></td>
      <td>Optional Gaussian smoothing before registration, in voxel units.</td>
    </tr>
    <tr>
      <td>Registration scale</td>
      <td><code>registration.working_resolution_um</code></td>
      <td><code>20</code></td>
      <td>Resolution, in microns, used for the registration working images. Native-resolution outputs are still written after transforms are applied.</td>
    </tr>
    <tr>
      <td>Transform model</td>
      <td><code>registration.transform_type</code></td>
      <td><code>SyN</code></td>
      <td>ANTs transform family. Accepted values are <code>Rigid</code>, <code>Affine</code>, <code>SyN</code>, and <code>SyNOnly</code>.</td>
    </tr>
    <tr>
      <td rowspan="3">Affine matching behavior</td>
      <td><code>registration.aff_metric</code></td>
      <td><code>mattes</code></td>
      <td>Similarity metric for the affine stage. Currently accepted value is <code>mattes</code>.</td>
    </tr>
    <tr>
      <td><code>registration.aff_sampling</code></td>
      <td><code>32</code></td>
      <td>Sampling/bin parameter passed to the affine metric. For Mattes mutual information, this is the number of histogram bins.</td>
    </tr>
    <tr>
      <td><code>registration.aff_random_sampling_rate</code></td>
      <td><code>0.25</code></td>
      <td>Fraction of voxels sampled for the affine metric. Lower values can run faster but may be less stable.</td>
    </tr>
    <tr>
      <td rowspan="3">Affine multiresolution schedule</td>
      <td><code>registration.aff_iterations</code></td>
      <td><code>[1000, 1000, 1000]</code></td>
      <td>Maximum affine iterations at each resolution level.</td>
    </tr>
    <tr>
      <td><code>registration.aff_shrink_factors</code></td>
      <td><code>[12, 8, 4]</code></td>
      <td>Downsampling factors for affine levels. Larger values make coarser early levels.</td>
    </tr>
    <tr>
      <td><code>registration.aff_smoothing_sigmas</code></td>
      <td><code>[4, 3, 2]</code></td>
      <td>Smoothing sigmas for affine levels. Values correspond to the affine multiresolution levels.</td>
    </tr>
    <tr>
      <td rowspan="2">Deformable matching behavior</td>
      <td><code>registration.syn_metric</code></td>
      <td><code>mattes</code></td>
      <td>Similarity metric for the deformable SyN stage. Accepted values are <code>mattes</code> and <code>CC</code>.</td>
    </tr>
    <tr>
      <td><code>registration.syn_sampling</code></td>
      <td><code>32</code></td>
      <td>Sampling parameter for the SyN metric. For <code>mattes</code>, this is bins; for <code>CC</code>, this is the neighborhood radius.</td>
    </tr>
    <tr>
      <td>Deformation aggressiveness</td>
      <td><code>registration.syn_gradient_step</code></td>
      <td><code>0.1</code></td>
      <td>SyN update step size. This corresponds to ANTsPy <code>grad_step</code>. Lower values are more conservative.</td>
    </tr>
    <tr>
      <td rowspan="2">Deformation regularization</td>
      <td><code>registration.syn_flow_sigma</code></td>
      <td><code>3.0</code></td>
      <td>Smoothing applied to the update field. This corresponds to ANTsPy <code>flow_sigma</code>. Higher values smooth local updates more.</td>
    </tr>
    <tr>
      <td><code>registration.syn_total_sigma</code></td>
      <td><code>0.0</code></td>
      <td>Smoothing applied to the total deformation field. This corresponds to ANTsPy <code>total_sigma</code>.</td>
    </tr>
    <tr>
      <td>SyN multiresolution schedule</td>
      <td><code>registration.syn_reg_iterations</code></td>
      <td><code>[1000, 1000, 1000]</code></td>
      <td>SyN iterations at each resolution level. This corresponds to ANTsPy <code>reg_iterations</code>.</td>
    </tr>
    <tr>
      <td rowspan="6">Execution</td>
      <td><code>execution.threads</code></td>
      <td><code>32</code></td>
      <td>Number of ITK/ANTs threads to use. Set to <code>0</code> for library default behavior.</td>
    </tr>
    <tr>
      <td><code>execution.singleprecision</code></td>
      <td><code>true</code></td>
      <td>Use single-precision registration. Usually faster and lower-memory than double precision.</td>
    </tr>
    <tr>
      <td><code>execution.use_legacy_histogram_matching</code></td>
      <td><code>false</code></td>
      <td>ANTsPy legacy histogram matching flag. Keep <code>false</code> unless reproducing older behavior.</td>
    </tr>
    <tr>
      <td><code>execution.verbose</code></td>
      <td><code>true</code></td>
      <td>Print verbose registration progress. Useful for logs.</td>
    </tr>
    <tr>
      <td><code>execution.random_seed</code></td>
      <td><code>0</code></td>
      <td>Random seed used by ANTsPy where supported. Use <code>null</code> to leave unset.</td>
    </tr>
    <tr>
      <td><code>execution.write_input_images</code></td>
      <td><code>false</code></td>
      <td>Write fixed/moving registration input images for inspection/debugging.</td>
    </tr>
  </tbody>
</table>

## Common Tuning Patterns

- **Try a more conservative SyN deformation:** lower `syn_gradient_step`, increase `syn_flow_sigma`, or reduce late-level `syn_reg_iterations`.
- **Try a CC-based deformable metric:** set `syn_metric: CC` and use a small `syn_sampling` value such as `2`. This is the main difference in `tuned_syn_cc`.
- **Screen faster before long runs:** shorten `syn_reg_iterations` first, then rerun promising settings at full length.
- **Avoid changing many knobs at once:** change one family at a time so you can tell whether affine behavior, deformable behavior, or the multiresolution schedule caused the difference.

## Notes

- Presets define method parameters only. Image/template pairing, template role, and output location belong in the registration TOML, not in the preset YAML.
- The old shorthand names `grad_step`, `flow_sigma`, `total_sigma`, and `reg_iterations` are ANTsPy names. In these presets, use `syn_gradient_step`, `syn_flow_sigma`, `syn_total_sigma`, and `syn_reg_iterations`.
- If a custom preset is used on HPC, keep it inside the project folder or use an absolute path that exists on the cluster.
