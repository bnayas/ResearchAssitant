# Unit Guide: `scripts`

`scripts` contains operational helpers for the repository.

## What This Unit Owns

- small shell helpers that support development or runtime setup,
- wrapper commands that should stay thinner than the application code they invoke.

## Main Files

- `project/scripts/build_launcher_image.sh` builds the Docker launcher image defined in `project/docker/launcher.Dockerfile`.

## Change Here When

Edit this directory when you are:

- improving a development or packaging helper,
- fixing operational glue around Docker or local setup.

## Boundary Rules

- Business logic should stay in Python packages, not in shell wrappers.
- Scripts should be easy to read, easy to run, and easy to replace if the underlying product flow changes.

## Common Risks

- Hiding important runtime behavior inside shell scripts instead of in the main application code.
- Letting scripts drift away from the actual Dockerfile or package layout.
