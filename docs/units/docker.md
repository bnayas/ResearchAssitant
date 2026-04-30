# Unit Guide: `docker`

`docker` contains the container runtime definition used by the simulation launcher path.

## What This Unit Owns

- the launcher runtime image definition,
- the baseline system packages and Python scientific stack used inside that image.

## Main Files

- `project/docker/launcher.Dockerfile` defines the slim Python-based execution image used by the launcher path.

## How It Fits

This directory supports `sim_tool.launcher.py`. It is infrastructure for deterministic execution, not application orchestration logic.

## Change Here When

Edit this directory when you are:

- adding a runtime dependency needed inside the launcher image,
- changing the execution image base,
- fixing reproducibility or startup issues in the container runtime.

## Common Risks

- Adding unnecessary packages that make the runtime heavy or slow to build.
- Changing the image without updating the script that builds it or the launcher expectations.
