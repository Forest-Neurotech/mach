# Contributing to mach

You can contribute in many ways:

# Types of Contributions

## Request features or report bugs

Open an issue at https://github.com/Forest-Neurotech/mach/issues and fill out the requested information.

If a feature or bug already exists, please upvote it! We will prioritize based on user input.

## Implement features or fix bugs

[Any issue](https://github.com/Forest-Neurotech/mach/issues) tagged with `help wanted` is open to contributors.

Check that it is not already assigned in the project tracker: https://github.com/orgs/Forest-Neurotech/projects/14

## Write Documentation

mach could always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

# Get Started!

Ready to contribute? Here's how to set up `mach` for local development.
Please note this documentation assumes you already have `uv` and `Git` installed and ready to go.

1. Fork the `mach` repo on GitHub.

2. Clone your fork locally:

```bash
git clone https://github.com/Forest-Neurotech/mach.git
```

3. Now we need to install the environment. Navigate into the directory

```bash
cd mach
```

If you don't have `uv` installed yet, install with:

```bash
make install-uv
```

Then, install and activate the environment with:

```bash
uv sync
```

4. Install pre-commit to run linters/formatters at commit time:

```bash
uv run pre-commit install
```

5. Create a branch for local development:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

6. Don't forget to add test cases for your added functionality to the `tests` directory.

7. When you're done making changes, check that your changes pass the formatting tests:

```bash
make check
```

And validate that all unit tests are passing:

```bash
make test
```

8. Then submit a pull request on GitHub.

# Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.

2. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring, and add the feature to the list in `README.md`.
