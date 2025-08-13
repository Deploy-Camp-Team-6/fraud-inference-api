import subprocess
import sys
from pathlib import Path


def get_exported_deps() -> set[str]:
    """Exports production dependencies from poetry.lock."""
    try:
        # We use --without-hashes because the hashes can change based on the platform.
        # We only care about the package versions.
        # We use --only main to get only production dependencies.
        result = subprocess.run(
            ["poetry", "export", "--without-hashes", "--only", "main"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Normalize by stripping whitespace and filtering out empty lines
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running 'poetry export': {e}", file=sys.stderr)
        # Handle cases where poetry might not have the export command or flags
        # Fallback for older poetry versions
        print("Falling back to 'poetry run pip freeze' method...", file=sys.stderr)
        result = subprocess.run(
            ["poetry", "run", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        # This fallback is less precise as it might include dev-related transitive deps
        # but it's better than failing completely.
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def get_constraints_file_deps() -> set[str]:
    """Reads dependencies from the constraints-serve.txt file."""
    constraints_path = Path("constraints-serve.txt")
    if not constraints_path.exists():
        print(f"Error: '{constraints_path}' not found!", file=sys.stderr)
        return set()

    with open(constraints_path) as f:
        return {line.strip() for line in f if line.strip()}


def main():
    """
    Compares the locked production dependencies with the constraints file
    and exits with a non-zero status code if they have drifted.
    """
    print("Checking for drift between poetry.lock and constraints-serve.txt...")

    exported_deps = get_exported_deps()
    constraints_deps = get_constraints_file_deps()

    if not exported_deps or not constraints_deps:
        print("Could not retrieve dependencies to compare. Aborting.", file=sys.stderr)
        sys.exit(1)

    if exported_deps == constraints_deps:
        print("✅ Dependencies are in sync.")
        sys.exit(0)
    else:
        print(
            "❌ Error: Drift detected between locked dependencies and constraints file!",
            file=sys.stderr,
        )

        in_lock_not_constraints = exported_deps - constraints_deps
        if in_lock_not_constraints:
            print(
                "\nDependencies in poetry.lock but not in constraints-serve.txt:",
                file=sys.stderr,
            )
            for dep in sorted(in_lock_not_constraints):
                print(f"  + {dep}", file=sys.stderr)

        in_constraints_not_lock = constraints_deps - exported_deps
        if in_constraints_not_lock:
            print(
                "\nDependencies in constraints-serve.txt but not in poetry.lock:",
                file=sys.stderr,
            )
            for dep in sorted(in_constraints_not_lock):
                print(f"  - {dep}", file=sys.stderr)

        print(
            "\nPlease regenerate the constraints file by running the appropriate command.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
