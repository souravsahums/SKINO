
import re
import os
import tempfile
import runpy

# Full path to your original script
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elm1.py")
for run_id in range(11,21):   # 0 to 10 inclusive
    print(f"\n=== START run_id={run_id} ===")

    # Read original script
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Replace only the run_id entry inside CFG
    code2 = re.sub(
        r'("run_id"\s*:\s*)\d+',
        rf'\g<1>{run_id}',
        code,
        count=1
    )

    # Write to a temporary real .py file so numba caching works
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8"
    ) as tmp:
        tmp.write(code2)
        tmp_path = tmp.name

    try:
        # Run the temporary script as a proper file
        runpy.run_path(tmp_path, run_name="__main__")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"=== END run_id={run_id} ===")
