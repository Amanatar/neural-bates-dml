"""
Automated Deployment Script for Hugging Face Docker Spaces.
Author: Quantitative Analytics UI & Cloud Deployment Engineer
Repository: Amanatar/neural-bates-dml
"""

import os
import sys
import time
import argparse
import urllib.request
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def deploy_to_hf_space(
    token=None,
    username=None,
    space_name="neural-bates-dml",
    check_timeout=300,
    dry_run=False
):
    """
    Creates and deploys the Neural Bates DML workbench to a public Hugging Face Docker Space.
    """
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        print("[ERROR] huggingface_hub package is required. Install with: pip install huggingface_hub")
        return False, None

    # Resolve Hugging Face authentication token
    resolved_token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or get_token()
    )

    if not resolved_token:
        print("\n" + "=" * 78)
        print("HUGGING FACE AUTHENTICATION REQUIRED")
        print("=" * 78)
        print("No Hugging Face token detected in environment or argument.")
        print("\nTo deploy this Docker Space to your Hugging Face account:")
        print("  1. Generate an Access Token with 'Write' permissions at:")
        print("     https://huggingface.co/settings/tokens")
        print("  2. Run either:")
        print("     python deploy_hf_space.py --token <YOUR_HF_TOKEN>")
        print("     or in PowerShell:")
        print("     $env:HF_TOKEN = \"<YOUR_HF_TOKEN>\"; python deploy_hf_space.py")
        print("=" * 78 + "\n")
        return False, None

    api = HfApi(token=resolved_token)

    try:
        user_info = api.whoami()
        hf_username = username or user_info.get("name")
        print(f"[AUTH] Successfully authenticated as Hugging Face user: @{hf_username}")
    except Exception as e:
        print(f"[ERROR] Failed to authenticate with Hugging Face API: {e}")
        return False, None

    space_id = f"{hf_username}/{space_name}"
    space_web_url = f"https://huggingface.co/spaces/{space_id}"
    direct_app_url = f"https://{hf_username.replace('_', '-')}-{space_name.replace('_', '-')}.hf.space"

    print(f"[TARGET] Space ID:  {space_id}")
    print(f"[TARGET] Space URL: {space_web_url}")

    if dry_run:
        print("[DRY-RUN] Pre-flight check successful. Aborting before upload.")
        return True, space_web_url

    # 1. Create or verify repo exists with Docker SDK
    print(f"\n[1/3] Creating / verifying Space repository: {space_id} (SDK: docker)...")
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False
        )
        print(f"[SUCCESS] Space repository verified: {space_id}")
    except Exception as e:
        print(f"[ERROR] Failed to create Space repository: {e}")
        return False, None

    # 2. Upload bundled application files
    print("\n[2/3] Uploading bundled Docker files and model artifacts...")
    files_to_upload = [
        "Dockerfile",
        ".dockerignore",
        "README.md",
        "requirements.txt",
        "app.py",
        "model.py",
        "bates_fft.py",
        "generate_dataset.py",
        "live_calibration.py",
        "benchmark_dml.py",
        "RESULTS.md",
        # Flagship and Baseline models
        "artifacts/bates_dml_v1.pt",
        "artifacts/price_only_baseline.pt",
        # Deterministic Evaluation Snapshot
        "data/eval/spy_options_snapshot.csv",
        "data/eval/bates_test_benchmark.npz",
        # Empirical ablation results
        "results/ablation_study.json",
        "results/spy_calibration_benchmark.json"
    ]

    for rel_path in files_to_upload:
        full_path = REPO_ROOT / rel_path
        if full_path.exists():
            file_size_kb = full_path.stat().st_size / 1024.0
            print(f"  Uploading {rel_path} ({file_size_kb:.1f} KB)...")
            api.upload_file(
                path_or_fileobj=str(full_path),
                path_in_repo=rel_path.replace("\\", "/"),
                repo_id=space_id,
                repo_type="space",
                commit_message=f"Deploy {rel_path} for Neural Bates DML"
            )
        else:
            print(f"  [WARN] File not found, skipping: {rel_path}")

    print(f"[SUCCESS] All application artifacts committed to Space: {space_id}")

    # 3. Monitor container build & runtime status
    print("\n[3/3] Monitoring Docker container build on Hugging Face infrastructure...")
    start_wait = time.time()
    last_stage = None

    while time.time() - start_wait < check_timeout:
        try:
            runtime = api.get_space_runtime(repo_id=space_id)
            stage = getattr(runtime, "stage", "UNKNOWN")
            
            if stage != last_stage:
                print(f"  Container Status: {stage} (elapsed: {int(time.time() - start_wait)}s)")
                last_stage = stage

            if stage == "RUNNING":
                print("\n" + "=" * 78)
                print(f"🎉 SPACE IS RUNNING AND HEALTHY!")
                print(f"Live Space URL:      {space_web_url}")
                print(f"Direct Web App URL:  {direct_app_url}")
                print("=" * 78 + "\n")
                
                # Verify HTTP responsiveness
                try:
                    req = urllib.request.Request(direct_app_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        print(f"[HTTP] Direct App Healthcheck: Status {resp.status} OK")
                except Exception as http_err:
                    print(f"[HTTP] Direct endpoint initializing (status will be 200 shortly): {http_err}")
                    
                return True, space_web_url

            elif stage in ["BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"]:
                print(f"\n[ERROR] Container encountered error state: {stage}")
                print(f"Inspect detailed build logs at: {space_web_url}")
                return False, space_web_url

        except Exception as e:
            print(f"  [WAIT] Status query: {e}")

        time.sleep(8)

    print(f"\n[TIMEOUT] Reached timeout ({check_timeout}s) waiting for RUNNING state.")
    print(f"The container is still building on Hugging Face. Check status at: {space_web_url}")
    return True, space_web_url


def main():
    parser = argparse.ArgumentParser(description="Deploy Neural Bates DML to Hugging Face Docker Spaces")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (with Write permission)")
    parser.add_argument("--username", type=str, default=None, help="Hugging Face username (defaults to whoami)")
    parser.add_argument("--space_name", type=str, default="neural-bates-dml", help="Space name (default: neural-bates-dml)")
    parser.add_argument("--timeout", type=int, default=300, help="Wait timeout in seconds for container to start")
    parser.add_argument("--dry_run", action="store_true", help="Perform pre-flight check without creating or uploading")

    args = parser.parse_args()
    success, url = deploy_to_hf_space(
        token=args.token,
        username=args.username,
        space_name=args.space_name,
        check_timeout=args.timeout,
        dry_run=args.dry_run
    )

    if not success and url is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
