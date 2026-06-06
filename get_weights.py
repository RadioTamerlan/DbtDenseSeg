"""Download the RadDad model weights (area.pt, muscle.pt, dense.pt) into weights/.

Default source: a (private) Hugging Face model repo. Set the repo id and, for a
private repo, a token:

    export DBTDENSESEG_HF_REPO="RadioTamerlan/DbtDenseSeg-weights"
    export HF_TOKEN="hf_xxx"          # only needed if the HF repo is private
    python get_weights.py

You can also just drop the three .pt files into weights/ manually.
"""
import os
import sys

REPO = os.environ.get("DBTDENSESEG_HF_REPO", "RadioTamerlan/DbtDenseSeg-weights")
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
FILES = ["area.pt", "muscle.pt", "dense.pt"]


def main():
    os.makedirs(DEST, exist_ok=True)
    have = [f for f in FILES if os.path.isfile(os.path.join(DEST, f))]
    if len(have) == len(FILES):
        print(f"All weights already present in {DEST}"); return
    if "<your-username>" in REPO:
        sys.exit("Set DBTDENSESEG_HF_REPO to your Hugging Face weights repo "
                 "(e.g. export DBTDENSESEG_HF_REPO='me/DbtDenseSeg-weights'), "
                 "or place area.pt/muscle.pt/dense.pt in weights/ manually.")
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    for f in FILES:
        p = hf_hub_download(repo_id=REPO, filename=f, local_dir=DEST, token=token)
        print("downloaded", p)
    print(f"\nWeights ready in {DEST}")


if __name__ == "__main__":
    main()
