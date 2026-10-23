"""Merge data/claims.proposed.*.json into data/claims.json.

Usage: python tools/merge_claims.py [--apply]

Proposed claims are never allowed to overwrite an existing id; conflicts are reported. Without
--apply it only prints what would change. Delete the proposal files after merging.
"""
import glob
import json
import os
import sys

root = os.path.join(os.path.dirname(__file__), "..", "data")
main_path = os.path.join(root, "claims.json")
main = json.load(open(main_path, encoding="utf-8"))
added, conflicts = {}, []
for path in sorted(glob.glob(os.path.join(root, "claims.proposed.*.json"))):
    prop = json.load(open(path, encoding="utf-8"))
    prop = prop.get("claims", prop)
    for cid, rec in prop.items():
        if cid in main["claims"] or cid in added:
            conflicts.append((cid, os.path.basename(path)))
        else:
            added[cid] = rec
print(f"{len(added)} new claim(s): {', '.join(added) or 'none'}")
for cid, src in conflicts:
    print(f"CONFLICT (kept existing): {cid} from {src}")
if "--apply" in sys.argv and added:
    main["claims"].update(added)
    json.dump(main, open(main_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("applied")
