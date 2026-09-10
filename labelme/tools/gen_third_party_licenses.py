"""Write the license texts of the third-party packages bundled in a build.

GPL/AGPL/LGPL, Apache-2.0 and the BSD/MIT licenses of the bundled packages
all require their license text and copyright notices to ship with binaries.
LabelMe-YOLO-NoONNX.spec calls write_license_texts() after every build. It can
also be run by hand on an existing build (from the labelme/ folder):

    python tools/gen_third_party_licenses.py build/LabelMe-YOLO-NoONNX \\
        dist/labelyolo26/THIRD_PARTY_LICENSES.txt
"""

import datetime
import importlib.metadata
import os
import re
import sys

_LICENSE_FILE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE)", re.IGNORECASE)
_TOC_ENTRY = re.compile(
    r"\('([^']+)',\s*(?:'[^']*'|None),\s*"
    r"'(?:PYMODULE|PYSOURCE|EXTENSION|BINARY|DATA)'\)"
)
_SKIP = {"labelme"}  # the app itself, covered by LICENSE


def _top_level(name):
    return re.split(r"[\\/.]", name, maxsplit=1)[0]


def top_level_names(*tocs):
    """Top-level import/folder names of the entries in PyInstaller TOC lists."""
    return {_top_level(entry[0]) for toc in tocs for entry in toc}


def _names_from_workpath(workpath):
    """Top-level names from the TOC files PyInstaller left in its work dir."""
    names = set()
    for toc_file in ("PYZ-00.toc", "COLLECT-00.toc"):
        with open(os.path.join(workpath, toc_file), encoding="utf-8") as f:
            names.update(_top_level(n) for n in _TOC_ENTRY.findall(f.read()))
    return names


def _license_texts(dist):
    files = [f for f in dist.files or [] if _LICENSE_FILE.match(f.name)]
    # Prefer the license files in the package metadata over copies elsewhere
    in_metadata = [f for f in files if ".dist-info" in str(f) or ".egg-info" in str(f)]
    texts = []
    for f in in_metadata or files:
        path = dist.locate_file(f)
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as fh:
                texts.append(fh.read().strip())
    return texts


def _section(title, license_name, body):
    return f"{'=' * 79}\n{title}\nLicense: {license_name}\n{'-' * 79}\n{body}\n"


def write_license_texts(names, out_path):
    """Write the license texts of the distributions providing `names`.

    Returns the number of packages written.
    """
    mapping = importlib.metadata.packages_distributions()
    dist_names = {d for n in names - _SKIP for d in mapping.get(n, [])}

    sections = []
    for dist_name in sorted(dist_names, key=str.lower):
        dist = importlib.metadata.distribution(dist_name)
        md = dist.metadata
        license_lines = (
            md.get("License-Expression") or md.get("License") or ""
        ).strip().splitlines()
        license_name = license_lines[0][:200] if license_lines else "see below"
        texts = _license_texts(dist)
        if texts:
            body = "\n\n".join(texts)
        elif re.fullmatch(r"GPL[\s-]*v?3.*", license_name, re.IGNORECASE):
            body = "The GPL-3.0 text is in the LICENSE file next to this one."
        else:
            # Prefer the source/homepage link over docs or changelog links
            urls = sorted(
                md.get_all("Project-URL") or [],
                key=lambda u: not re.match(r"(source|repo|home|code)", u, re.I),
            )
            url = md.get("Home-page") or next(
                (u.split(",", 1)[-1].strip() for u in urls), "its project page"
            )
            body = f"(No license file is shipped with this package. See {url}.)"
        sections.append(_section(f"{md['Name']} {md['Version']}", license_name, body))

    for name in ("LICENSE_PYTHON.txt", "LICENSE.txt"):
        path = os.path.join(sys.base_prefix, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                python = f"Python {sys.version.split()[0]}"
                sections.append(_section(python, "PSF-2.0", f.read().strip()))
            break

    header = (
        "LabelYolo26 - third-party license texts\n"
        f"Generated {datetime.date.today():%Y-%m-%d} for the {len(sections)} "
        "packages bundled in this build.\n"
        "LabelYolo26 itself is licensed under GPL-3.0 (see LICENSE); "
        "see THIRD_PARTY_NOTICES.md for a summary.\n\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(sections))
    return len(sections)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    count = write_license_texts(_names_from_workpath(sys.argv[1]), sys.argv[2])
    print(f"Wrote license texts for {count} packages to {sys.argv[2]}")
