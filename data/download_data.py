from subprocess import check_call
from os import remove
from shutil import unpack_archive
import os


def download(url, directory):
    check_call(["aria2c",
                "--check-certificate=false",
                "--allow-overwrite=true",
                "--seed-time=0",
                "--follow-torrent=mem",
                "-d", directory,
                url])


def download_unpack_delete(url, directory, filename=None):
    if filename is None:
        filename = url.split("/")[-1]
    print(f"Downloading '{filename}'")
    download(url, directory)
    print(f"'{filename}' is downloaded")
    archive_path = os.path.join(directory, filename)
    unpack_archive(archive_path, extract_dir=directory)
    print(f"'{filename}' is unpacked")
    remove(archive_path)
    print(f"'{filename}' is cleaned")


if __name__ == "__main__":
    download_unpack_delete(
        "http://academictorrents.com/download/48e5e770aa8469c0826ae322209cdc0ac115a385.torrent",
        "flyingthings3d",
        "flyingthings3d__frames_finalpass.tar"
    )

    download_unpack_delete(
        "http://academictorrents.com/download/93a54256fe2f56dea2c7d247af11d9affa06a06d.torrent",
        "flyingthings3d",
        "flyingthings3d__optical_flow.tar.bz2"
    )

    download_unpack_delete(
        "https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__motion_boundaries.tar.bz2",
        "flyingthings3d",
    )

    download_unpack_delete(
        "https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_finalpass.tar",
        "monkaa",
    )

    download_unpack_delete(
        "https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__optical_flow.tar.bz2",
        "monkaa",
    )

    download_unpack_delete(
        "https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__motion_boundaries.tar.bz2",
        "monkaa",
    )
