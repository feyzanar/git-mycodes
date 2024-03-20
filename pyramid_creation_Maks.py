#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:09:59 2020

@author: Maks Hess
"""

from pathlib import Path
import numpy as np
import h5py
from abbott.h5_files import (
    h5_rename_dataset,
    h5_copy_attributes,
    h5_write_attributes,
    h5_select,
    h5_set,
)
from abbott.conversions import to_itk, to_numpy
from abbott.itk_image import image_pyramid
import sys
import argparse
import re


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument('idx', type=int)
    CLI.add_argument('fld', type=str)
    CLI.add_argument('--n_levels', type=int, default=5)
    args = CLI.parse_args()

    fns = sorted(list(Path(args.fld).glob('*.h5')))
    fn = fns[args.idx]

    print(f"Processing {fn}...")

    # Detect datasets that are not yet converted to pyramids of the requested size or larger.
    dset_names = []
    with h5py.File(fn) as f:
        for potential_dset in h5_select(f, {'level': min(h5_set(f, 'level'))}):
            stain = potential_dset.attrs['stain']
            cycle = potential_dset.attrs['cycle']
            if len(h5_select(f, {'stain': stain, 'cycle': cycle})) < args.n_levels:
                dset_names.append(potential_dset.name)

    # with h5py.File(fn) as f:
    #     for k in f.keys():
    #         if isinstance(f[k], h5py.Dataset):
    #             dset_names.append(k)

    for dset_name in dset_names:
        print(f"Creating image pyramid for {dset_name}...")
        create_image_pyramid_channel(fn, dset_name, levels=args.n_levels)


def create_image_pyramid_channel(fn, channel_name, levels=3):
    with h5py.File(fn, 'r+') as f:
        dset = f[channel_name]
        base_level = dset.attrs.get('level', 0)
        img = to_itk(dset)
        pyr = image_pyramid(img, n=levels - 1)

        # Handle old convention where channels would be written as datasets directly into the root.
        if mo := re.search(r'(.*)/\d{1,2}', dset.name):
            new_name_fstring = mo.group(1) + '/{}'
        else:
            new_name_fstring = dset.name + '/{}'
            new_name = new_name_fstring.format(base_level)
            h5_rename_dataset(f, dset.name, new_name)
            h5_write_attributes(f[new_name], {'level': base_level})

        for i, im in enumerate(pyr, start=1):
            data = to_numpy(im)
            new_name = new_name_fstring.format(base_level + i)
            if new_name in f.keys():
                del f[new_name]
            new_dset = f.create_dataset(
                data=data,
                name=new_name,
                compression="gzip",
                chunks=True,
            )
            h5_copy_attributes(dset, new_dset)
            h5_write_attributes(
                new_dset,
                {
                    "element_size_um": np.array(im.GetSpacing(), dtype=np.float64)[
                                       ::-1
                                       ],
                    "level": base_level + i,
                },
                overwrite=True,
            )


def create_image_pyramid_legacy(fn, levels=3):
    print("Processing {}...".format(fn.name))
    with h5py.File(fn, "r+") as f:
        base_level = min(h5_set(f, 'level'))
        for dset in h5_select(f, {"img_type": "intensity", "level": base_level}):
            if dset.parent.name == '/':
                print("Processing {}...".format(dset.name))
                img = to_itk(dset)
                new_name = dset.name.replace("channel", "ch") + "/{}"
                new_dset = h5_rename_dataset(f, dset.name, (new_name.format(0)))
                h5_write_attributes(new_dset, {"level": 0})
                pyr = image_pyramid(img, n=levels - 1)
                for i, im in enumerate(pyr, start=1):
                    data = to_numpy(im)
                    new_dset = f.create_dataset(
                        data=data,
                        name=new_name.format(i),
                        compression="gzip",
                        chunks=(1, data.shape[1], data.shape[2]),
                    )
                    h5_copy_attributes(dset, new_dset, exclude=("element_size_um",))
                    h5_write_attributes(
                        new_dset,
                        {
                            "element_size_um": np.array(im.GetSpacing(), dtype=np.float64)[
                                               ::-1
                                               ],
                            "level": base_level + i,
                        },
                        overwrite=True,
                    )


if __name__ == "__main__":
    main()
