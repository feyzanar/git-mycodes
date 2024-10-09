import argparse
import sys
from pathlib import Path
import h5py
from itertools import product
from abbott.conversions import *
from abbott.h5_files import h5_select, h5_copy_attributes, h5_write_attributes
from abbott.np_image import background_seeded_watershed, resample_to_shape
from abbott.itk_image import median, apply_image_filter, to_contours, smooth
from skimage.segmentation import watershed
from skimage.morphology import remove_small_holes
from abbott.itk_image import bw_closing


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument('idx', type=int)
    CLI.add_argument('fld', type=str)
    CLI.add_argument('--membrane_stain', type=str, default='bCatenin')
    CLI.add_argument('--seed_structure', type=str, default='nuclei')
    CLI.add_argument('--fg_structure', type=str, default='embryo')
    CLI.add_argument('--filter_radii', type=int, nargs='+', default=[2])
    CLI.add_argument('--compactnesses', type=float, nargs='+', default=[0.15])
    CLI.add_argument('--dilations', type=int, nargs='+', default=[0])
    args = CLI.parse_args()

    fns = sorted(list(Path(args.fld).glob('*.h5')))
    fn = fns[args.idx]

    print(fn)

    segment_cells_skimage(
        fn,
        membrane_stain=args.membrane_stain,
        membrane_level="same",
        seed_structure=args.seed_structure,
        fg_structure=args.fg_structure,
        filter_radii=args.filter_radii,
        compactnesses=args.compactnesses,
        dilations=args.dilations,
        add_membranes_and_cytoplasm=True,
    )


def cleanup_binary_mask(mask: h5py.Dataset, radius=5, area_threshold=5_000_000):
    itk_img = bw_closing(to_itk(mask), radius=radius)
    out = remove_small_holes(to_numpy(itk_img), area_threshold)


def segment_cells_skimage(
        fn,
        membrane_stain="bCatenin",
        membrane_level="same",
        seed_structure="nuclei",
        fg_structure="embryo",
        filter_radii=(2,),
        compactnesses=(0.1,),
        dilations=(0,),
        add_membranes_and_cytoplasm=True,
):
    with h5py.File(fn, "a") as f:
        print("loading images...")
        nuclei_dset = h5_select(f, {"stain": seed_structure})[0]
        nuclei_raw = to_itk(nuclei_dset)

        fg_dset = h5_select(f, {"stain": fg_structure})[0]
        fg = to_itk(fg_dset)
        # fg_dil = to_itk(apply_image_filter(to_labelmap(fg), "dilation", 3))
        fg_dil = fg

        if membrane_level == "same":
            membrane_level = nuclei_dset.attrs.get("level")
        membranes_dset = h5_select(
            f, {"stain": membrane_stain, "level": membrane_level}
        )[0]
        membranes = to_itk(membranes_dset)

        for dilation in dilations:
            print("dilating nuclei...")
            if dilation == 0:
                nuclei = nuclei_raw
            elif dilation > 0:
                nuclei = to_itk(apply_image_filter(to_labelmap(nuclei_raw), "dilation", dilation))
            else:
                nuclei = to_itk(apply_image_filter(to_labelmap(nuclei_raw), "erosion", abs(dilation)))

            for filter_radius in filter_radii:
                print("filtering membrane image...")
                membranes_filt = to_numpy(median(membranes, radius=filter_radius))

                print("upsampling labelimages...")
                nuclei = resample_to_shape(to_numpy(nuclei), membranes_filt.shape, order=0)
                fg_dil = resample_to_shape(to_numpy(fg_dil), membranes_filt.shape, order=0)

                for compactness in compactnesses:
                    print("applying background seeded watershed...")
                    cells = watershed(
                        image=membranes_filt,
                        markers=nuclei,
                        mask=fg_dil,
                        compactness=compactness,
                    ).astype(np.uint16)

                    print("writing results...")
                    out_dset_name = "lbl_cells"
                    if len(filter_radii) > 1:
                        out_dset_name += '_f{}'.format(filter_radius)
                    if len(compactnesses) > 1:
                        out_dset_name += '_c{}'.format(compactness)
                    if len(dilations) > 1:
                        out_dset_name += '_d{}'.format(dilation)
                    if membrane_level != nuclei_dset.attrs.get('level'):
                        out_dset_name += '_m{}'.format(membrane_level)
                    if out_dset_name in f:
                        del f[out_dset_name]
                    out_dset = f.create_dataset(out_dset_name, data=cells, compression="gzip", )
                    h5_copy_attributes(
                        nuclei_dset, out_dset, exclude=("stain", "element_size_um", "level")
                    )
                    h5_copy_attributes(
                        membranes_dset, out_dset, include=("element_size_um", "level")
                    )
                    h5_write_attributes(
                        out_dset, {"stain": "cells"}
                    )
                    if add_membranes_and_cytoplasm:
                        print("calculating membranes and cytoplasm...")
                        lbl_membranes = to_numpy(to_contours(to_itk(cells)))
                        lbl_cytoplasm = (
                                cells
                                * np.logical_not(
                            to_numpy(nuclei_raw) > 0
                        )  # use raw nuclei not dilated ones!
                                * np.logical_not(lbl_membranes > 0)
                        )

                        lbl_mem_dset_name = "lbl_mem"
                        if lbl_mem_dset_name in f:
                            del f[lbl_mem_dset_name]
                        lbl_mem_dset = f.create_dataset(
                            lbl_mem_dset_name, data=lbl_membranes, compression="gzip"
                        )
                        h5_copy_attributes(out_dset, lbl_mem_dset, exclude=("stain",))
                        h5_write_attributes(lbl_mem_dset, {"stain": "membrane"})

                        lbl_cyto_dset_name = "lbl_cyto"
                        if lbl_cyto_dset_name in f:
                            del f[lbl_cyto_dset_name]
                        lbl_cyto_dset = f.create_dataset(
                            lbl_cyto_dset_name, data=lbl_cytoplasm, compression="gzip"
                        )
                        h5_copy_attributes(out_dset, lbl_cyto_dset, exclude=("stain",))
                        h5_write_attributes(lbl_cyto_dset, {"stain": "cyto"})

    print("done")

def segment_cells_skimage_no_membranes(
        fn,
        seed_structure="nuclei",
        fg_structure="embryo",
        compactness=0.1,
        add_membranes_and_cytoplasm=True,
):
    with h5py.File(fn, "a") as f:
        print("loading images...")
        nuclei_dset = h5_select(f, {"stain": seed_structure})[0]
        nuclei_raw = to_itk(nuclei_dset)

        fg_dset = h5_select(f, {"stain": fg_structure})[0]
        fg = to_itk(fg_dset)
        # fg_dil = to_itk(apply_image_filter(to_labelmap(fg), "dilation", 3))
        fg_dil = fg

        print("dilating nuclei...")
        # nuclei = to_itk(apply_image_filter(to_labelmap(nuclei_raw), "dilation", 1))
        # nuclei = to_itk(apply_image_filter(to_labelmap(nuclei_raw), "erosion", 1))
        nuclei = nuclei_raw

        print("filtering membrane image...")
        membranes_filt = np.zeros_like(to_numpy(nuclei))

        print("upsampling labelimages...")
        nuclei = resample_to_shape(to_numpy(nuclei), membranes_filt.shape, order=0)
        fg_dil = resample_to_shape(to_numpy(fg_dil), membranes_filt.shape, order=0)

        print("applying background seeded watershed...")
        cells = watershed(
            image=membranes_filt,
            markers=nuclei,
            mask=fg_dil,
            compactness=compactness,
        ).astype(np.uint16)

        print("writing results...")
        out_dset_name = "lbl_cells"
        if out_dset_name in f:
            del f[out_dset_name]
        out_dset = f.create_dataset(out_dset_name, data=cells, compression="gzip", )
        h5_copy_attributes(
            nuclei_dset, out_dset, exclude=("stain", )
        )
        h5_write_attributes(
            out_dset, {"stain": "cells"}
        )
        if add_membranes_and_cytoplasm:
            print("calculating membranes and cytoplasm...")
            lbl_membranes = to_numpy(to_contours(to_itk(cells)))
            lbl_cytoplasm = (
                    cells
                    * np.logical_not(
                to_numpy(nuclei_raw) > 0
            )  # use raw nuclei not dilated ones!
                    * np.logical_not(lbl_membranes > 0)
            )

            lbl_mem_dset_name = "lbl_mem"
            if lbl_mem_dset_name in f:
                del f[lbl_mem_dset_name]
            lbl_mem_dset = f.create_dataset(
                lbl_mem_dset_name, data=lbl_membranes, compression="gzip"
            )
            h5_copy_attributes(out_dset, lbl_mem_dset, exclude=("stain",))
            h5_write_attributes(lbl_mem_dset, {"stain": "membrane"})

            lbl_cyto_dset_name = "lbl_cyto"
            if lbl_cyto_dset_name in f:
                del f[lbl_cyto_dset_name]
            lbl_cyto_dset = f.create_dataset(
                lbl_cyto_dset_name, data=lbl_cytoplasm, compression="gzip"
            )
            h5_copy_attributes(out_dset, lbl_cyto_dset, exclude=("stain",))
            h5_write_attributes(lbl_cyto_dset, {"stain": "cyto"})

    print("done")


def segment_cells(
        fn,
        membrane_stain="bCatenin",
        membrane_level="same",
        seed_structure="nuclei",
        fg_structure="embryo",
        filter_radii=(2,),
):
    with h5py.File(fn, "a") as f:
        print("loading images...")
        nuclei_dset = h5_select(f, {"stain": seed_structure})[0]
        nuclei = to_itk(nuclei_dset)

        fg_dset = h5_select(f, {"stain": fg_structure})[0]
        fg = to_itk(fg_dset)
        fg_dil = to_itk(apply_image_filter(to_labelmap(fg), "dilation", 4))

        if membrane_level == "same":
            membrane_level = nuclei_dset.attrs.get("level")
        membranes_dset = h5_select(
            f, {"stain": membrane_stain, "level": membrane_level}
        )[0]
        membranes = to_itk(membranes_dset)

        print("dilating nuclei...")
        nuclei = to_itk(apply_image_filter(to_labelmap(nuclei), "dilation", 1))

        for filter_radius in filter_radii:
            print("filtering membrane image...")
            membranes_filt = to_numpy(median(membranes, radius=filter_radius))

            print("adding embryo outline")
            outline = to_contours(fg)
            outline = smooth(outline, variance=10)
            outline = resample_to_shape(
                to_numpy(outline), membranes_filt.shape, order=1
            )
            membranes_filt = membranes_filt + outline

            print("upsampling labelimages...")
            nuclei = resample_to_shape(to_numpy(nuclei), membranes_filt.shape, order=0)
            fg_dil = resample_to_shape(to_numpy(fg_dil), membranes_filt.shape, order=0)

            print("applying background seeded watershed...")
            cells = background_seeded_watershed(membranes_filt, nuclei, fg_dil)

            print("writing results...")
            out_dset_name = "lbl_cells_{}_{}".format(filter_radius, membrane_level)
            if out_dset_name in f:
                del f[out_dset_name]
            out_dset = f.create_dataset(out_dset_name, data=cells, compression="gzip", )
            h5_copy_attributes(
                nuclei_dset, out_dset, exclude=("stain", "element_size_um", "level")
            )
            h5_copy_attributes(
                membranes_dset, out_dset, include=("element_size_um", "level")
            )
            h5_write_attributes(
                out_dset, {"stain": "cells_{}_{}".format(filter_radius, membrane_level)}
            )

    print("done")


if __name__ == "__main__":
    main()