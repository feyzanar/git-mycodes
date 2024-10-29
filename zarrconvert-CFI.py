
# activate the env run the code below in command line
# (conda create -n zarr_convert ome::bioformats2raw)
bioformats2raw "H:\PROJECTS-03\Feyza\240925-NcadGFPxH2Bch\20240925_151619_20240925_NcadxH2B_05z_timelapse\pos4_cropped_t1t2\ome-tiff.companion.ome" H:\PROJECTS-03\Feyza\240925-NcadGFPxH2Bch\20240925_151619_20240925_NcadxH2B_05z_timelapse\pos4_cropped_t1t2\zarr_test\Pos4_bioformats2raw_2.ome.zarr -p --max-workers 30 -h 64 -w 64 -z 16 --downsample-type GAUSSIAN
