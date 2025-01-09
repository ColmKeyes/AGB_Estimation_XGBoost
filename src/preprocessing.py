# -*- coding: utf-8 -*-
"""
This Script provides functions for preprocessing Sentinel-2 data for AGB Estimation.
"""
"""
@Time    : 05/09/2024 11:34
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : sentinel2_preprocessing
"""
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from rasterio.mask import mask
import os
import numpy as np

class preprocessing:

    def __init__(self, sentinel2_path, output_stack_path, bands, shp=None):

        self.sentinel2_path = sentinel2_path
        self.output_stack_path = output_stack_path
        self.bands = bands
        self.ordered_band_list = []
        self.shp = shp
        self.cube = None

    def stack_sentinel(self):
        """
        Write a stack of Sentinel-2 jp2s into a single GeoTIFF file.
        """
        # write path
        stack_file_path = os.path.join(self.output_stack_path, 'sentinel2_stack.tif')
        if os.path.exists(stack_file_path):
            print(f"File {stack_file_path} already exists. Skipping function.")
            return stack_file_path

        # Collect bands
        files = [os.path.join(self.sentinel2_path, file) for file in os.listdir(self.sentinel2_path)
                 if any(band in file for band in self.bands) and file.endswith('.jp2')]
        for file in files:
            band = os.path.basename(file).split('_')[-2]
            self.ordered_band_list.append(band)
        print("Band Order:", self.ordered_band_list)

        # update metadata from jp2 Uint16 to Geotiff float32
        with rasterio.open(files[0]) as src0:
            meta = src0.meta
        meta.update(driver="GTiff",count=len(files), dtype="float32")

        # write
        with rasterio.open(stack_file_path, 'w', **meta) as dst:
            for id, layer in enumerate(files, start=1):
                with rasterio.open(layer) as src1:
                    dst.write_band(id, src1.read(1))

        return stack_file_path


    def crop_stack_to_aoi(self, stack_path, aoi_geojson, output_path):
        """
        Crop the Sentinel-2 stack to the AOI.
        """
        # write path
        output_cropped_path = os.path.join(output_path, 'cropped_sentinel_stack.tif')
        if os.path.exists(output_cropped_path):
            print(f"File {output_cropped_path} already exists. Skipping")
            return output_cropped_path

        # align aoi and src CRS
        aoi = gpd.read_file(aoi_geojson)
        with rasterio.open(stack_path) as src:
            aoi = aoi.to_crs(src.crs)

            # Extract geometry and crop
            aoi_geom = [aoi.geometry.unary_union.__geo_interface__]
            out_image, out_transform = mask(src, aoi_geom, crop=True)

            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform})

            # Write
            with rasterio.open(output_cropped_path, 'w', **out_meta) as dest:
                dest.write(out_image)

        return output_cropped_path





    def convert_labels_to_raster(self, output_stack_path, output_path):
        """
        Convert GeoJSON AGB values to a raster.
        """
        # write path
        output_label_path = os.path.join(output_path, 'cropped_label.tif')
        if os.path.exists(output_label_path):
            print(f"File {output_label_path} already exists. Skipping")
            return output_label_path

        # Extract metadata
        with rasterio.open(output_stack_path) as stack_src:
            width = stack_src.width
            height = stack_src.height
            transform = stack_src.transform
            crs = stack_src.crs

        # reproject geo_df to stack_src crs
        geo_df = gpd.read_file(self.shp)
        geo_df = geo_df.to_crs(crs=stack_src.crs)

        # Convert to suitable format
        shapes = [(geom, agb_value) for geom, agb_value in zip(geo_df.geometry, geo_df['agbd_ton_ha'])]

        # Rasterise
        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=np.nan,
            dtype='float32'
        )

        # Write
        with rasterio.open(
                output_label_path, 'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype='float32',
                crs=crs,
                transform=transform
        ) as dst:
            dst.write(raster, 1)

        return output_label_path


    def compute_traditional_indices(self, sentinel_stack):
        """
        Compute Traditional vegetation indices & stack.
        """
        # read
        with rasterio.open(sentinel_stack) as src:
            sentinel_stack = src.read()
        band_map = {band: idx for idx, band in enumerate(self.ordered_band_list)}

        rBlue = sentinel_stack[band_map['B02']]
        rGreen = sentinel_stack[band_map['B03']]
        rRed = sentinel_stack[band_map['B04']]
        rNIR = sentinel_stack[band_map['B8A']]
        rSWIR1 = sentinel_stack[band_map['B11']]

        ########################
        # Traditional Indices:
        # 1. ARVI (Atmospherically Resistant Vegetation Index)
        # 2. CIg (Chlorophyll Index - Green)
        # 3. DVI (Difference Vegetation Index)
        # 4. EVI (Enhanced Vegetation Index)
        # 5. GNDVI (Green Normalized Difference Vegetation Index)
        # 6. MSAVI (Modified Soil Adjusted Vegetation Index)
        # 7. NDII (Normalized Difference Infrared Index)
        # 8. NDVI (Normalized Difference Vegetation Index)
        # 9. SR (Simple Ratio)
        ########################

        # Compute
        ARVI = np.where((rNIR + (2 * rRed - rBlue)) == 0, np.nan, (rNIR - (2 * rRed - rBlue)) / (rNIR + (2 * rRed - rBlue)))
        CIg = np.where(rGreen == 0, np.nan, (rNIR / rGreen) - 1)
        DVI = rNIR - rRed
        EVI = np.where((rNIR + 6 * rRed - 7.5 * rBlue + 1) == 0, np.nan, 2.5 * (rNIR - rRed) / (rNIR + 6 * rRed - 7.5 * rBlue + 1))
        GNDVI = np.where((rNIR + rGreen) == 0, np.nan, (rNIR - rGreen) / (rNIR + rGreen))
        MSAVI = np.where((2 * rNIR + 1) ** 2 - 8 * (rNIR - rRed) < 0, np.nan,
                         (2 * rNIR + 1 - np.sqrt((2 * rNIR + 1) ** 2 - 8 * (rNIR - rRed))) / 2)
        NDII = np.where((rNIR + rSWIR1) == 0, np.nan, (rNIR - rSWIR1) / (rNIR + rSWIR1))
        NDVI = np.where((rNIR + rRed) == 0, np.nan, (rNIR - rRed) / (rNIR + rRed))
        SR = np.where(rRed == 0, np.nan, rNIR / rRed)

        # Stack
        traditional_indices = np.stack([ARVI, CIg, DVI, EVI, GNDVI, MSAVI, NDII, NDVI, SR], axis=0)
        combined_stack = np.vstack([sentinel_stack, traditional_indices])

        # Write
        meta = src.meta
        meta.update(count=combined_stack.shape[0])
        output_stack_vi_path = os.path.join(self.output_stack_path, 'sentinel2_with_traditional_indices_stack.tif')
        with rasterio.open(output_stack_vi_path, 'w', **meta) as dst:
            for i in range(combined_stack.shape[0]):
                dst.write(combined_stack[i], i + 1)

        return output_stack_vi_path


    def compute_red_edge_indices(self, sentinel_stack):
        """
        Compute  Red Edge indices and stack.
        """
        # Read
        with rasterio.open(sentinel_stack) as src:
            sentinel_stack = src.read()
        band_map = {band: idx for idx, band in enumerate(self.ordered_band_list)}

        rBlue = sentinel_stack[band_map['B02']]
        rGreen = sentinel_stack[band_map['B03']]
        rRed = sentinel_stack[band_map['B04']]
        rRE1 = sentinel_stack[band_map['B05']]
        rRE2 = sentinel_stack[band_map['B06']]
        rRE3 = sentinel_stack[band_map['B07']]
        rNIR = sentinel_stack[band_map['B8A']]

        ########################
        # Red Edge Indices:
        # 1. CIre (Chlorophyll Index - Red Edge)
        # 2. IRECI (Inverted Red-Edge Chlorophyll Index)
        # 3. MCARI (Modified Chlorophyll Absorption Ratio Index)
        # 4. NDVIre1 (Normalized Difference Vegetation Index - Red Edge 1)
        # 5. NDVIre2 (Normalized Difference Vegetation Index - Red Edge 2)
        # 6. NDVIre3 (Normalized Difference Vegetation Index - Red Edge 3)
        # 7. NDre1 (Normalized Difference Red Edge 1)
        # 8. NDre2 (Normalized Difference Red Edge 2)
        # 9. SRre (Simple Ratio - Red Edge)
        # 10. S2REP (Sentinel-2 Red-Edge Position)
        ########################

        # Compute
        CIre = np.where(rRE1 == 0, np.nan, (rRE3 / rRE1) - 1)
        IRECI = np.where(rRE1 == 0, np.nan, (rRE3 - rRed) / (rRE1 / rRE2))
        MCARI = np.where(rRed == 0, np.nan, ((rRE1 - rRed) - 0.2 * (rRE1 - rGreen)) * (rRE1 / rRed))
        NDVIre1 = np.where((rNIR + rRE1) == 0, np.nan, (rNIR - rRE1) / (rNIR + rRE1))
        NDVIre2 = np.where((rNIR + rRE2) == 0, np.nan, (rNIR - rRE2) / (rNIR + rRE2))
        NDVIre3 = np.where((rNIR + rRE3) == 0, np.nan, (rNIR - rRE3) / (rNIR + rRE3))
        NDre1 = np.where((rRE2 + rRE1) == 0, np.nan, (rRE2 - rRE1) / (rRE2 + rRE1))
        NDre2 = np.where((rRE3 + rRE1) == 0, np.nan, (rRE3 - rRE1) / (rRE3 + rRE1))
        SRre = np.where(rRE1 == 0, np.nan, rNIR / rRE1)

        # Stack
        additional_indices = np.stack([CIre, IRECI, MCARI, NDVIre1, NDVIre2, NDVIre3, NDre1, NDre2, SRre], axis=0)
        combined_stack = np.vstack([sentinel_stack, additional_indices])

        # Write
        meta = src.meta
        meta.update(count=combined_stack.shape[0])
        output_stack_vi_path = os.path.join(self.output_stack_path, 'sentinel2_with_red_edge_stack.tif')
        with rasterio.open(output_stack_vi_path, 'w', **meta) as dst:
            for i in range(combined_stack.shape[0]):
                dst.write(combined_stack[i], i + 1)

        return output_stack_vi_path

    def normalise_stack(self, combined_stack):
        """
        Normalise each band to the range [0, 1].
        """

        # write path
        output_stack_path = os.path.join(self.output_stack_path, 'normalised_sentinel_stack.tif')

        # Read
        with rasterio.open(combined_stack) as src:
            sentinel_stack = src.read()
        meta = src.meta.copy()

        # get band range & normalise
        for i in range(sentinel_stack.shape[0]):
            band = sentinel_stack[i]
            band[band == 0] = np.nan
            min_val = np.nanmin(band)
            max_val = np.nanmax(band)

            normalised_stack = np.zeros_like(sentinel_stack, dtype='float32')
            normalised_stack[i] = (band - min_val) / (max_val - min_val)

        # Write
        meta.update(dtype='float32', count=normalised_stack.shape[0])
        with rasterio.open(output_stack_path, 'w', **meta) as dst:
            for i in range(normalised_stack.shape[0]):
                dst.write(normalised_stack[i], i + 1)  # Write each band to the output file

        return output_stack_path


    def stack_label(self, sentinel_stack_path, label_path):
        """
        Add the label raster on top of the existing Sentinel-2 stack.
        """
        # Write path
        stacked_with_labels_path = os.path.join(self.output_stack_path, 'sen2_stack_re_labels.tif')

        # Write
        with rasterio.open(sentinel_stack_path) as src:
            meta = src.meta
            meta.update(count=src.count + 1)
            with rasterio.open(stacked_with_labels_path, 'w', **meta) as dst:
                for band_id in range(1, src.count + 1):
                    dst.write_band(band_id, src.read(band_id))
                with rasterio.open(label_path) as label_src:
                    label_data = label_src.read(1)
                    dst.write_band(src.count + 1, label_data)

        return stacked_with_labels_path




    ## not required, MSI gives RGB at 10,20 and 60m.
    # def resample_rgb_to_20m(self):


