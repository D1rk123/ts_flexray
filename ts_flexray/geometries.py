#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

import tomosipo as ts

"""
The calibration profiles were copied with modifed names from the flexDATA
repository.
To be able to re-use these calibration profiles the code to read the
scan_settings.txt files was also based on similar code in the flexDATA
repository.

flexDATA repository: https://github.com/cicwi/flexDATA
"""

"""
Created on Mon May  8 14:58:40 2023

@author: des
"""
profiles = {
     'cwi-flexray-2023-08-21': {
        'description': """
Correction profile deduced from Acquila hc/vc/cor settings after the exchange of the detector motor.
""",
        'tra_det': 24.1215,
        'ver_tube': -6.2917,
        'tra_obj': -0.5405
    },
    'cwi-flexray-2022-10-28': {
        'description': """
Correction profile deduced from markers after the October 2022 maintenance.
""",
        'tra_det': 24.0485,
        'ver_tube': -5.7730,
        'tra_obj': -0.5010
    },
    'cwi-flexray-2022-05-31': {
        'description': """
correction profile deduced from acquila hc/vc/cor settings after the 31 may 2022 re-calibration. Includes det_roll determined by Robert using markers.
""",
        'tra_det': 24.4203,
        'ver_tube': -6.2281,
        'tra_obj': -0.5010,
        'roll_det': -0.262,
    },
    'cwi-flexray-2022-05-31-norotation': {
        'description': """
correction profile deduced from acquila hc/vc/cor settings after the 31 may 2022 re-calibration.
""",
        'tra_det': 24.4203,
        'ver_tube': -6.2281,
        'tra_obj': -0.5010,
    },
    'cwi-flexray-2020-03-26': {
        'description': """
Correction profile deduced from Acquila HC/VC/COR settings after the March 2020 maintenance. Includes empirically determined det_roll.
""",
        'tra_det': 24.300,
        'ver_tube': -6.086,
        'tra_obj': -0.524,
        'roll_det': -0.175,
    },
    'cwi-flexray-2020-03-26-norotation': {
        'description': """
Correction profile deduced from Acquila HC/VC/COR settings after the March 2020 maintenance.
""",
        'tra_det': 24.300,
        'ver_tube': -6.086,
        'tra_obj': -0.524,
    },
    'cwi-flexray-2019-04-24': {
        'description': """
This profile was last updated by Alex Kostenko on 24 April 2019.
It was concurrently updated with the documentation and some other changes in the flexdata codebase.
See:
https://github.com/cicwi/flexDATA/commit/8859bc8073880efcb32cc57b152ef23746993ec1#diff-08f83b989c80a05906f380a66964d8d3L393
""",
        'tra_det': 24,
        'ver_tube': -7,
        'axs_tan': -0.5,
    }
}
    
def apply_roi_offset(geom_dict):
    roi = geom_dict['ROI']

    # ROI is written for a pixel in 1x1 binning, so the centre of the detector
    # is at (767,971), and detector pixel size 74.8 um
    centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]
    detector_pixel_size = 0.0748

    ver_det = centre[1] * detector_pixel_size
    tra_det = centre[0] * detector_pixel_size

    geom_dict['ver_det'] += ver_det
    geom_dict['tra_det'] += tra_det

def apply_calibration_profile(geom_dict, profile):
    for key in profile.keys():
        if key == "description":
                continue
        geom_dict[key] += profile[key]

def parse_data_settings(path):
    float_keys = [
        "Voxel size", "Binned pixelsize (mm)",
        "SOD", "SDD",
        "Start angle", "Last angle",
        "ver_tube", "tra_tube", "ver_det", "tra_det", "tra_obj"]
    int_keys = ["total projections", "Binning value"]
    roi_key = "ROI"
    
    geom_dict = {"roll_det" : 0}
    with open(path, "r") as file:
        for line in file:
            split_point =  line.find("=")
            if split_point == -1:
                continue

            key = line[:split_point].strip()
            value = line[split_point+1:]
            value = value.replace('"', '')
            
            if key in geom_dict.keys():
                continue

            if key in float_keys:
                geom_dict[key] = float(value)
            if key in int_keys:
                geom_dict[key] = int(float(value))
            if key == roi_key:
                geom_dict[key] = [int(x) for x in value.split(";")]

    geom_dict["ODD"] = geom_dict["SDD"]-geom_dict["SOD"]
    return geom_dict

def make_flexray_geometries(path, profile=None, skip_last=True, data_settings_file_name="data settings XRE.txt"):
    geom_dict = parse_data_settings(path / data_settings_file_name)
    apply_roi_offset(geom_dict)
    if profile is not None:
        apply_calibration_profile(geom_dict, profile=profiles[profile])
        
    pixel_size = geom_dict["Binned pixelsize (mm)"]
    roi = geom_dict["ROI"]
    det_shape = np.array((roi[3]-roi[1]+1, roi[2]-roi[0]+1)) // geom_dict["Binning value"]
    roll_det = np.radians(geom_dict["roll_det"])
    det_v = np.array((math.cos(roll_det), 0, math.sin(roll_det))) * pixel_size
    det_u = np.array((-math.sin(roll_det), 0, math.cos(roll_det))) * pixel_size
    
    pg = ts.cone_vec(
	    shape = det_shape,
        src_pos = (geom_dict["ver_tube"], -geom_dict["SOD"], geom_dict["tra_tube"]),
        det_pos = (geom_dict["ver_det"], geom_dict["ODD"], geom_dict["tra_det"]),
        det_v = det_v,
        det_u = det_u
    )
    
    angles = np.radians(np.linspace(
        geom_dict["Start angle"],
        geom_dict["Last angle"],
        geom_dict["total projections"]
        ))
    if skip_last:
        angles = angles[:-1]

    vol_ver = (geom_dict["ver_det"]*geom_dict["SOD"] +
               geom_dict["ver_tube"]*geom_dict["ODD"]) / geom_dict["SDD"]
    vol_pos = np.array((vol_ver, 0, geom_dict["tra_obj"]))
    vol_shape = np.array((det_shape[0], det_shape[1], det_shape[1]))
    
    rot = ts.rotate(pos=vol_pos, axis=np.array((-1, 0, 0)), angles=angles)
    vg = rot * ts.volume(
        shape = vol_shape,
        size = vol_shape*geom_dict["Voxel size"],
        pos = vol_pos
    ).to_vec()
    return vg, pg, geom_dict
