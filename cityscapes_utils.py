from pathlib import Path
from typing import Dict,List, Tuple
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


class CityscapeUtils:

    #les categories principales
    MAIN_CLASSES = [
        "void",
        "flat",
        "construction",
        "object",
        "nature",
        "sky",
        "human",
        "vehicle",
    ]

    # Mapping labels fins -> catégories principales (Cityscapes)
    FINE_TO_MAIN: Dict[str, str] = {

        # flat
        "road": "flat",
        "sidewalk": "flat",
        "parking": "flat",
        "rail track": "flat",

        # human
        "person": "human",
        "rider": "human",

        # vehicle
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "on rails": "vehicle",
        "motorcycle": "vehicle",
        "bicycle": "vehicle",
        "caravan": "vehicle",
        "trailer": "vehicle",


        # construction
        "building": "construction",
        "wall": "construction",
        "fence": "construction",
        "guard rail": "construction",
        "bridge": "construction",
        "tunnel": "construction",

        # object
        "pole": "object",
        "polegroup": "object",
        "traffic sign": "object",
        "traffic light": "object",


        # nature
        "vegetation": "nature",
        "terrain": "nature",

        # sky
        "sky": "sky",

        #void
        "ground": "void",
        "dynamic": "void",
        "static": "void",
    }

    # Mapping officiel Cityscapes : labelId -> (nom, catégorie principale)
    # Source : https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    LABEL_ID_TO_MAIN: Dict[int, str] = {
        0: "void",           # unlabeled
        1: "void",           # ego vehicle
        2: "void",           # rectification border
        3: "void",           # out of roi
        4: "void",           # static
        5: "void",           # dynamic
        6: "void",           # ground
        7: "flat",           # road
        8: "flat",           # sidewalk
        9: "flat",           # parking
        10: "flat",          # rail track
        11: "construction",  # building
        12: "construction",  # wall
        13: "construction",  # fence
        14: "construction",  # guard rail
        15: "construction",  # bridge
        16: "construction",  # tunnel
        17: "object",        # pole
        18: "object",        # polegroup
        19: "object",        # traffic light
        20: "object",        # traffic sign
        21: "nature",        # vegetation
        22: "nature",        # terrain
        23: "sky",           # sky
        24: "human",         # person
        25: "human",         # rider
        26: "vehicle",       # car
        27: "vehicle",       # truck
        28: "vehicle",       # bus
        29: "vehicle",       # caravan
        30: "vehicle",       # trailer
        31: "vehicle",       # train (on rails)
        32: "vehicle",       # motorcycle
        33: "vehicle",       # bicycle
    }

    # Palette RGB pour les 8 classes (0..7)
    PALETTE_8 = np.array(
        [
            [0, 0, 0],         # 0 void
            [128, 64, 128],    # 1 flat
            [70, 70, 70],      # 2 construction
            [220, 220, 0],     # 3 object
            [107, 142, 35],    # 4 nature
            [70, 130, 180],    # 5 sky
            [220, 20, 60],     # 6 human
            [0, 0, 142],       # 7 vehicle
        ],
        dtype=np.uint8,
    )

    @staticmethod
    def build_label_lut_8() -> np.ndarray:
        """
        Construit une LUT (Look-Up Table) : labelId (0..255) -> indice 8 classes (0..7)

        Utilisation :
            lut = CityscapeUtils.build_label_lut_8()
            mask_8 = lut[mask_labelIds]  # conversion instantanée de tout le masque
        """
        # Par défaut tout est void (classe 0)
        lut = np.zeros(256, dtype=np.uint8)

        # Index des 8 classes principales
        class_to_idx = {name: idx for idx, name in enumerate(CityscapeUtils.MAIN_CLASSES)}

        # Remplir la LUT
        for label_id, main_class in CityscapeUtils.LABEL_ID_TO_MAIN.items():
            lut[label_id] = class_to_idx[main_class]

        return lut

    @staticmethod
    def colorize_mask_8(mask_8: np.ndarray) -> np.ndarray:
        """
        Convertit un masque 8 classes (H, W) en image RGB (H, W, 3) avec la palette.

        Utilisation :
            mask_rgb = CityscapeUtils.colorize_mask_8(mask_8)
        """
        return CityscapeUtils.PALETTE_8[mask_8]

    @staticmethod
    def labelIds_to_8(label_mask: np.ndarray) -> np.ndarray:
        """
        Convertit un masque labelIds (0..33) en masque 8 classes (0..7).

        Utilisation :
            mask_8 = CityscapeUtils.labelIds_to_8(label_mask)
        """
        lut = CityscapeUtils.build_label_lut_8()
        return lut[label_mask]

    @staticmethod
    def get_light_paths(
        img_paths: List[str],
        mask_paths: List[str],
        cities: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Filtre les chemins pour ne garder que les images des villes spécifiées.
        Utile pour un entraînement léger ("light training") sur 2-3 villes.

        Paramètres
        ----------
        img_paths  : liste complète des chemins images
        mask_paths : liste complète des chemins masques (même ordre)
        cities     : liste des villes à conserver, ex. ['aachen', 'bremen']

        Retourne
        --------
        (img_paths_filtered, mask_paths_filtered)

        Utilisation :
            train_imgs, train_masks = CityscapeUtils.get_light_paths(
                all_imgs, all_masks, cities=['aachen', 'bremen']
            )
        """
        cities_set = set(cities)
        filtered = [
            (img, mask)
            for img, mask in zip(img_paths, mask_paths)
            if Path(mask).parent.name in cities_set
        ]
        if not filtered:
            raise ValueError(f"Aucune image trouvée pour les villes : {cities}")
        img_f, mask_f = zip(*filtered)
        return list(img_f), list(mask_f)
