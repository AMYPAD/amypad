'''
Processing of PET images for AmyPET
'''

__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2022-3"

import logging
import os, csv
from glob import glob
import shutil
from pathlib import Path, PurePath

import numpy as np
import spm12
from miutil.fdio import hasext
from niftypet import nimpa

try:          # py<3.9
    import importlib_resources as resources
except ImportError:
    from importlib import resources

from .dyn_tools import dyn_timing
from .utils import get_atlas

log = logging.getLogger(__name__)

#----------------------------------------------------------
# VOI codes for AAL and Hammers atlases
# > new AAL codes!
aal_vois = {
    'cerebellum': list(range(95, 120)), 'frontal': list(range(1, 25)) + [73, 74],
    'parietal': list(range(61, 72)), 'occipital': list(range(47, 59)),
    'temporal': [59, 60] + list(range(83, 95)), 'insula': [33, 34],
    'precuneus': [71, 72], 'antmidcingulate': list(range(151, 157)) + [37, 38],
    'postcingulate': [39, 40], 'hippocampus': [41, 42], 'caudate': [75, 76],
    'putamen': [77, 78], 'thalamus': list(range(121, 151)),
    'composite': list(range(3, 29)) + list(range(31, 37)) + list(range(59, 69)) +
    list(range(63, 72)) + list(range(85, 91))}

hmmrs_vois = {
    'cerebellum': [17, 18],
    'frontal': [28, 29] + list(range(50, 60)) + list(range(68,
        74)) + list(range(76, 82)),
    'parietal': [32, 33, 60, 61, 62, 63, 84, 85],
    'occipital': [22, 23, 64, 65, 66, 67],
    'temporal': list(range(5, 17)) + [82, 83],
    'insula': [20, 21] + list(range(86, 96)),
    'antecingulate': [24, 25],
    'postcingulate': [26, 27],
    'hippocampus': [1, 2],
    'caudate': [34, 35],
    'putamen': [38, 39],
    'thalamus': [40, 41],
    'composite': [28, 29] + list(range(52, 60)) + list(range(76,
        82)) + list(range(86, 96)) + [32, 33, 62, 63, 84, 85]}

sch_vois = {}
sch_csv = resources.files('amypet').resolve() / 'data' / 'atlas' / 'Schaefer_2018_100_Parcels.csv'
with open(sch_csv) as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    for row in reader:
        key = row[1].split('7Networks_')[1]
        value = [int(row[0])]
        sch_vois[key] = value
#----------------------------------------------------------


def atl2pet(fatl, cldct, fpet=None, outpath=None):
    '''
    Atlas and GM from the centiloid (CL) pipeline to the reference
    PET space.
    Arguments:
    - fatl:     the file path of the atlas in MNI space
    - cldct:    the CL output dictionary
    - fpet:     the reference PET (file path or dictionary)
                for reslicing into PET space
    '''

    # > output path
    if outpath is None:
        opth = Path(cldct['opth']).parent.parent
    else:
        opth = Path(outpath)
    nimpa.create_dir(opth)

    # > decipher the CL dictionary
    if len(cldct) == 1:
        cl_dct = cldct[next(iter(cldct))]
    elif 'norm' in cldct:
        cl_dct = cldct
    else:
        raise ValueError('unrecognised CL dictionary')

    # > get the reference PET image if provided, otherwise get it from CL pipeline
    if isinstance(fpet, (str, PurePath)) and os.path.isfile(fpet):
        frefpet = Path(fpet)
    elif isinstance(fpet, dict) and 'voxsize' in fpet:
        frefpet = fpet
    else:
        frefpet = cl_dct['petc']['fim'] 

    # > read the PET image
    petdct = nimpa.getnii(frefpet, output='all')

    # > SPM bounding box of the PET image
    bbox = spm12.get_bbox(petdct)

    # # > get the affine PET->MR
    # if isinstance(affine, (Path, PurePath)) or isinstance(affine, str):
    #     aff = np.loadtxt(affine)
    # elif isinstance(affine, np.array):
    #     aff = np.array(affine)

    # > get the inverse affine transform to PET native space
    M = np.linalg.inv(cl_dct['reg2']['affine'])
    import matlab as ml
    Mm = ml.double(M.tolist())

    # > copy the inverse definitions to be modified with affine to native PET space
    fmod = shutil.copyfile(
        cl_dct['norm']['invdef'],
        opth / (Path(cl_dct['norm']['invdef']).name.split('.')[0] + '_2nat.nii'))
    eng = spm12.ensure_spm('')
    eng.amypad_coreg_modify_affine(str(fmod), Mm)

    # > unzip the atlas and transform it to PET space
    fniiatl = nimpa.nii_ugzip(fatl, outpath=opth)

    # > inverse transform the atlas to PET space
    finvatl = spm12.normw_spm(str(fmod), fniiatl, voxsz=np.flip(petdct['voxsize']),
                              intrp=0., bbox=bbox, outpath=str(opth))[0]

    # > remove the uncompressed input atlas after transforming it
    os.remove(fniiatl)

    # > GM mask
    fgmpet = spm12.resample_spm(frefpet, cl_dct['norm']['c1'], M, intrp=1.0, outpath=opth,
                                pickname='flo', fcomment='_GM_in_PET', del_ref_uncmpr=True,
                                del_flo_uncmpr=True, del_out_uncmpr=True)
    # > WM mask
    fwmpet = spm12.resample_spm(frefpet, cl_dct['norm']['c2'], M, intrp=1.0, outpath=opth,
                                pickname='flo', fcomment='_WM_in_PET', del_ref_uncmpr=True,
                                del_flo_uncmpr=True, del_out_uncmpr=True)

    # > remove NaNs
    atl = nimpa.getnii(finvatl)
    atl[np.isnan(atl)] = 0
    gm = nimpa.getnii(fgmpet)
    gm[np.isnan(gm)] = 0
    wm = nimpa.getnii(fwmpet)
    wm[np.isnan(wm)] = 0

    return {
        'fatlpet': finvatl,
        'fwmpet': fwmpet,
        'fgmpet': fgmpet,
        'atlpet': atl,
        'gmpet': gm,
        'wmpet': wm,
        'outpath': opth,
        'bbox': bbox}


# ========================================================================================
def extract_vois(impet, atlas, voi_dct, atlas_mask=None, outpath=None, output_masks=False):
    '''
    Extract VOI mean values from PET image `impet` using image labels `atals`.
    Both can be dictionaries, file paths or Numpy arrays.
    They have to be aligned and have the same dimensions.
    If path (output) is given, the ROI masks will be saved to file(s).
    Arguments:
        - impet:    PET image as Numpy array
        - atlas:  image of labels (integer values); the labels can come
                    from T1w-based parcellation or an atlas.
        - voi_dct:  dictionary of VOIs, with entries of labels creating
                    composite volumes
        - atlas_mask: masks the atlas with an additional maks, e.g., with the
                    grey matter probability mask.
        - output_masks: if `True`, output Numpy VOI masks in the output
                    dictionary
        - outpath:  if given as a folder path, the VOI masks will be saved
    '''

    # > assume none of the below are given
    # > used only for saving ROI mask to file if requested
    affine, flip, trnsp = None, None, None

    # ----------------------------------------------
    # PET
    if isinstance(impet, dict):
        im = np.array(impet['im'])
        if 'affine' in impet:
            affine = impet['affine']
        if 'flip' in impet:
            flip = impet['flip']
        if 'transpose' in impet:
            trnsp = impet['transpose']

    elif isinstance(impet, (str, PurePath)) and os.path.isfile(impet):
        imd = nimpa.getnii(impet, output='all')
        im = np.array(imd['im'])
        flip = imd['flip']
        trnsp = imd['transpose']

    elif isinstance(impet, np.ndarray):
        im = impet
    # ----------------------------------------------

    # ----------------------------------------------
    # LABELS
    if isinstance(atlas, dict):
        lbls = atlas['im']
        if 'affine' in atlas and affine is None:
            affine = atlas['affine']
        if 'flip' in atlas and flip is None:
            flip = atlas['flip']
        if 'transpose' in atlas and trnsp is None:
            trnsp = atlas['transpose']

    elif isinstance(atlas, (str, PurePath)) and os.path.isfile(atlas):
        prd = nimpa.getnii(atlas, output='all')
        lbls = prd['im']
        if affine is None:
            affine = prd['affine']
        if flip is None:
            flip = prd['flip']
        if trnsp is None:
            trnsp = prd['transpose']

    elif isinstance(atlas, np.ndarray):
        lbls = atlas

    # > get rid of NaNs if any in the parcellation/label image
    lbls[np.isnan(lbls)] = 0

    # > atlas mask
    if atlas_mask is not None:
        if isinstance(atlas_mask, (str, PurePath)) and os.path.isfile(atlas_mask):
            amsk = nimpa.getnii(atlas_mask)
        elif isinstance(atlas_mask, np.ndarray):
            amsk = atlas_mask
        else:
            raise ValueError('Incorrectly provided atlas mask')

        # > remove any NaNs if present
        amsk[np.isnan(amsk)] = 0

    else:
        amsk = 1

    # > output dictionary
    out = {}

    # > get metric + ref (if UR/CL parametric images as input)
    metric = os.path.basename(impet)[:2] if 'ref-' in str(impet) else None
    metric = 'suvr_' if metric == 'UR' else 'cl_' if metric == 'CL' else None
    ref = impet.split('ref-', 1)[-1].split('wUR', 1)[0] if 'ref-' in str(impet) else None
    log.info(f' METRIC_REF: {metric}{ref}')
    # ----------------------------------------------

    log.debug('Extracting volumes of interest (VOIs):')
    for voi in voi_dct:
        log.info(f'  VOI: {voi}')

        # > ROI mask
        rmsk = np.zeros(lbls.shape, dtype=bool)

        for ri in voi_dct[voi]:
            log.debug(f'   label{ri}')
            rmsk += np.equal(lbls, ri)

        # > apply the mask on mask
        if not isinstance(amsk, np.ndarray) and amsk == 1:
            msk2 = rmsk
        else:
            msk2 = rmsk * amsk

        if msk2.dtype==type(True):
            msk2 = np.int8(msk2)

        if msk2.shape != (181, 217, 181):
            # Crop amsk to fit the desired shape
            msk2 = msk2[0:-1, 0:-1, 0:-1]

        if outpath is not None and not isinstance(atlas, np.ndarray):
            nimpa.create_dir(outpath)
            fvoi = Path(outpath) / f'{voi}_mask.nii.gz'
            nimpa.array2nii(msk2, affine, fvoi,
                            trnsp=(trnsp.index(0), trnsp.index(1), trnsp.index(2)), flip=flip)
        else:
            fvoi = None

        vxsum = np.sum(msk2)

        if im.ndim == 4:
            nfrm = im.shape[0]
            emsum = np.zeros(nfrm, dtype=np.float64)
            for fi in range(nfrm):
                emsum[fi] = np.sum(im[fi, ...].astype(np.float64) * msk2)

        elif im.ndim == 3:
            emsum = np.sum(im.astype(np.float64) * msk2)

        else:
            raise ValueError('unrecognised image shape or dimensions')

        out[voi] = {'vox_no': vxsum, 'sum': emsum, 'avg': emsum / vxsum, 'fvoi': fvoi}

        if output_masks:
            out[voi]['roimsk'] = msk2

        if ref:
            voi_name = f"{metric}{voi}_ref_{ref}"
            voi_metric = float(out[voi]['avg'])
            out[voi]['voi_dict'] = {voi_name: voi_metric}

    return out


# ========================================================================================
def proc_vois(
        aligned,
        cl_dct,
        atlas='hammers',
        default_mni=False,
        voi_idx=None,
        res=1,
        outpath=None,
        apply_mask='gm',
        timing=None,
        fcsv=None):
    '''
    Process and prepare the VOI dynamic data for kinetic analysis.
    Arguments:
    niidat:     dictionary with NIfTI file paths and properties with time.
    aligned:    dictionary of aligned frames, with properties or the path to 
                4D PET image;
    cl_dct:     dictionary of centiloid (CL) processing outputs - used
                for inverse transformation to native image spaces.
    atlas:      choice of atlas; default is the Hammers atlas (atlas='hammers'');
                AAL also is supported (atlas='aal'); any other custom atlas
                can be used if atlas is a path to the NIfTI file of the atlas;
                for custom atlas `voi_idx` must be provided as a dictionary.
    default_mni:option to use when applying atlas in MNI
    voi_ids:    VOI indices for composite VOIs.  Every atlas has its own
                labelling strategy.
    res:        resolution of the atlas - the default is 1 mm voxel size
                isotropically.
    apply_mask: applies either the grey matter mask ('gm') or the white matter
                mask ('wm') based on the T1w image to refine the VOI sampling.
    '''

    # > get sorted the dynamic PET input
    if isinstance(aligned, dict) and 'fpet' in aligned:
        fdynin = aligned['fpet']
    elif isinstance(aligned, (str, PurePath)) and os.path.isfile(aligned):
        fdynin = aligned

    # > output path
    if outpath is None:
        opth = aligned['fpet'].parent / 'DYN'
    else:
        opth = outpath
    nimpa.create_dir(opth)

    # > get the atlas
    if isinstance(atlas, (str, PurePath)) and hasext(atlas, ('nii', 'nii.gz')):
        fatl = atlas
    elif isinstance(atlas, str) and atlas in ['hammers', 'aal', 'schaefer']:
        datl = get_atlas(atlas=atlas, res=res)
        fatl = datl['fatlas']

    if voi_idx is not None and isinstance(voi_idx, dict):
        dvoi = voi_idx
    else:
        if atlas == 'aal':
            # > New AAL3 codes!
            dvoi = aal_vois
        elif atlas == 'hammers':
            dvoi = hmmrs_vois
        elif atlas == 'schaefer':
            dvoi = sch_vois
        else:
            raise ValueError('unrecognised atlas name!')

        if default_mni:
            cl_dct['fur'] = glob(os.path.dirname(cl_dct['fqc']) + '/ur-image/*')
            cl_dct['fcl'] = glob(os.path.dirname(cl_dct['fqc']) + '/cl-image/*wcw*')
            voi_dct = {}
            for fur in cl_dct['fur']:
                rvoi = extract_vois(fur, fatl, dvoi, atlas_mask=None,
                                    outpath=opth / f'masks_{atlas}', output_masks=True)
                for voi in dvoi.keys():
                    voi_dct.update(rvoi[voi]['voi_dict'])
            rvoi = extract_vois(cl_dct['fcl'][0], fatl, dvoi, atlas_mask=None,
                                outpath=opth / f'masks_{atlas}', output_masks=True)
            for voi in dvoi.keys():
                voi_dct.update(rvoi[voi]['voi_dict'])
            vois_dct = {"path_out_pet": cl_dct['opth']}
            vois_dct.update(voi_dct)
            with open(fcsv, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(list(vois_dct.keys()))
                csv_writer.writerow(list(vois_dct.values()))

            return {'voi': rvoi, 'outpath': opth}

        else:
            # > get the atlas and GM probability mask in PET space (in UR space) using CL inverse pipeline
            atlgm = atl2pet(
                fatl,
                cl_dct,
                fpet=None, #aligned['ur']['fur'] - this will not work
                outpath=opth)

            if apply_mask=='gm':
                msk = atlgm['fgmpet']
            elif apply_mask=='wm':
                msk = atlgm['fwmpet']
            else:
                msk = None

            rvoi = extract_vois(fdynin, atlgm['fatlpet'], dvoi, atlas_mask=msk,
                                outpath=opth / 'masks', output_masks=True)


            if isinstance(timing, dict) and 'descr' in timing:
                # > timing of all frames
                tdct = dyn_timing(timing)

                # > frame time definitions for NiftyPAD
                dt = tdct['niftypad']

            elif isinstance(timing, dict) and 'timings' in timing:
                dt = timing['niftypad']
            else:
                dt = None

    return {'dt': dt, 'voi': rvoi, 'atlas_gm': atlgm, 'outpath': opth}


# ========================================================================================
def iinorm(cldct, fpet=None, refvoi=None, atlas='hammers', fcomment=None, outpath=None,
           output_masks=True, apply_gmmask=True):
    '''
    Image intensity normalise, `iinorm`.
    Arguments:
    - cldct:    CL dictionary with registration and spatial normalisation output
    - fpet:     The file path of PET image to be intensity normalised; if None
                the centre-of-mass corrected PET will be used from CL output.
    - refvoi:   the indexes of the reference region as defined in the atlas;
                if None, the default cerebellum reference region from Hammers
                atlas will be used.
    - apply_gmmask: if True (default), applies the GM mask to refine the atlas
                and create more accurate reference VOI.
    '''

    # > decipher the CL dictionary
    if len(cldct) == 1:
        cl_dct = cldct[next(iter(cldct))]
    elif 'norm' in cldct:
        cl_dct = cldct
    else:
        raise ValueError('unrecognised CL dictionary')

    # > output path
    if outpath is None:
        if Path(fpet).is_file():
            opth = fpet.parent

    # > reference VOI, if non use cerebellum for Hammers atlas
    if refvoi is None:
        refidx = [17, 18]
    elif isinstance(refvoi, list):
        refidx = refvoi
    else:
        raise ValueError('Unrecognised definition of reference region indexes')

    # > PET modified for centre of mass
    fpetc = cl_dct['petc']['fim']

    # > PET to be intensity normalised
    if not fpet:
        fpet = fpetc

    # > get the atlas in MNI space
    datl = get_atlas(atlas='hammers')

    # > atlas and GM probabilistic mask in the native PET space
    atlgm = atl2pet(fpetc, datl['fatlas'], cl_dct, outpath=opth)

    if apply_gmmask:
        gmmsk = atlgm['fgmpet']
    else:
        gmmsk = None

    # > get the cerebellum GM VOI to act as a reference region
    rvoi = extract_vois(fpetc, atlgm['fatlpet'], {'cerebellum': refidx}, atlas_mask=gmmsk,
                        outpath=opth / 'masks', output_masks=output_masks)

    dpet = nimpa.getnii(fpet, output='all')
    pet = dpet['im']

    # > intensity normalised PET (to be saved)
    ipet = pet / rvoi['cerebellum']['avg']

    if fcomment is None:
        fout = opth / (Path(fpetc).name.split('.nii')[0] + '_intensity_normalised.nii.gz')
    else:
        fout = opth / (Path(fpetc).name.split('.nii')[0] + '_' + fcomment + '.nii.gz')

    # > save new normalised PET
    nimpa.array2nii(ipet, dpet['affine'], fout, trnsp=dpet['transpose'], flip=dpet['flip'])

    return fout
