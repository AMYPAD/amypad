"""Centiloid pipeline

Usage:
  centiloid [options] <fpets> <fmris>

Arguments:
  <fpets>  : PET NIfTI directory [default: DirChooser]
  <fmris>  : MRI NIfTI directory [default: DirChooser]

Options:
  --outpath FILE  : Output directory
  --visual  : whether to plot
"""

__author__ = ("Pawel J Markiewicz", "Casper da Costa-Luis")
__copyright__ = "Copyright 2022"

import csv
import logging
import os
import pickle
from pathlib import Path
from typing import Optional
import pickle

import matplotlib.pyplot as plt
import numpy as np
import spm12
from miutil.fdio import hasext, nsort
from niftypet import nimpa

from .utils import cl_anchor_fldr, cl_masks_fldr

log = logging.getLogger(__name__)


#---- DIPY ----
from dipy.data.fetcher import fetch_mni_template, read_mni_template
from dipy.io.image import load_nifti
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric

from dipy.align import _public as align
import nibabel as nib

def fetch_dipy_mni():
    mni_dipy = fetch_mni_template()
    ft1_mni_dipy = [k for k in mni_dipy[0].keys() if 't1' in k and '09c' in k and 'mask' not in k]
    if ft1_mni_dipy:
        return str(Path(mni_dipy[1])/ft1_mni_dipy[0])
    else:
        return None
#---- ----



# ----------------------------------------------------------------------
def load_masks(mskpath, voxsz: int = 2):
    ''' Load the Centiloid PET masks for calculating
        the uptake ratio (UR, aka SUVr) to then convert it to Centiloid.

        Return the paths to the masks and the masks themselves
        Options:
        - voxsz: voxel size used in the normalised PET/MR images
                 and the atlas mask.  Default is 2 mm.
    '''

    voxsz = int(voxsz)

    if voxsz not in [1, 2]:
        raise ValueError('Incorrect voxel size - only 1 and 2 are accepted.')

    log.info('loading CL masks...')
    fmasks = {
        'cg': mskpath / f'voi_CerebGry_{voxsz}mm.nii', 'wc': mskpath / f'voi_WhlCbl_{voxsz}mm.nii',
        'wcb': mskpath / f'voi_WhlCblBrnStm_{voxsz}mm.nii',
        'pns': mskpath / f'voi_Pons_{voxsz}mm.nii', 'ctx': mskpath / f'voi_ctx_{voxsz}mm.nii'}
    masks = {fmsk: nimpa.getnii(fmasks[fmsk]) for fmsk in fmasks}

    return fmasks, masks


def sort_input(fpets, fmris, flip_pet=None):
    ''' Classify input data of PET and MRI and optionally flip PET
        if needed.
        Arguments:
        - fpets:    list or string or Path to PET image(s)
        - fmris:    list or string or Path to MRI image(s)
        - flip_pet: a list of flips (3D tuples) which flip any dimension
                    of the 3D PET image (z,y,x); the list has to have the
                    same length as the lists of `fpets` and `fmris`.
    '''

    if isinstance(fpets, (str, Path)) and isinstance(fmris, (str, Path)):
        # when single PET and MR files are given
        if os.path.isfile(fpets) and os.path.isfile(fmris):
            pet_mr_list = [[fpets], [fmris]]
        # when folder paths are given for PET and MRI files
        # the content is sorted for both PET and MRI and then matched
        # according to the order
        elif os.path.isdir(fpets) and os.path.isdir(fmris):
            fp = nsort(
                os.path.join(fpets, f) for f in os.listdir(fpets) if hasext(f, ('nii', 'nii.gz')))
            fm = nsort(
                os.path.join(fmris, f) for f in os.listdir(fmris) if hasext(f, ('nii', 'nii.gz')))
            pet_mr_list = [fp, fm]
        else:
            raise ValueError('unrecognised or unmatched input for PET and MRI image data')
    elif isinstance(fpets, list) and isinstance(fmris, list):
        if len(fpets) != len(fmris):
            raise ValueError('the number of PET and MRI files have to match!')
        # check if all files exists
        if not all(os.path.isfile(f) and hasext(f, ('nii', 'nii.gz')) for f in fpets) or not all(
                os.path.isfile(f) and hasext(f, ('nii', 'nii.gz')) for f in fmris):
            raise ValueError('the number of paired files do not match')
        pet_mr_list = [fpets, fmris]
    else:
        raise ValueError('unrecognised input image data')

    # -------------------------------------------------------------
    if flip_pet is not None:
        if isinstance(flip_pet, tuple) and len(pet_mr_list[0]) == 1:
            flips = [flip_pet]
        elif isinstance(flip_pet, list) and len(flip_pet) == len(pet_mr_list[0]):
            flips = flip_pet
        else:
            log.warning('the flip definition is not compatible with the list of PET images')
    else:
        flips = [None] * len(pet_mr_list[0])
    # -------------------------------------------------------------

    return pet_mr_list, flips


def run(fpets, fmris, Cnt, tracer='pib', flip_pet=None, bias_corr=True, cmass_corr_pet=True,
        stage='f', standalone=False, voxsz: int = 2, outpath=None, use_stored=False, climage=True, 
        urimage=True, cl_anchor_path: Optional[Path] = None, csv_metrics='short', fcsv=None):
    """
    Process centiloid (CL) using input file lists for PET and MRI
    images, `fpets` and `fmris` (must be in NIfTI format).
    Args:
      outpath: path to the output folder
      bias_corr: if True, applies bias field correction to the MR image
      tracer: specifies what tracer is being used and so the right
              transformation is used; by default it is PiB.  Currently
              [18F]flutemetamol, 'flute', [18F]florbetaben, 'fbb', and
              [18F]florbetapir, 'fbp', are supported.
              IMPORTANT: when calibrating a new tracer, ensure that
              `tracer`='new'.
      standalone: if True, it uses the standalone SPM (with MATLAB Runtime without
              the need of a license); by default it is False, and uses the 
              standard MATLAB.
      stage: processes the data up to:
             (1) registration - both PET and MRI are registered
             to MNI space, `stage`='r'
             (2) normalisation - includes the non-linear MRI
             registration and segmentation, `stage`='n'
             (3) Centiloid process/scaling, `stage`='c'
             (4) Full with visualisation, `stage`='f' (default)
      cmass_corr_pet: correct PET centre of mass if True (default)

      voxsz: voxel size for SPM normalisation writing function
             (output MR and PET images will have this voxel size).
      flip_pet: a list of flips (3D tuples) which flip any dimension
               of the 3D PET image (z,y,x); the list has to have the
               same length as the lists of `fpets` and `fmris`
      use_stored: if True, looks for already saved normalised PET
                images and loads them to avoid processing time;
                works only when `outpath` is provided.
      visual: SPM-based progress visualisations of image registration or
              or image normalisation.
      climage: outputs the CL-converted PET image in the MNI space
      cl_anchor_path: The path where optional CL anchor dictionary is
                saved.
      csv_metrics: output metrics saved to csv:
            - for UR (all reference regions) and CLwc only,`csv_metrics`='short'
            - for SUV, UR, corresponding UR PiB, UR transformations
            and CL ,`csv_metrics`='long'
    """

    if not voxsz in [1, 2]:
        raise ValueError('Voxel size can only be integer and 1 or 2 mm')

    # > the processing stage must be one of registration 'r',
    # > normalisation 'n', CL scaling 'c' or full 'f':
    if stage not in ('r', 'n', 'c', 'f'):
        raise ValueError('unrecognised processing stage')

    # > output dictionary file name
    # > if exists and requested, it will be loaded without computing.
    if outpath is not None:
        fcldct = Path(outpath) / f'CL_output_stage-{stage}.npy'
        if use_stored and fcldct.is_file():
            out = np.load(fcldct, allow_pickle=True)
            out = out.item()
            return out
    else:
        fcldct = None

    # supported F-18 tracers
    f18_amy_tracers = ['fbp', 'fbb', 'flute']

    spm_path = Path(spm12.spm_dir()) # _eng <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # > output dictionary
    out = {}      

    # > MNI template path
    if not standalone:                                  
        tmpl_avg = spm_path/'canonical'/'avg152T1.nii'
    else:
        tmpl_avg = spm12.standalone_path().parent/'spm12_mcr'/'spm12'/'spm12'/'canonical'/'avg152T1.nii'

    pet_mr_list, flips = sort_input(fpets, fmris, flip_pet=flip_pet)

    # > number of PET/MR pairs
    npair = len(pet_mr_list[0])

    # -------------------------------------------------------------
    # > get the CL masks
    fmasks, masks = load_masks(cl_masks_fldr, voxsz=voxsz)
    # -------------------------------------------------------------

    log.info('iterate through all the input data...')

    fi = -1
    for fpet, fmri in zip(*pet_mr_list):

        fi += 1
        # > make the files Path objects
        fpet, fmri = Path(fpet), Path(fmri)
        opth = Path(fpet.parent if outpath is None else outpath)

        # > create output dictionary and folder specific to the PET
        # > file which will be CL quantified; name of the folder
        # > and dictionary output is based on PET name
        onm = fpet.name.rsplit('.nii', 1)[0]

        # > if more than one PET/MR pair, add one more folder level
        if npair>1:
            spth = opth / onm
        else:
            spth = opth

        # > scan level output dictionary
        odct = {'opth': spth, 'fpet': fpet, 'fmri': fmri}

        # >output path for centre of mass alignment and registration
        opthc = spth / 'centre-of-mass'
        opthr = spth / 'registration'
        opthn = spth / 'normalisation'
        optho = spth / 'normalised'
        opths = spth / 'results'
        opthi = opths / 'cl-image'
        opthu = opths / 'ur-image'

        # > find if the normalised PET is already there
        if optho.is_dir():
            fnpets = [f for f in optho.iterdir() if fpet.name.split('.nii')[0] in f.name]
        else:
            fnpets = []

        # run bias field correction unless cancelled
        if bias_corr:
            log.info(f'subject {onm}: running MR bias field correction')
            odct['n4'] = nimpa.bias_field_correction(fmri, executable='sitk', outpath=spth)
            fmri = odct['n4']['fim']

        # > check if flipping the PET is requested
        if flips[fi] is not None and any(flips[fi]):
            flip = flips[fi]
        else:
            flip = None

        if cmass_corr_pet:
            log.info(f'subject {onm}: centre of mass correction')
            # > modify for the centre of mass being at O(0,0,0)
            # > check first if PET is already modified
            tmp = nimpa.getnii(fpet, output='all')
            try:
                dscr = tmp['hdr']['descrip'].item().decode()
            except Exception:
                dscr = None
            if isinstance(dscr, str) and 'CoM-modified' in dscr:
                odct['petc'] = petc = {'fim': fpet}
                log.info('the PET data is already modified for the centre of mass.')
            elif dscr is None and 'com-modified' in fpet.name:
                odct['petc'] = petc = {'fim': fpet}
                log.info('the PET data is already modified for the centre of mass.')
            else:
                odct['petc'] = petc = nimpa.centre_mass_corr(fpet, flip=flip, outpath=opthc)
        else:
            odct['petc'] = petc = {'fim': fpet}
            log.info('PET image has NOT been corrected for the centre of mass.')

        # > centre of mass correction for the MR part
        odct['mric'] = mric = nimpa.centre_mass_corr(fmri, outpath=opthc)

        #-------------- SPM --------------

        log.info(f'subject {onm}: MR registration to MNI space (SPM)')
        odct['reg1'] = reg1 = spm12.coreg_spm(
            tmpl_avg,
            mric['fim'],
            fwhm_ref=0,
            fwhm_flo=Cnt['regpars']['fwhm_t1_mni'],
            outpath=opthr,
            fname_aff="",
            fcomment="",
            pickname="ref",
            costfun=Cnt['regpars']['costfun'],
            graphics=1,
            visual=int(Cnt['regpars']['visual']),
            del_uncmpr=True,
            save_arr=True,
            save_txt=True,
            modify_nii=True,
            standalone=standalone,
        )

        log.info(f'subject {onm}: PET -> MR registration (SPM)')
        odct['reg2'] = reg2 = spm12.coreg_spm(
            reg1['freg'],
            petc['fim'],
            fwhm_ref=Cnt['regpars']['fwhm_t1'],
            fwhm_flo=Cnt['regpars']['fwhm_pet'],
            outpath=opthr,
            fname_aff="",
            fcomment='_mr-reg',
            pickname="ref",
            costfun=Cnt['regpars']['costfun'],
            graphics=1,
            visual=int(Cnt['regpars']['visual']),
            del_uncmpr=True,
            save_arr=True,
            save_txt=True,
            modify_nii=True,
            standalone=standalone,
        )


        if stage == 'r':
            if npair>1:
                out[onm] = odct
            else:
                out = odct
            continue

        #-------------- SPM (non-linear) --------------
        log.info(f'subject {onm}: MR normalisation/segmentation...')
        odct['norm'] = norm = spm12.seg_spm(
            reg1['freg'],
            spm_path,
            outpath=opthn,
            store_nat_gm=Cnt['segpars']['store_nat_gm'],
            store_nat_wm=Cnt['segpars']['store_nat_wm'],
            store_nat_csf=Cnt['segpars']['store_nat_csf'],
            store_fwd=Cnt['segpars']['store_fwd'],
            store_inv=Cnt['segpars']['store_inv'],
            visual=int(Cnt['regpars']['visual']),
            standalone=standalone)

        # > normalise
        list4norm = [
            reg1['freg'],
            reg2['freg']
            ]

        if Cnt['segpars']['store_nat_gm']:
            list4norm.append(odct['norm']['c1'])
        if Cnt['segpars']['store_nat_wm']:
            list4norm.append(odct['norm']['c2'])
        if Cnt['segpars']['store_nat_csf']:
            list4norm.append(odct['norm']['c3'])

        odct['fnorm'] = spm12.normw_spm(
            norm['fordef'],
            list4norm,
            voxsz=float(voxsz),
            outpath=optho,
            standalone=standalone)

        log.info(f'subject {onm}: load normalised PET image...')
        fnpets = [
            f for f in optho.iterdir()
            if fpet.name.split('.nii')[0] in f.name and 'n4bias' not in f.name.lower()]
        # and 'mr' not in f.name.lower()

        if len(fnpets) == 0:
            raise ValueError('could not find normalised PET image files')
        elif len(fnpets) > 1:
            raise ValueError('too many potential normalised PET image files found')

        npet_dct = nimpa.getnii(fnpets[0], output='all')
        npet = npet_dct['im']
        npet[np.isnan(npet)] = 0 # get rid of NaNs


        # > extract mean values and UR
        odct['avgvoi'] = avgvoi = {fmsk: np.mean(npet[masks[fmsk] > 0]) for fmsk in fmasks}
        odct['ur'] = ur = {
            fmsk: avgvoi['ctx'] / avgvoi[fmsk]
            for fmsk in fmasks if fmsk != 'ctx'}

        if stage == 'n' or (tracer!='pib' and tracer not in f18_amy_tracers):
            if npair>1:
                out[onm] = odct
            else:
                out = odct
            continue 

        # **************************************************************
        # C E N T I L O I D   S C A L I N G
        # **************************************************************
        # ---------------------------------
        # > path to anchor point dictionary
        if cl_anchor_path is None:
            cl_fldr = cl_anchor_fldr
        else:
            cl_fldr = Path(cl_anchor_path)
        assert cl_fldr.is_dir()
        # ---------------------------------

        # ---------------------------------
        # > centiloid transformation for PiB
        pth = cl_fldr / 'CL_PiB_anchors.pkl'

        if not os.path.isfile(pth):
            tracer = 'new'
            log.warning('Could not find the PiB CL anchor point definitions/tables')
        else:
            log.info(f'using the following CL anchor table:\n   {pth}')

        with open(pth, 'rb') as f:
            CLA = pickle.load(f)
        # ---------------------------------

        # ---------------------------------
        # > centiloid transformation for PiB
        # > check if UR transformation is needed for F-18 tracers
        if tracer in f18_amy_tracers:
            pth = cl_fldr / f'ur_{tracer}_to_ur_pib__transform.pkl'

            if not os.path.isfile(pth):
                log.warning(
                    'The conversion dictionary/table for the specified tracer is not found')
            else:
                with open(pth, 'rb') as f:
                    CNV = pickle.load(f)
                print('loaded UR (aka SUVr) transformations for tracer:', tracer)

                # > save the PiB converted URs
                odct['ur_pib_calc'] = ur_pib_calc = {
                    fmsk: (ur[fmsk] - CNV[fmsk]['b_std']) / CNV[fmsk]['m_std']
                    for fmsk in fmasks if fmsk != 'ctx'}

                # > save the linear transformation parameters
                odct['ur_pib_calc_transf'] = {
                    fmsk: (CNV[fmsk]['m_std'], CNV[fmsk]['b_std'])
                    for fmsk in fmasks if fmsk != 'ctx'}

            # > used now the new PiB converted URs
            ur = ur_pib_calc
        # ---------------------------------

        if tracer != 'new':
            odct['cl'] = cl = {
                fmsk: 100 * (ur[fmsk] - CLA[fmsk][0]) / (CLA[fmsk][1] - CLA[fmsk][0])
                for fmsk in fmasks if fmsk != 'ctx'}

        # ---------------------------------
        # > save CSV with UR and CL outputs
        if csv_metrics == 'short':
            csv_dict = {
                'path_outputs': odct['opth'],
                **{f'ur_{key}': value
                   for key, value in odct['ur'].items()},
                **{f'cl_{key}': value
                   for key, value in odct['cl'].items() if key == 'wc'}}

        elif csv_metrics == 'long':
            csv_dict = {
                'path_outputs': odct['opth'],
                **{f'suv_{key}': value
                   for key, value in odct['avgvoi'].items()},
                **{f'ur_{key}': value
                   for key, value in odct['ur'].items()},
                **{f'ur_pib_calc_{key}': value
                   for key, value in odct['ur_pib_calc'].items()}, **{
                    f'ur_pib_calc_transf_{key}': value
                    for key, value in odct['ur_pib_calc_transf'].items()},
                **{f'cl_{key}': value
                   for key, value in odct['cl'].items()}}
        elif csv_metrics:
            raise KeyError(f"`csv_metrics`: unknown value ({csv_metrics})")

        if csv_dict:
            if fcsv is not None:
                fcsv = Path(fcsv)
                fcsv.parent.mkdir(parents=True, exist_ok=True)
            else:
                nimpa.create_dir(opths)
                fcsv = opths / 'amypet_outputs.csv'

            append = fcsv.is_file()
            with open(fcsv, 'a' if append else 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_dict.keys())
                if not append:
                    writer.writeheader()
                writer.writerow(csv_dict)

            odct['fcsv'] = fcsv
        # ---------------------------------

        # **************************************************************

        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # > output the CL-converted PET image in the MNI space
        if (climage or urimage) and tracer != 'new':

            for refvoi in fmasks:
                if refvoi == 'ctx':
                    continue

                refavg = odct['avgvoi'][refvoi]

                # > obtain the UR image for the given reference VOI
                npet_ur = npet / refavg

                if urimage:
                    # > save the UR image files
                    nimpa.create_dir(opthu)
                    fout = opthu / ('UR_image_ref-' + refvoi + fnpets[0].name.split('.nii')[0] +
                                    '.nii.gz')
                    nimpa.array2nii(
                        npet_ur, npet_dct['affine'], fout,
                        trnsp=(npet_dct['transpose'].index(0), npet_dct['transpose'].index(1),
                               npet_dct['transpose'].index(2)), flip=npet_dct['flip'])

                # > convert to PiB scale if it is an F18 tracer
                if tracer in f18_amy_tracers:
                    npet_ur = (npet_ur - CNV[refvoi]['b_std']) / CNV[refvoi]['m_std']

                # > convert the (PiB) uptake ratio (UR) image to CL scale
                npet_cl = 100 * (npet_ur - CLA[refvoi][0]) / (CLA[refvoi][1] - CLA[refvoi][0])

                # > get the CL global value by applying the CTX mask
                cl_ = np.mean(npet_cl[masks['ctx'].astype(bool)])

                cl_refvoi = cl[refvoi]

                if climage:
                    if tracer != 'new' and abs(cl_ - cl_refvoi) < 0.25:

                        # > save the CL-converted file
                        nimpa.create_dir(opthi)

                        fout = opthi / ('CL_image_ref-' + refvoi +
                                        fnpets[0].name.split('.nii')[0] + '.nii.gz')
                        nimpa.array2nii(
                            npet_cl, npet_dct['affine'], fout,
                            trnsp=(npet_dct['transpose'].index(0), npet_dct['transpose'].index(1),
                                   npet_dct['transpose'].index(2)), flip=npet_dct['flip'])

                    elif tracer != 'new' and abs(cl_ - cl_refvoi) > 0.25:
                        log.warning(
                            'The CL of CL-converted image is different to the calculated CL'
                            f' (CL_img={cl_:.4f} vs CL={cl_refvoi:.4f}).')
                        log.warning('The CL image has not been generated!')
                    else:
                        log.warning(
                            'The CL image has not been generated due to new tracer being used')
        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

        if not stage == 'f':
            if npair>1:
                out[onm] = odct
            else:
                out = odct
            continue

        # -------------------------------------------------------------
        # VISUALISATION
        # pick masks for visualisation
        msk = 'ctx'
        mskr = 'wc' # 'pons'#

        showpet = nimpa.imsmooth(
            npet.astype(np.float32),
            voxsize=npet_dct['voxsize'],
            fwhm=3.,
            dev_id=False)

        nimpa.create_dir(opths)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # transaxial tiling for QC
        # choose the shape of the mosaic/tiled image
        #izs = np.array([[30, 40, 50], [60, 70, 80]])
        izs = np.array([[60, 80, 100], [120, 140, 160]]) // voxsz

        # get the shape of the mosaic
        shp = izs.shape

        # initialise mosaic for PET and masks
        mscp_t = np.zeros((shp[0], shp[1]) + showpet[0, ...].shape, dtype=np.float32)
        mscm_t = mscp_t.copy()

        # fill in the images
        for i in range(shp[0]):
            for j in range(shp[1]):
                mscp_t[i, j, ...] = showpet[izs[i, j], ...]
                mscm_t[i, j, ...] = masks[msk][izs[i, j], ...] + masks[mskr][izs[i, j], ...]

        # reshape for the final touch
        mscp_t = mscp_t.swapaxes(1, 2)
        mscm_t = mscm_t.swapaxes(1, 2)
        mscp_t = mscp_t.reshape(shp[0] * showpet.shape[1], shp[1] * showpet.shape[2])
        mscm_t = mscm_t.reshape(shp[0] * showpet.shape[1], shp[1] * showpet.shape[2])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Sagittal tiling for QC

        # choose the shape of the mosaic/tiled image
        # ixs = np.array([[20, 30, 40], [50, 60, 70]])
        ixs = np.array([[40, 60, 80], [100, 120, 140]]) // voxsz

        # get the shape of the mosaic
        shp = ixs.shape

        # initialise mosaic for PET and masks
        mscp_s = np.zeros((shp[0], shp[1]) + showpet[..., 0].shape, dtype=np.float32)
        mscm_s = mscp_s.copy()

        # fill in the images
        for i in range(shp[0]):
            for j in range(shp[1]):
                mscp_s[i, j, ...] = showpet[..., ixs[i, j]]
                mscm_s[i, j, ...] = masks[msk][..., ixs[i, j]] + masks[mskr][..., ixs[i, j]]

        # reshape for the final touch
        mscp_s = mscp_s.swapaxes(1, 2)
        mscm_s = mscm_s.swapaxes(1, 2)
        mscp_s = mscp_s.reshape(shp[0] * showpet.shape[0], shp[1] * showpet.shape[1])
        mscm_s = mscm_s.reshape(shp[0] * showpet.shape[0], shp[1] * showpet.shape[1])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        fig, ax = plt.subplots(2, 1, figsize=(9, 12))

        thrsh = 0.9 * showpet.max()

        ax[0].imshow(mscp_t, cmap='magma', vmax=thrsh)
        ax[0].imshow(mscm_t, cmap='gray_r', alpha=0.25)
        ax[0].set_axis_off()
        ax[0].set_title(f'{onm}: transaxial centiloid sampling')

        ax[1].imshow(mscp_s, cmap='magma', vmax=thrsh)
        ax[1].imshow(mscm_s, cmap='gray_r', alpha=0.25)
        ax[1].set_axis_off()
        ax[1].set_title(f'{onm} sagittal centiloid sampling')

        urstr = ",   ".join([
            f"PiB transformed: $UR_{{WC}}=${ur['wc']:.3f}", f"$UR_{{GMC}}=${ur['cg']:.3f}",
            f"$UR_{{CBS}}=${ur['wcb']:.3f}", f"$UR_{{PNS}}=${ur['pns']:.3f}"])
        ax[1].text(0, 380/voxsz, urstr, fontsize=12)

        if tracer != 'new':
            clstr = ",   ".join([
                f"$CL_{{WC}}=${cl['wc']:.1f}", f"$CL_{{GMC}}=${cl['cg']:.1f}",
                f"$CL_{{CBS}}=${cl['wcb']:.1f}", f"$CL_{{PNS}}=${cl['pns']:.1f}"])
            ax[1].text(0, 410/voxsz, clstr, fontsize=12)

        plt.tight_layout()
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)

        fqcpng = opths / f'{onm}_CL-UR-mask_PET_sampling.png'
        plt.savefig(fqcpng, dpi=150, facecolor='auto', edgecolor='auto')
        odct['_amypet_imscroll'] = fig
        plt.close('all')
        odct['fqc'] = fqcpng

        if npair>1:
            out[onm] = odct
        else:
            out = odct


    if fcldct is not None:
        
        np.save(fcldct, out)

    return out
