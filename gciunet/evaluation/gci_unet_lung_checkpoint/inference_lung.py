import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
from sklearn.neighbors import KDTree
from scipy import ndimage


def read_nii(path):
    itk_img=sitk.ReadImage(path)
    spacing=np.array(itk_img.GetSpacing())
    return sitk.GetArrayFromImage(itk_img),spacing

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def process_label(label):
    tumor = label == 1
   
    
    return tumor
'''    
def hd(pred,gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = binary.dc(pred, gt)
        hd95 = binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
'''

def hd(pred,gt):
    #labelPred=sitk.GetImageFromArray(lP.astype(np.float32), isVector=False)
    #labelTrue=sitk.GetImageFromArray(lT.astype(np.float32), isVector=False)
    #hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    #hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    #return hausdorffcomputer.GetAverageHausdorffDistance()
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        print(hd95)
        return  hd95
    else:
        return 0






def rtest():
    path='./'
    label_list=sorted(glob.glob(os.path.join(path,'labelsTs','*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(path,'inferTs','*nii.gz')))
    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice_tumor=[]

    
    hd_tumor=[]

    
    file=path + 'inferTs/'
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file+'/dice_pre.txt', 'w')
    
    for label_path,infer_path in zip(label_list,infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label,spacing= read_nii(label_path)
        infer,spacing= read_nii(infer_path)
        label_tumor=process_label(label)
        infer_tumor=process_label(infer)
        
        Dice_tumor.append(dice(label_tumor,infer_tumor))

        hd_tumor.append(hd(label_tumor,infer_tumor))

        

        #fw.write('*'*20+'\n')
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('Dice_tumor: {:.4f}\n'.format(Dice_tumor[-1]))
        fw.write('hd_tumor: {:.4f}\n'.format(hd_tumor[-1]))
    
        fw.write('*'*20+'\n')
        

    dsc=[]
    dsc.append(np.mean(Dice_tumor))

    avg_hd=[]
    avg_hd.append(np.mean(hd_tumor))

    fw.write('avg_hd:'+str(np.mean(avg_hd))+'\n')

    fw.write('DSC:'+str(np.mean(dsc))+'\n')
    fw.write('HD:'+str(np.mean(avg_hd))+'\n')

    print('done')

if __name__ == '__main__':

    rtest()
