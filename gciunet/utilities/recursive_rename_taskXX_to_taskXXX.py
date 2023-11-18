#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from batchgenerators.utilities.file_and_folder_operations import *
import os


def recursive_rename(folder):
    s = subdirs(folder, join=False)
    for ss in s:
        if ss.startswith("Task") and ss.find("_") == 6:
            task_id = int(ss[4:6])
            name = ss[7:]
            os.rename(join(folder, ss), join(folder, "Task%03.0d_" % task_id + name))
    s = subdirs(folder, join=True)
    for ss in s:
        recursive_rename(ss)

if __name__ == "__main__":
    recursive_rename("/media/qiaoqiang/Results/gci_unet")
    recursive_rename("/media/qiaoqiang/gci_unet")
    recursive_rename("/media/qiaoqiang/My Book/MedicalDecathlon")
    recursive_rename("/home/qiaoqiang/drives/datasets/gci_unet_raw")
    recursive_rename("/home/qiaoqiang/drives/datasets/gci_unet_preprocessed")
    recursive_rename("/home/qiaoqiang/drives/datasets/gci_unet_testSets")
    recursive_rename("/home/qiaoqiang/drives/datasets/results/gci_unet")
    recursive_rename("/home/qiaoqiang/drives/e230-dgx2-1-data_qiaoqiang/Decathlon_raw")
    recursive_rename("/home/qiaoqiang/drives/e230-dgx2-1-data_qiaoqiang/gci_unet_preprocessed")

