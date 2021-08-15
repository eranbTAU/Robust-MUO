# Robust Multi-User In-Hand Object Recognition inHuman-RobotCollaboration Using a WearableForce-Myography Device
Applicable human-robot collaboration requires in-tuitive  recognition  of  human  intention  during  shared  work.  Agrasped object such as a tool held by the human provides vitalinformation about the upcoming task. In this paper, we explorethe  use  of  a  wearable  device  to  non-visually  recognize  objectswithin the human hand in various possible grasps. The device isbased on Force-Myography (FMG) where simple and affordableforce  sensors  measure  perturbations  of  forearm  muscles.  Wepropose   a   novel   Deep   Neural-Network   architecture   termedFlip-U-Netinspired  by  the  familiar  U-Net  architecture  usedfor  image  segmentation.  The  Flip-U-Net  is  trained  over  datacollected  from  several  human  participants  and  with  multipleobjects  of  each  class.  Data  is  collected  while  manipulating  theobjects between different grasps and arm postures. The data isalso pre-processed with data augmentation and used to train aVariational Autoencoder for dimensionality reduction mapping.While  prior  work  did  not  provide  a  transferable  FMG-basedmodel, we show that the proposed network can classify objectsgrasped  by  multiple  new  users  without  additional  trainingefforts. Experiment with 12 test participants show classificationaccuracy   of   approximately   95%   over   multiple   grasps   andobjects.  Correlations  between  accuracy  and  various  anthropo-metric measures are also presented. Furthermore, we show thatthe  model  can  be  fine-tuned  to  a  particular  user  based  on  ananthropometric  measure.

![scheme](https://user-images.githubusercontent.com/77546342/129474852-a064b950-1b6a-4733-b957-25bab8618c7c.png)
