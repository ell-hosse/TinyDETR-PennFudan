# TinyDETR-PennFudan

A minimal end-to-end **DETR (DEtection TRansformer)** demo on the 170-image **Penn-Fudan Pedestrian** dataset.  
Perfect for understanding a Transformer-based object detector on extremely limited data.

* **Train on GPU:** `python train.py --device cuda`
* **Visualise results:** `python visualize.py --weights checkpoints/best.pth`
* **Files**
  * `config.py` – hyper-parameters in a single place
  * `dataset.py` – Penn-Fudan loader returning COCO-style targets
  * `model.py` – wrapper that builds a torchvision DETR with a custom class head
  * `train.py` – thin training loop with checkpointing
  * `visualize.py` – draws predicted boxes